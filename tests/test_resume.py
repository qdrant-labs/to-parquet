import json
import signal
import subprocess
import sys
import time

import pyarrow.parquet as pq
import pytest
from conftest import SERVER_URL, make_simple, unique
from qdrant_client import models

import qdrant_to_parquet.ranges as ranges
from qdrant_to_parquet.exporter import ClientConfig, ExportError, export_collection


class Crash(Exception):
    pass


def crash_after(client, monkeypatch, pages):
    """Make the export stop (like a crash) after ``pages`` successful scroll requests. Returns the call log."""
    original = client.scroll
    offsets = []

    def scroll(*args, **kwargs):
        offsets.append(kwargs.get("offset"))
        if len(offsets) > pages:
            raise Crash
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "scroll", scroll)
    return offsets


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(
        ranges, "CHUNK_SECONDS", 0
    )  # progress is saved after every page


def test_resume_continues_from_the_saved_offset(
    client, tmp_path, monkeypatch, small_chunks
):
    name = make_simple(client)
    expected, out = tmp_path / "expected.parquet", tmp_path / "out.parquet"
    export_collection(client, name, expected, progress=False)

    with monkeypatch.context() as m:
        crash_after(client, m, pages=3)
        with pytest.raises(Crash):
            export_collection(client, name, out, batch_size=4, progress=False)
    assert not out.exists()
    assert ranges.saved_points(out) == 12  # 3 pages of 4 points

    offsets = crash_after(client, monkeypatch, pages=100)
    stats = export_collection(client, name, out, batch_size=4, progress=False)

    assert (
        offsets[0] == 13
    )  # continued where the saved progress ended, not from the start
    assert stats.resumed_points == 12 and stats.points == 25
    assert pq.read_table(out).equals(pq.read_table(expected), check_metadata=True)
    assert not ranges.state_dir(out).exists()


@pytest.mark.parametrize(
    ("seconds", "size", "saved"),
    [(0, 2**40, 12), (10**9, 1, 12), (10**9, 2**40, 0)],
    ids=["after some seconds", "at a size", "neither"],
)
def test_when_progress_is_saved(client, tmp_path, monkeypatch, seconds, size, saved):
    monkeypatch.setattr(ranges, "CHUNK_SECONDS", seconds)
    monkeypatch.setattr(ranges, "CHUNK_BYTES", size)
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    crash_after(client, monkeypatch, pages=3)
    with pytest.raises(Crash):
        export_collection(client, name, out, batch_size=4, progress=False)
    assert (ranges.saved_points(out) or 0) == saved


def test_stray_chunks_are_ignored(client, tmp_path, monkeypatch, small_chunks):
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    with monkeypatch.context() as m:
        crash_after(client, m, pages=2)
        with pytest.raises(Crash):
            export_collection(client, name, out, batch_size=5, progress=False)
    # A chunk written after the last saved progress (the export stopped in between).
    directory = ranges.state_dir(out)
    saved = json.loads((directory / "range-00000.json").read_text())["chunks"]
    stray = directory / f"range-00000-{len(saved):05d}.parquet"
    stray.write_bytes((directory / saved[0]).read_bytes())

    stats = export_collection(client, name, out, progress=False)
    ids = pq.read_table(out)["id"].to_pylist()
    assert stats.points == len(ids) == len(set(ids)) == 25


def test_inconsistent_saved_progress_is_detected(
    client, tmp_path, monkeypatch, small_chunks
):
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    with monkeypatch.context() as m:
        crash_after(client, m, pages=2)
        with pytest.raises(Crash):
            export_collection(client, name, out, batch_size=5, progress=False)
    # A saved chunk is replaced by a copy of another one: duplicates and a gap, same number of rows.
    directory = ranges.state_dir(out)
    (directory / "range-00000-00001.parquet").write_bytes(
        (directory / "range-00000-00000.parquet").read_bytes()
    )

    with pytest.raises(ExportError, match="point 1 is duplicated or out of order"):
        export_collection(client, name, out, progress=False)
    assert not out.exists()


def test_restart_discards_saved_progress(client, tmp_path, monkeypatch, small_chunks):
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    with monkeypatch.context() as m:
        crash_after(client, m, pages=2)
        with pytest.raises(Crash):
            export_collection(client, name, out, batch_size=5, progress=False)

    offsets = crash_after(client, monkeypatch, pages=100)
    stats = export_collection(
        client, name, out, batch_size=5, progress=False, restart=True
    )
    assert offsets[0] is None and stats.resumed_points == 0 and stats.points == 25


def test_saved_progress_of_another_collection(
    client, tmp_path, monkeypatch, small_chunks
):
    first, second = make_simple(client), make_simple(client)
    out = tmp_path / "out.parquet"
    with monkeypatch.context() as m:
        crash_after(client, m, pages=2)
        with pytest.raises(Crash):
            export_collection(client, first, out, batch_size=5, progress=False)

    with pytest.raises(ExportError, match="interrupted export of another collection"):
        export_collection(client, second, out, progress=False)
    assert (
        export_collection(client, second, out, progress=False, restart=True).points
        == 25
    )


def test_one_export_per_output_at_a_time(client, tmp_path):
    fcntl = pytest.importorskip("fcntl")
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    directory = ranges.state_dir(out)
    directory.mkdir()
    with open(directory / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ExportError, match="another export to .* is running"):
            export_collection(client, name, out, progress=False)
        with pytest.raises(ExportError, match="another export"):
            export_collection(
                client, name, out, progress=False, restart=True
            )  # does not delete anything
    assert export_collection(client, name, out, progress=False).points == 25


def test_count_mismatch_is_an_error(client, tmp_path, monkeypatch, caplog):
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    original = client.count
    monkeypatch.setattr(
        client,
        "count",
        lambda *a, **kw: models.CountResult(count=original(*a, **kw).count + 1),
    )

    with pytest.raises(
        ExportError, match="exported 25 points, but the collection has 26"
    ):
        export_collection(client, name, out, progress=False)
    assert not out.exists()


def test_collection_changed_during_the_export(client, tmp_path, monkeypatch, caplog):
    name = make_simple(client)
    out = tmp_path / "out.parquet"
    counts = iter([25, 30])  # 5 points were added during the export
    monkeypatch.setattr(
        client, "count", lambda *a, **kw: models.CountResult(count=next(counts))
    )

    assert export_collection(client, name, out, progress=False).points == 25
    assert (
        "the collection changed during the export (25 points at the start, 30 now). Exported 25"
        in caplog.text
    )


@pytest.mark.skipif(not SERVER_URL, reason="needs a Qdrant server (QDRANT_URL)")
@pytest.mark.parametrize("transport", [[], ["--rest"]])
def test_interrupted_parallel_export_resumes(tmp_path, transport):
    """The CLI is killed during a parallel export. Running it again finishes it, identical to a clean export."""
    # The reference export uses the same transport: older servers report slightly different
    # collection configurations over gRPC and REST (e.g. Qdrant 1.13 omits wal_retain_closed).
    config = ClientConfig(url=SERVER_URL, prefer_grpc="--rest" not in transport)
    client = config.make()
    name = unique("resume")
    client.create_collection(
        name, vectors_config=models.VectorParams(size=16, distance=models.Distance.DOT)
    )
    client.upload_collection(
        name,
        vectors=[[float(i % 97)] * 16 for i in range(30_000)],
        payload=({"i": i} for i in range(30_000)),
        ids=range(30_000),
        wait=True,
    )
    try:
        expected, out = tmp_path / "expected.parquet", tmp_path / "out.parquet"
        export_collection(client, name, expected, progress=False)

        cli = [
            sys.executable,
            "-m",
            "qdrant_to_parquet",
            name,
            str(out),
            "--url",
            SERVER_URL,
            *transport,
        ]
        # One point per request, so the export is still running when it is interrupted.
        proc = subprocess.Popen(
            [*cli, "--workers", "4", "--batch-size", "1"],
            stderr=subprocess.PIPE,
            text=True,
        )
        directory = ranges.state_dir(out)
        while (ranges.saved_points(out) or 0) < 5_000:
            assert proc.poll() is None, (
                "the export finished before it could be interrupted"
            )
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 143, err
        assert "progress is saved" in err
        saved = ranges.saved_points(out)
        assert 5_000 <= saved < 30_000 and not out.exists()
        plan = json.loads((directory / "plan.json").read_text())
        assert len(plan["ranges"]) > 4

        result = subprocess.run(
            [*cli, "--workers", "2"], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert "resuming an interrupted export" in result.stderr
        assert out.read_bytes() == expected.read_bytes()
        assert not directory.exists()
    finally:
        client.delete_collection(name)
        client.close()
