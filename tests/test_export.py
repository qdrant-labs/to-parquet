import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import UUID_ID, make_multi, make_simple, unique
from qdrant_client import models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import UnexpectedResponse

import qdrant_to_parquet.convert as convert
from qdrant_to_parquet.exporter import ClientConfig, ExportError, export_collection
from qdrant_to_parquet.qdrant import describe_error


def rows_by_id(table):
    return {row["id"]: row for row in table.to_pylist()}


def flaky(client, method, monkeypatch, fail_on, error):
    """Make ``client.method`` raise ``error`` on the calls numbered in ``fail_on`` (1-based)."""
    original = getattr(client, method)
    calls = 0

    def wrapper(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls in fail_on:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(client, method, wrapper)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("qdrant_to_parquet.qdrant.time.sleep", sleeps.append)
    return sleeps


def test_unnamed_vector_and_payload(client, tmp_path):
    name = make_simple(client)
    out = tmp_path / "out.parquet"

    stats = export_collection(client, name, out, batch_size=7, progress=False)

    table = pq.read_table(out)
    assert stats.points == table.num_rows == 25
    assert table.schema.names == ["id", "vector", "payload"]
    assert table.schema.field("vector").type == pa.list_(pa.float32())

    rows = rows_by_id(table)
    assert set(rows) == {str(i) for i in range(1, 25)} | {UUID_ID}
    assert rows["3"]["vector"] == pytest.approx([3.0, 1.0, 2.0, 3.0])
    assert (
        rows["3"]["payload"] == '{"even":false,"n":3,"tags":["a","3"],"title":"doc 3"}'
    )  # sorted keys
    assert json.loads(rows[UUID_ID]["payload"]) == {}

    meta = table.schema.metadata
    assert meta[b"qdrant.collection"].decode() == name
    assert (
        json.loads(meta[b"qdrant.collection_config"])["params"]["vectors"]["size"] == 4
    )
    assert json.loads(meta[b"qdrant.vectors"])["vector"] == {
        "name": "",
        "kind": "dense",
        "size": 4,
        "distance": "Euclid",
    }
    assert list(tmp_path.iterdir()) == [
        out
    ]  # the temporary file was renamed into place


def test_named_multi_and_sparse_vectors(client, tmp_path):
    name = make_multi(client)
    out = tmp_path / "out.parquet"

    export_collection(client, name, out, progress=False)

    table = pq.read_table(out)
    assert table.schema.names == [
        "id",
        "vector_colbert",
        "vector_text",
        "vector_bm25",
        "payload",
    ]
    assert table.schema.field("vector_colbert").type == pa.list_(pa.list_(pa.float32()))
    assert table.schema.field("vector_bm25").type == pa.struct(
        [("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))]
    )
    rows = rows_by_id(table)
    assert rows["1"]["vector_text"] == pytest.approx([0.1, 0.2, 0.3])
    assert [pytest.approx(v) for v in rows["1"]["vector_colbert"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]
    assert rows["1"]["vector_bm25"]["indices"] == [3, 17]
    assert rows["1"]["vector_bm25"]["values"] == pytest.approx([0.5, 1.5])
    # Named vectors are optional per point.
    assert rows["2"]["vector_colbert"] is None and rows["2"]["vector_bm25"] is None
    assert rows["3"]["vector_text"] is None
    assert rows["3"]["vector_colbert"] == [[2.0, 3.0]]
    assert (
        json.loads(table.schema.metadata[b"qdrant.vectors"])["vector_colbert"]["size"]
        == 2
    )


def test_row_groups(client, tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "MAX_ROW_GROUP_ROWS", 10)
    name = make_simple(client)
    out = tmp_path / "out.parquet"

    export_collection(client, name, out, batch_size=7, progress=False)

    meta = pq.ParquetFile(out).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [
        10,
        10,
        5,
    ]


def test_empty_collection(client, tmp_path):
    name = unique("empty")
    client.create_collection(
        name, vectors_config=models.VectorParams(size=8, distance=models.Distance.DOT)
    )
    out = tmp_path / "out.parquet"

    assert export_collection(client, name, out, progress=False).points == 0

    table = pq.read_table(out)
    assert table.num_rows == 0
    assert table.schema.names == ["id", "vector", "payload"]


def test_missing_collection(client, tmp_path):
    with pytest.raises(ExportError, match="does not exist"):
        export_collection(
            client, "test_missing_nope", tmp_path / "out.parquet", progress=False
        )
    assert not list(tmp_path.iterdir())


def test_failure_leaves_no_partial_file(client, tmp_path, monkeypatch):
    name = make_simple(client)
    flaky(
        client, "scroll", monkeypatch, fail_on={2}, error=ValueError("boom")
    )  # not retryable

    with pytest.raises(ValueError, match="boom"):
        export_collection(
            client, name, tmp_path / "out.parquet", batch_size=5, progress=False
        )
    assert [p.name for p in tmp_path.iterdir()] == [
        ".out.parquet.export"
    ]  # only the saved progress


def test_transient_errors_are_retried(client, tmp_path, monkeypatch, no_sleep):
    name = make_simple(client)
    flaky(
        client,
        "scroll",
        monkeypatch,
        fail_on={2, 3},
        error=ConnectionError("temporary"),
    )

    assert (
        export_collection(
            client, name, tmp_path / "out.parquet", batch_size=5, progress=False
        ).points
        == 25
    )
    assert no_sleep == [1, 2]


def test_rate_limiting_waits_as_long_as_the_server_asks(
    client, tmp_path, monkeypatch, no_sleep
):
    name = make_simple(client)
    # Five times in a row: rate limiting does not use up the retries.
    flaky(
        client,
        "scroll",
        monkeypatch,
        fail_on={2, 3, 4, 5, 6},
        error=ResourceExhaustedResponse("Rate limiting exceeded", retry_after_s=7),
    )

    stats = export_collection(
        client, name, tmp_path / "out.parquet", batch_size=5, retries=1, progress=False
    )
    assert stats.points == 25
    assert no_sleep == [7] * 5


def test_rate_limiting_gives_up_eventually(client, tmp_path, monkeypatch, no_sleep):
    name = make_simple(client)
    flaky(
        client,
        "scroll",
        monkeypatch,
        fail_on=range(1, 10**6),
        error=ResourceExhaustedResponse("Rate limiting exceeded", retry_after_s=30),
    )

    with pytest.raises(
        ExportError, match="scroll still rate limited after waiting 600s"
    ):
        export_collection(client, name, tmp_path / "out.parquet", progress=False)
    assert not (tmp_path / "out.parquet").exists()


@pytest.mark.parametrize("method", ["get_collection", "count"])
def test_every_request_is_retried(client, tmp_path, monkeypatch, no_sleep, method):
    name = make_simple(client)
    flaky(
        client,
        method,
        monkeypatch,
        fail_on={1},
        error=ResourceExhaustedResponse("Rate limiting exceeded", 1),
    )

    # progress=True counts the points for the progress bar
    assert (
        export_collection(client, name, tmp_path / "out.parquet", progress=True).points
        == 25
    )
    assert no_sleep == [1]


def test_retry_warnings_are_single_lines(
    client, tmp_path, monkeypatch, no_sleep, caplog
):
    name = make_simple(client)
    html = b"<html>\n<body>\n" + b"<p>Bad gateway</p>\n" * 500 + b"</body>\n</html>"
    flaky(
        client,
        "scroll",
        monkeypatch,
        fail_on={1},
        error=UnexpectedResponse(502, "Bad Gateway", html, headers=None),
    )
    flaky(
        client,
        "count",
        monkeypatch,
        fail_on={1},
        error=ResourceExhaustedResponse("Rate\nlimited", 1),
    )

    with caplog.at_level(logging.WARNING):
        export_collection(client, name, tmp_path / "out.parquet", progress=True)

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert all("\n" not in m and len(m) < 600 for m in messages)
    assert messages[0].startswith("count rate limited (")
    assert messages[1].startswith(
        "scroll failed (Qdrant returned HTTP 502 Bad Gateway: <html> <body> <p>Bad gateway"
    )


def test_describe_error():
    assert describe_error(
        UnexpectedResponse(503, "Service Unavailable", b"", headers=None)
    ) == ("Qdrant returned HTTP 503 Service Unavailable")
    assert describe_error(
        UnexpectedResponse(502, "Bad Gateway", None, headers=None)
    ) == ("Qdrant returned HTTP 502 Bad Gateway")
    long = describe_error(ExportError("x" * 1000))
    assert len(long) == 500 and long.endswith("...")
    assert describe_error(ValueError("a\n  b")) == "ValueError: a b"


def test_client_config_makes_clients():
    # check_compatibility is an argument of QdrantClient since qdrant-client 1.13
    for check in (True, False):
        ClientConfig(url="http://localhost:1", prefer_grpc=False).make(
            check_compatibility=check
        ).close()


def test_row_group_size_does_not_depend_on_chunking(tmp_path):
    rows = 3_000
    table = pa.table(
        {
            "id": [str(i) for i in range(rows)],
            "vector": pa.array(
                [[float(i)] * 3072 if i % 7 else None for i in range(rows)],
                pa.list_(pa.float32()),
            ),
            "payload": ['{"text":"%s"}' % ("x" * (i % 300)) for i in range(rows)],
        }
    )
    layouts = {
        "one file": [rows],
        "even": [1_000] * 3,
        "uneven": [7, 300, 993, 1, 1_699],
    }
    results = {}
    for name, sizes in layouts.items():
        files, start = [], 0
        for i, n in enumerate(sizes):
            path = tmp_path / f"{name}-{i}.parquet"
            pq.write_table(table.slice(start, n), path, row_group_size=max(1, n // 2))
            files.append(path)
            start += n
        results[name] = convert.row_group_rows(files)
    assert len(set(results.values())) == 1, results
    assert 2_500 < results["one file"] < 4_000  # ~32 MB of ~10 KB rows


def test_output_is_reproducible(client, tmp_path):
    name = make_multi(client)
    first, second = tmp_path / "1.parquet", tmp_path / "2.parquet"
    export_collection(client, name, first, progress=False)
    export_collection(client, name, second, progress=False)
    assert pq.read_schema(first).metadata == pq.read_schema(second).metadata
    config = json.loads(pq.read_schema(first).metadata[b"qdrant.collection_config"])
    assert list(config["params"]["vectors"]) == sorted(config["params"]["vectors"])
