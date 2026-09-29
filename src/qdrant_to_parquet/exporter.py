"""Export a Qdrant collection to a Parquet file."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from tqdm import tqdm

from qdrant_to_parquet import __version__
from qdrant_to_parquet.convert import COMPRESSION, RecordConverter, RowGroupWriter, row_group_rows, vector_specs
from qdrant_to_parquet.qdrant import ClientConfig, ExportError, count_points, get_collection, point_id_key
from qdrant_to_parquet.ranges import Plan, RangeProgress, RangeTask, chunk_files, plan_ranges, run_ranges, state_dir

__all__ = ["ClientConfig", "ExportError", "ExportStats", "export_collection"]

log = logging.getLogger(__name__)

AUTO_MAX_WORKERS = 4
AUTO_MIN_POINTS_PER_WORKER = 5_000  # below this, starting workers costs more than it saves


@dataclass
class ExportStats:
    points: int
    output: Path
    bytes_written: int
    seconds: float
    columns: list[str]
    workers: int
    resumed_points: int  # exported by an earlier, interrupted run


def export_collection(
    client: QdrantClient,
    collection: str,
    output: str | os.PathLike[str],
    *,
    batch_size: int = 256,
    retries: int = 3,
    progress: bool = True,
    workers: int | None = 1,
    client_config: ClientConfig | None = None,
    restart: bool = False,
) -> ExportStats:
    """Export all points of ``collection`` (a name or alias) to the Parquet file ``output``.

    An interrupted export to ``output`` is resumed, unless ``restart``. ``output`` is only
    written once all points are exported. With ``workers`` > 1, or None for automatic,
    ranges of points are exported by worker processes connecting with ``client_config``.
    """
    if batch_size <= 0:
        raise ExportError("batch size must be positive")
    if retries < 0:
        raise ExportError("retries must not be negative")
    if workers is not None and workers <= 0:
        raise ExportError("workers must be positive")
    started = time.monotonic()
    output = Path(output)
    if output.is_dir():
        raise ExportError(f"{output} is a directory")

    info = get_collection(client, collection, retries)
    converter = RecordConverter(vector_specs(info.config.params))
    config = info.config.model_dump(mode="json")
    config["metadata"] = config.get("metadata") or None  # REST reports none as null, gRPC as {}
    schema = converter.schema.with_metadata({
        "qdrant.collection": collection,
        # Sorted: over gRPC, named vectors arrive as a protobuf map in no stable order.
        "qdrant.collection_config": json.dumps(config, sort_keys=True),
        "qdrant.vectors": json.dumps({v.column: v.describe() for v in converter.vectors}),
        "qdrant.exporter": f"qdrant-to-parquet {__version__}",
    })
    url = client_config.url if client_config is not None else None
    vectors = [{"column": v.column, **v.describe()} for v in converter.vectors]

    directory = state_dir(output)
    directory.mkdir(parents=True, exist_ok=True)
    with _locked(directory, output):
        if restart:
            for path in directory.iterdir():
                if path.name != "lock":
                    path.unlink()
        try:
            plan = Plan.load(directory)
            if plan is None:
                points_at_start = count_points(client, collection, retries)
                workers = _number_of_workers(workers, client_config, points_at_start)
                plan = Plan(collection, url, vectors, plan_ranges(client, collection, workers, retries), points_at_start)
                plan.save(directory)
            elif (plan.collection, plan.url, plan.vectors) != (collection, url, vectors):
                raise ExportError(f"{directory} holds an interrupted export of another collection. Use --restart")
            else:
                workers = _number_of_workers(workers, client_config, plan.points_at_start)

            progresses = [RangeProgress.load(directory, i, start) for i, (start, _) in enumerate(plan.ranges)]
            resumed_points = sum(p.points for p in progresses)
            if resumed_points:
                log.warning("resuming an interrupted export: %s points were already exported", f"{resumed_points:,}")
            tasks = [
                RangeTask(i, start, end, str(directory), client_config, collection, converter, batch_size, retries)
                for i, (start, end) in enumerate(plan.ranges)
                if not progresses[i].done
            ]
            with tqdm(desc="Exporting", total=plan.points_at_start, initial=resumed_points, unit="pt",
                      disable=not progress, **(_bar_size() if progress else {})) as bar:
                run_ranges(tasks, workers, client, bar)

            points = _join(chunk_files(directory, plan), output, schema, plan.points_at_start,
                           lambda: count_points(client, collection, retries))
        except BaseException:
            if not (directory / "plan.json").exists():  # nothing worth resuming
                shutil.rmtree(directory, ignore_errors=True)
            raise
        shutil.rmtree(directory, ignore_errors=True)

    return ExportStats(
        points=points,
        output=output,
        bytes_written=output.stat().st_size,
        seconds=time.monotonic() - started,
        columns=schema.names,
        workers=min(workers, len(tasks)) if len(tasks) > 1 else 1,
        resumed_points=resumed_points,
    )


def _join(chunks: list[Path], output: Path, schema: pa.Schema, points_at_start: int, count: Callable[[], int]) -> int:
    """Write the chunks to ``output``, checking every point is there exactly once."""
    fd, tmp_name = tempfile.mkstemp(prefix=".qdrant-to-parquet-", suffix=".tmp", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        points = 0
        last = None
        with pq.ParquetWriter(tmp, schema, compression=COMPRESSION) as writer:
            row_groups = RowGroupWriter(writer, row_group_rows(chunks))
            for chunk in chunks:
                with pq.ParquetFile(chunk) as pf:
                    for batch in pf.iter_batches(batch_size=1024, use_threads=False):
                        for point_id in batch.column(0).to_pylist():  # ids strictly increase
                            key = point_id_key(int(point_id) if point_id.isdigit() else point_id)
                            if last is not None and key <= last:
                                raise ExportError(
                                    f"point {point_id} is duplicated or out of order in the saved progress. "
                                    "Use --restart"
                                )
                            last = key
                        row_groups.add(pa.Table.from_batches([batch]))
                        points += batch.num_rows
            row_groups.flush(final=True)

        points_now = count()
        if points != points_now:
            if points_now == points_at_start:
                raise ExportError(f"exported {points:,} points, but the collection has {points_now:,}. Use --restart")
            log.warning(
                "the collection changed during the export (%s points at the start, %s now). Exported %s points",
                f"{points_at_start:,}", f"{points_now:,}", f"{points:,}",
            )

        umask = os.umask(0)  # mkstemp makes the file private. Use the usual permissions
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return points


def _number_of_workers(workers: int | None, client_config: ClientConfig | None, points: int) -> int:
    if client_config is None:
        return 1
    if workers is not None:
        return workers
    workers = min(AUTO_MAX_WORKERS, os.cpu_count() or 1)
    return workers if points >= AUTO_MIN_POINTS_PER_WORKER * workers else 1


def _bar_size() -> dict[str, int]:
    # Some pseudo-terminals report 0x0, and tqdm then draws nothing.
    try:
        size = os.get_terminal_size(sys.stderr.fileno())
    except (OSError, ValueError):
        size = os.terminal_size((0, 0))
    return {} if size.columns and size.lines else {"ncols": 80, "nrows": 24}


@contextmanager
def _locked(directory: Path, output: Path) -> Iterator[None]:
    """An exclusive lock on the export's state (released if the process dies)."""
    try:
        import fcntl
    except ImportError:  # Windows
        yield
        return
    with open(directory / "lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExportError(f"another export to {output} is running") from None
        yield
