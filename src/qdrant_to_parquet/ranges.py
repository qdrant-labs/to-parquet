from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import time
import warnings
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from qdrant_to_parquet.convert import COMPRESSION, RecordConverter
from qdrant_to_parquet.qdrant import (
    ClientConfig,
    ExportError,
    call_with_retries,
    describe_error,
    point_id_key,
    scroll_batches,
)

log = logging.getLogger(__name__)

PLAN_VERSION = 1
SAMPLES_PER_WORKER = 10  # more, smaller ranges balance the load between workers
# Progress is saved every few seconds, so an interruption loses at most that much work.
CHUNK_SECONDS = 5
# Also saved when a chunk reaches this size, which caps the memory a worker holds before writing.
CHUNK_BYTES = 32 * 2**20


def state_dir(output: Path) -> Path:
    return output.parent / f".{output.name}.export"


def _write_json(path: Path, data: Any) -> None:
    """Replace ``path`` atomically and durably."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass
class Plan:
    collection: str
    url: str | None
    vectors: list[dict[str, Any]]
    ranges: list[list[Any]]  # [start, end]
    points_at_start: int
    version: int = PLAN_VERSION

    @classmethod
    def load(cls, directory: Path) -> Plan | None:
        try:
            data = json.loads((directory / "plan.json").read_text())
        except FileNotFoundError:
            return None
        except ValueError as exc:
            raise ExportError(
                f"the saved progress in {directory} is unreadable ({exc}). Use --restart"
            ) from None
        if data.get("version") != PLAN_VERSION:
            raise ExportError(
                f"the saved progress in {directory} is from another version. Use --restart"
            )
        return cls(**data)

    def save(self, directory: Path) -> None:
        _write_json(directory / "plan.json", asdict(self))


@dataclass
class RangeProgress:
    offset: Any  # where the next scroll starts
    points: int = 0
    chunks: list[str] = field(default_factory=list)
    done: bool = False

    @staticmethod
    def path(directory: Path, index: int) -> Path:
        return directory / f"range-{index:05d}.json"

    @classmethod
    def load(cls, directory: Path, index: int, start: Any) -> RangeProgress:
        try:
            return cls(**json.loads(cls.path(directory, index).read_text()))
        except FileNotFoundError:
            return cls(offset=start)

    def save(self, directory: Path, index: int) -> None:
        _write_json(self.path(directory, index), asdict(self))


def saved_points(output: Path) -> int | None:
    """Points saved by an interrupted export to ``output``. None if there is none."""
    directory = state_dir(output)
    plan = Plan.load(directory) if directory.is_dir() else None
    if plan is None:
        return None
    return sum(
        RangeProgress.load(directory, i, None).points for i in range(len(plan.ranges))
    )


def plan_ranges(
    client: QdrantClient, collection: str, workers: int, retries: int
) -> list[list[Any]]:
    if workers <= 1:
        return [[None, None]]
    try:
        points = call_with_retries(
            lambda: client.query_points(
                collection,
                query=models.SampleQuery(sample=models.Sample.RANDOM),
                limit=workers * SAMPLES_PER_WORKER,
                with_payload=False,
                with_vectors=False,
            ),
            retries=retries,
            what="sample point ids",
        ).points
    except Exception as exc:
        log.warning(
            "cannot sample point ids for a parallel export (%s). Using one worker",
            describe_error(exc),
        )
        return [[None, None]]
    boundaries = {point_id_key(p.id): p.id for p in points}
    bounds = [None, *(boundaries[k] for k in sorted(boundaries)), None]
    return [[start, end] for start, end in zip(bounds[:-1], bounds[1:], strict=True)]


def chunk_files(directory: Path, plan: Plan) -> list[Path]:
    """All chunks, in id order."""
    files = []
    for index, (start, _) in enumerate(plan.ranges):
        progress = RangeProgress.load(directory, index, start)
        if not progress.done:
            raise ExportError(f"range {index} is not finished")
        files += [directory / name for name in progress.chunks]
    return files


@dataclass
class RangeTask:
    index: int
    start: Any
    end: Any
    directory: str
    client_config: ClientConfig | None
    collection: str
    converter: RecordConverter
    batch_size: int
    retries: int


def run_range(
    task: RangeTask, report: Callable[[int], None], client: QdrantClient | None = None
) -> None:
    """Export a range to chunk files, saving its progress after each chunk."""
    directory = Path(task.directory)
    progress = RangeProgress.load(directory, task.index, task.start)
    if progress.done:
        return
    prefix = f"range-{task.index:05d}-"
    for stray in directory.glob(prefix + "*"):  # written after the last saved progress
        if stray.name not in progress.chunks:
            stray.unlink()

    own_client = client is None
    if own_client:
        client = task.client_config.make(
            check_compatibility=False
        )  # the main process checked it
    try:
        pending: list[pa.Table] = []
        rows = size = 0
        since = time.monotonic()

        def commit(next_offset: Any, done: bool) -> None:
            nonlocal pending, rows, size, since
            if pending:
                name = f"{prefix}{len(progress.chunks):05d}.parquet"
                tmp = directory / (name + ".tmp")
                with pq.ParquetWriter(
                    tmp, task.converter.schema, compression=COMPRESSION
                ) as writer:
                    writer.write_table(pa.concat_tables(pending))
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, directory / name)
                progress.chunks.append(name)
            progress.points += rows
            progress.offset, progress.done = next_offset, done
            progress.save(directory, task.index)  # a chunk counts once this is saved
            pending, rows, size, since = [], 0, 0, time.monotonic()

        for records, next_offset in scroll_batches(
            client,
            task.collection,
            batch_size=task.batch_size,
            retries=task.retries,
            start=progress.offset,
            end=task.end,
        ):
            table = task.converter.convert(records)
            pending.append(table)
            rows += table.num_rows
            size += table.nbytes
            report(table.num_rows)
            if time.monotonic() - since >= CHUNK_SECONDS or size >= CHUNK_BYTES:
                commit(next_offset, done=next_offset is None)
        commit(None, done=True)
    finally:
        if own_client:
            client.close()


_progress = None


def _init_worker(progress, log_level: int) -> None:
    global _progress
    _progress = progress
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process handles Ctrl-C
    logging.addLevelName(logging.WARNING, "warning")
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logging.getLogger("grpc").setLevel(logging.CRITICAL)
    warnings.filterwarnings(
        "ignore", message="Api key is used with an insecure connection"
    )  # shown once already


def _report_to_parent(n: int) -> None:
    with _progress.get_lock():
        _progress.value += n


def _run_range_in_worker(task: RangeTask) -> None:
    try:
        run_range(task, _report_to_parent)
    except Exception as exc:
        raise ExportError(describe_error(exc)) from None  # gRPC errors can't be pickled


def run_ranges(tasks: list[RangeTask], workers: int, client: QdrantClient, bar) -> None:
    """Export the ranges, in worker processes if there are several workers and ranges."""
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            run_range(task, bar.update, client)
        return

    # Processes, not threads: decoding responses is CPU-bound Python. Spawned, because
    # forking breaks gRPC. A ProcessPoolExecutor fails when a worker dies, where
    # multiprocessing.Pool would wait forever.
    ctx = multiprocessing.get_context("spawn")
    progress = ctx.Value("q", 0)
    executor = ProcessPoolExecutor(
        min(workers, len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(progress, logging.getLogger().level),
    )
    try:
        futures = [executor.submit(_run_range_in_worker, task) for task in tasks]
        shown = 0
        while True:
            done, pending = wait(futures, timeout=0.2, return_when=FIRST_EXCEPTION)
            for future in done:
                error = future.exception()
                if isinstance(error, BrokenProcessPool):
                    raise ExportError(
                        "a worker process died unexpectedly (out of memory? killed?)"
                    ) from error
                if error is not None:
                    raise error
            bar.update(progress.value - shown)
            shown = progress.value
            if not pending:
                return
    except BaseException:
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()  # running tasks can't be cancelled
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
