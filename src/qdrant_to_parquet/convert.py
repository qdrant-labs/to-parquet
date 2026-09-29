from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import models

VectorKind = Literal["dense", "multi", "sparse"]

COMPRESSION = "zstd"
# Reading or writing a row group takes several times its size in memory.
ROW_GROUP_BYTES = 32 * 2**20
MAX_ROW_GROUP_ROWS = 10_000

SPARSE_TYPE = pa.struct(
    [("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))]
)


@dataclass(frozen=True)
class VectorSpec:
    name: str  # "" for the unnamed vector
    column: str
    kind: VectorKind
    size: int | None = None
    distance: str | None = None

    @property
    def arrow_type(self) -> pa.DataType:
        # Not fixed_size_list: any point may lack any vector, and pyarrow cannot read
        # fixed_size_list columns with nulls back from Parquet.
        if self.kind == "sparse":
            return SPARSE_TYPE
        if self.kind == "multi":
            return pa.list_(pa.list_(pa.float32()))
        return pa.list_(pa.float32())

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "size": self.size,
            "distance": self.distance,
        }


def vector_specs(params: models.CollectionParams) -> list[VectorSpec]:
    """Sorted by name: REST and gRPC list named vectors in different orders."""

    def dense(name: str, column: str, vp: models.VectorParams) -> VectorSpec:
        kind: VectorKind = "multi" if vp.multivector_config is not None else "dense"
        return VectorSpec(name, column, kind, vp.size, vp.distance.value)

    specs = []
    if isinstance(params.vectors, models.VectorParams):
        specs.append(dense("", "vector", params.vectors))
    elif isinstance(params.vectors, dict):
        specs += [
            dense(name, f"vector_{name}", params.vectors[name])
            for name in sorted(params.vectors)
        ]
    specs += [
        VectorSpec(name, f"vector_{name}", "sparse")
        for name in sorted(params.sparse_vectors or {})
    ]
    return specs


def _vector_value(record: models.Record, spec: VectorSpec) -> Any:
    vector = record.vector
    value = (
        vector.get(spec.name)
        if isinstance(vector, dict)
        else (vector if spec.name == "" else None)
    )
    if spec.kind == "sparse" and value is not None:
        return {"indices": value.indices, "values": value.values}
    return value


class RecordConverter:
    def __init__(self, vectors: Sequence[VectorSpec]) -> None:
        self.vectors = list(vectors)
        self.schema = pa.schema(
            [pa.field("id", pa.string(), nullable=False)]
            + [pa.field(v.column, v.arrow_type) for v in self.vectors]
            + [pa.field("payload", pa.string())]
        )

    def convert(self, records: Sequence[models.Record]) -> pa.Table:
        columns = [pa.array([str(r.id) for r in records], pa.string())]
        columns += [
            pa.array([_vector_value(r, v) for r in records], v.arrow_type)
            for v in self.vectors
        ]
        # Sorted keys: over gRPC, payloads arrive as protobuf maps in no stable order.
        payloads = [
            json.dumps(
                r.payload or {},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for r in records
        ]
        columns.append(pa.array(payloads, pa.string()))
        return pa.Table.from_arrays(columns, schema=self.schema)


class RowGroupWriter:
    """Writes tables as row groups of exactly ``rows`` rows (except the last)."""

    def __init__(self, writer: pq.ParquetWriter, rows: int) -> None:
        self.writer = writer
        self.rows = rows
        self.pending: list[pa.Table] = []
        self.pending_rows = 0

    def add(self, table: pa.Table) -> None:
        self.pending.append(table)
        self.pending_rows += table.num_rows
        if self.pending_rows >= self.rows:
            self.flush()

    def flush(self, final: bool = False) -> None:
        if not self.pending:
            return
        # Contiguous arrays: the writer splits pages by input batch, which would make the bytes depend on chunking.
        table = (
            pa.concat_tables(self.pending)
            .combine_chunks()
            .replace_schema_metadata(self.writer.schema.metadata)
        )
        full = table.num_rows if final else table.num_rows - table.num_rows % self.rows
        if full:
            self.writer.write_table(table.slice(0, full), row_group_size=self.rows)
        rest = table.slice(full)
        self.pending, self.pending_rows = (
            ([rest], rest.num_rows) if rest.num_rows else ([], 0)
        )


def row_group_rows(files: Sequence[Path], sample_rows: int = 1_000) -> int:
    """Rows per row group for about ``ROW_GROUP_BYTES`` of data.

    Measured on the first rows, which don't depend on how the export was chunked, so
    parallel and sequential exports get the same row groups.
    """
    sample: list[pa.Table] = []
    rows = 0
    for path in files:
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=sample_rows - rows):
                sample.append(pa.Table.from_batches([batch]))
                rows += batch.num_rows
                if rows >= sample_rows:
                    break
        if rows >= sample_rows:
            break
    if not rows:
        return MAX_ROW_GROUP_ROWS
    # Serialized size, not Table.nbytes: that counts whole buffers, which vary with how the data was read.
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, sample[0].schema) as stream:
        stream.write_table(pa.concat_tables(sample).combine_chunks())
    return max(1, min(MAX_ROW_GROUP_ROWS, ROW_GROUP_BYTES * rows // sink.tell()))
