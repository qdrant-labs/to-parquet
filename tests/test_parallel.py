import random
import uuid

import pyarrow.parquet as pq
import pytest
from conftest import SERVER_URL, unique
from qdrant_client import models

import qdrant_to_parquet.convert as convert
from qdrant_to_parquet.exporter import ClientConfig, export_collection
from qdrant_to_parquet.qdrant import point_id_key, scroll_batches

server = pytest.mark.skipif(not SERVER_URL, reason="needs a Qdrant server (QDRANT_URL)")

IDS = (
    [0, 1, 7, 2**63 - 1, 2**63, 2**64 - 1]
    + [str(uuid.UUID(int=n)) for n in (0, 1, 2**64, 2**127, 2**128 - 1)]
    + [str(uuid.UUID(int=random.Random(0).getrandbits(128) + i)) for i in range(30)]
)


def test_point_id_key_matches_qdrant_order():
    ordered = sorted(IDS, key=point_id_key)
    assert ordered[:6] == [0, 1, 7, 2**63 - 1, 2**63, 2**64 - 1]  # integers first
    uuids = ordered[6:]
    assert [uuid.UUID(u).int for u in uuids] == sorted(uuid.UUID(u).int for u in uuids)
    assert point_id_key(str(uuid.UUID(int=5)).upper()) == point_id_key(
        str(uuid.UUID(int=5))
    )


@pytest.fixture(params=["grpc", "rest"])
def server_config(request):
    if not SERVER_URL:
        pytest.skip("needs a Qdrant server (QDRANT_URL)")
    return ClientConfig(url=SERVER_URL, prefer_grpc=request.param == "grpc")


@pytest.fixture
def mixed_collection(server_config):
    client = server_config.make()
    name = unique("parallel")
    client.create_collection(
        name,
        vectors_config={"v": models.VectorParams(size=3, distance=models.Distance.DOT)},
        sparse_vectors_config={"s": models.SparseVectorParams()},
    )
    rng = random.Random(1)
    points = []
    for i, pid in enumerate(IDS + list(range(100, 2100))):
        payload = {
            "i": i,
            "kind": ["a", "b"][i % 2],
            **({"extra": {"x": i * 0.5}} if i % 3 else {}),
        }
        vector = {"v": [rng.random() for _ in range(3)]}
        if i % 4:
            vector["s"] = models.SparseVector(indices=[i % 50], values=[1.0])
        points.append(models.PointStruct(id=pid, vector=vector, payload=payload))
    client.upsert(name, points, wait=True)
    yield client, name
    client.delete_collection(name)
    client.close()


@server
def test_scroll_range_is_half_open(mixed_collection):
    client, name = mixed_collection
    ordered = sorted(IDS, key=point_id_key)
    start, end = ordered[3], ordered[20]
    got = [
        r.id
        for batch, _ in scroll_batches(client, name, batch_size=4, start=start, end=end)
        for r in batch
    ]
    assert [str(i) for i in got] == [str(i) for i in ordered[3:20]]


@server
@pytest.mark.parametrize("workers", [2, 7, 40])
def test_parallel_export_equals_sequential(
    server_config, mixed_collection, tmp_path, monkeypatch, workers
):
    monkeypatch.setattr(convert, "MAX_ROW_GROUP_ROWS", 500)
    client, name = mixed_collection
    sequential, parallel = tmp_path / "seq.parquet", tmp_path / "par.parquet"
    export_collection(client, name, sequential, batch_size=64, progress=False)
    stats = export_collection(
        client,
        name,
        parallel,
        batch_size=64,
        workers=workers,
        client_config=server_config,
        progress=False,
    )

    assert stats.workers == workers
    total = len(IDS) + 2000
    got = pq.read_table(parallel)
    assert got.num_rows == total
    assert parallel.read_bytes() == sequential.read_bytes()
    meta = pq.ParquetFile(parallel).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [
        500
    ] * 4 + [total % 500]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["par.parquet", "seq.parquet"]


@server
def test_parallel_needs_a_client_config(mixed_collection, tmp_path):
    client, name = mixed_collection
    stats = export_collection(
        client, name, tmp_path / "out.parquet", workers=4, progress=False
    )
    assert stats.workers == 1


@server
def test_parallel_empty_collection(server_config, tmp_path):
    client = server_config.make()
    name = unique("parallel_empty")
    client.create_collection(
        name, vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    try:
        stats = export_collection(
            client,
            name,
            tmp_path / "out.parquet",
            workers=4,
            client_config=server_config,
            progress=False,
        )
        assert (
            stats.points == 0 and pq.read_table(tmp_path / "out.parquet").num_rows == 0
        )
    finally:
        client.delete_collection(name)
        client.close()
