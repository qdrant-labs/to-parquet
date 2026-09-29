import os
import uuid

import pytest
from qdrant_client import QdrantClient, models

SERVER_URL = os.environ.get("QDRANT_URL")

BACKENDS = ["local"] + (["rest", "grpc"] if SERVER_URL else [])


@pytest.fixture(params=BACKENDS)
def client(request):
    """A Qdrant client: in-process local mode, plus a real server when QDRANT_URL is set."""
    if request.param == "local":
        c = QdrantClient(location=":memory:")
    else:
        c = QdrantClient(url=SERVER_URL, prefer_grpc=request.param == "grpc")
    yield c
    if request.param != "local":
        for collection in c.get_collections().collections:
            if collection.name.startswith("test_"):
                c.delete_collection(collection.name)
    c.close()


def unique(name: str) -> str:
    return f"test_{name}_{uuid.uuid4().hex[:8]}"


UUID_ID = "8f5c1b2e-6a4d-4c3b-9e7f-1a2b3c4d5e6f"


# The collections use non-cosine distances, so Qdrant stores the vectors unchanged.


def make_simple(client: QdrantClient, n: int = 25) -> str:
    """One unnamed 4-dim vector. Integer and UUID ids."""
    name = unique("simple")
    client.create_collection(
        name, vectors_config=models.VectorParams(size=4, distance=models.Distance.EUCLID)
    )
    points = [
        models.PointStruct(
            id=i,
            vector=[float(i), 1.0, 2.0, 3.0],
            payload={"n": i, "even": i % 2 == 0, "title": f"doc {i}", "tags": ["a", str(i)]},
        )
        for i in range(1, n)
    ]
    points.append(models.PointStruct(id=UUID_ID, vector=[0.5, 0.5, 0.5, 0.5], payload=None))
    client.upsert(name, points, wait=True)
    return name


def make_multi(client: QdrantClient) -> str:
    """A named dense, a multi-vector and a sparse vector, not all present on every point."""
    name = unique("multi")
    client.create_collection(
        name,
        vectors_config={
            "text": models.VectorParams(size=3, distance=models.Distance.DOT),
            "colbert": models.VectorParams(
                size=2,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )
    client.upsert(
        name,
        [
            models.PointStruct(
                id=1,
                vector={
                    "text": [0.1, 0.2, 0.3],
                    "colbert": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                    "bm25": models.SparseVector(indices=[3, 17], values=[0.5, 1.5]),
                },
                payload={"lang": "en"},
            ),
            models.PointStruct(
                id=2,
                vector={"text": [0.4, 0.5, 0.6]},
                payload={"lang": "de"},
            ),
            models.PointStruct(id=3, vector={"colbert": [[2.0, 3.0]]}, payload={"lang": "fr"}),
        ],
        wait=True,
    )
    return name
