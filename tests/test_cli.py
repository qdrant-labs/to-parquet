import warnings

import grpc
import pyarrow.parquet as pq
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from conftest import make_simple
from qdrant_to_parquet import cli
from qdrant_to_parquet.cli import main
from qdrant_to_parquet.exporter import ClientConfig


@pytest.fixture
def collection(tmp_path, monkeypatch):
    """The CLI talks to a qdrant-client local-mode storage instead of a server."""
    path = str(tmp_path / "storage")
    client = QdrantClient(path=path)
    name = make_simple(client)
    client.close()
    monkeypatch.setattr(ClientConfig, "make", lambda self, check_compatibility=True: QdrantClient(path=path))
    return name


def test_export(collection, tmp_path, capsys):
    out = tmp_path / "out.parquet"

    assert main([collection, str(out)]) == 0

    assert pq.read_table(out).num_rows == 25
    err = capsys.readouterr().err
    assert "Exported 25 points" in err and "Columns: id, vector, payload" in err


def test_quiet(collection, tmp_path, capsys):
    filters = list(warnings.filters)
    assert main([collection, str(tmp_path / "out.parquet"), "-q"]) == 0
    assert capsys.readouterr().err == ""
    assert warnings.filters == filters  # restored


def test_refuses_to_overwrite(collection, tmp_path):
    out = tmp_path / "out.parquet"
    out.write_text("keep me")

    with pytest.raises(SystemExit) as exc:
        main([collection, str(out)])
    assert exc.value.code == 2
    assert out.read_text() == "keep me"

    assert main([collection, str(out), "--force", "-q"]) == 0
    assert pq.read_table(out).num_rows == 25


def test_missing_collection(collection, tmp_path, capsys):
    assert main(["nope", str(tmp_path / "out.parquet")]) == 1
    assert capsys.readouterr().err == "error: collection 'nope' does not exist\n"


class FakeRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "failed to connect to all addresses"


@pytest.mark.parametrize(
    ("error", "args", "message"),
    [
        (UnexpectedResponse(404, "Not Found", b"", headers=None), [],
         "error: Qdrant returned HTTP 404 Not Found (empty response. Is --url correct?)"),
        (FakeRpcError(), [], "error: Qdrant returned gRPC UNAVAILABLE: failed to connect to all addresses "
                             "(is the gRPC port 6334 reachable? --rest uses the REST API instead)"),
        (FakeRpcError(), ["--grpc-port", "7334"], "(is the gRPC port 7334 reachable?"),
        (ConnectionError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"), ["--rest"],
         "(to trust a private certificate authority, set SSL_CERT_FILE to its certificate file)"),
        (ConnectionError("CERTIFICATE_VERIFY_FAILED"), [],
         "set SSL_CERT_FILE and GRPC_DEFAULT_SSL_ROOTS_FILE_PATH to its certificate file)"),
    ],
)
def test_error_hints(tmp_path, monkeypatch, capsys, error, args, message):
    def failing_export(*a, **kw):
        raise error

    monkeypatch.setattr(ClientConfig, "make", lambda self, check_compatibility=True: QdrantClient(":memory:"))
    monkeypatch.setattr(cli, "export_collection", failing_export)
    assert main(["c", str(tmp_path / "out.parquet"), "--retries", "0", *args]) == 1
    assert message in capsys.readouterr().err


def test_invalid_arguments(tmp_path):
    for args in (["--workers", "0"], ["--batch-size", "0"], ["--retries", "-1"]):
        with pytest.raises(SystemExit) as exc:
            main(["c", str(tmp_path / "out.parquet"), *args])
        assert exc.value.code == 2
