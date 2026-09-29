"""Export a Qdrant collection to a Parquet file."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qdrant-to-parquet")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
