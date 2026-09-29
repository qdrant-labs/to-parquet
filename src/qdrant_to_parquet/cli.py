# ruff: noqa: E402
from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "NONE")

import argparse
import logging
import signal
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_to_parquet import __version__
from qdrant_to_parquet.exporter import AUTO_MAX_WORKERS, ClientConfig, export_collection
from qdrant_to_parquet.qdrant import describe_error
from qdrant_to_parquet.ranges import saved_points


class Terminated(BaseException):
    """Raised on SIGTERM, so it is handled like Ctrl-C."""


def _on_sigterm(signum, frame):
    raise Terminated


def _show_warning(message, category, filename, lineno, file=None, line=None):
    print(f"warning: {message}", file=sys.stderr)


def _positive_int(value: str) -> int:
    if int(value) <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return int(value)


def _non_negative_int(value: str) -> int:
    if int(value) < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {value}")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdrant-to-parquet",
        description="Export all points of a Qdrant collection (ids, vectors and payloads) to a Parquet file.",
    )
    parser.add_argument("collection", help="collection name or alias")
    parser.add_argument("output", type=Path, help="Parquet file to write")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (env: QDRANT_URL. Default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="API key (env: QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--grpc-port", type=int, default=6334, help="gRPC port (default: %(default)s)"
    )
    parser.add_argument(
        "--rest",
        action="store_true",
        help="use REST instead of gRPC (exact for integers above 2^63-1 and deeply nested payloads)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=60,
        help="request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retries",
        type=_non_negative_int,
        default=3,
        help="retries per failed request (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        help=f"parallel worker processes (default: up to {AUTO_MAX_WORKERS} for large collections)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=256,
        help="points per request (default: 256)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard the progress of an interrupted export instead of resuming it",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite OUTPUT if it exists"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="no progress bar, warnings or summary",
    )
    return parser


def _human_size(size: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


def _error_message(exc: Exception, args: argparse.Namespace) -> str:
    message = describe_error(exc)
    if isinstance(exc, UnexpectedResponse) and not exc.content:
        message += " (empty response. Is --url correct?)"
    if "CERTIFICATE_VERIFY_FAILED" in message:
        variables = (
            "SSL_CERT_FILE"
            if args.rest
            else "SSL_CERT_FILE and GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"
        )
        message += f" (to trust a private certificate authority, set {variables} to its certificate file)"
    elif not args.rest and "gRPC UNAVAILABLE" in message:
        message += f" (is the gRPC port {args.grpc_port} reachable? --rest uses the REST API instead)"
    return message


def _print_saved_progress(output: Path) -> None:
    try:
        points = saved_points(output)
    except Exception:
        return
    if points is not None:
        print(
            f"progress is saved ({points:,} points exported). Run the same command again to resume, "
            "or add --restart to start over",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.addLevelName(logging.WARNING, "warning")
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("grpc").setLevel(
        logging.CRITICAL
    )  # it logs failed calls with tracebacks

    if args.output.is_dir():
        parser.error(f"{args.output} is a directory")
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} already exists (use --force to overwrite)")

    config = ClientConfig(
        url=args.url,
        api_key=args.api_key,
        prefer_grpc=not args.rest,
        grpc_port=args.grpc_port,
        timeout=args.timeout,
    )
    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        with warnings.catch_warnings():
            warnings.showwarning = _show_warning  # no source lines
            if args.quiet:
                warnings.simplefilter("ignore")
            client = config.make()
            try:
                stats = export_collection(
                    client,
                    args.collection,
                    args.output,
                    batch_size=args.batch_size,
                    retries=args.retries,
                    progress=not args.quiet and sys.stderr.isatty(),
                    workers=args.workers,
                    client_config=config,
                    restart=args.restart,
                )
            finally:
                client.close()
    except (KeyboardInterrupt, Terminated) as exc:
        print(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "terminated",
            file=sys.stderr,
        )
        _print_saved_progress(args.output)
        return 130 if isinstance(exc, KeyboardInterrupt) else 143
    except Exception as exc:
        print(f"error: {_error_message(exc, args)}", file=sys.stderr)
        _print_saved_progress(args.output)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    if not args.quiet:
        details = f", {stats.workers} workers" if stats.workers > 1 else ""
        if stats.resumed_points:
            details += f", {stats.resumed_points:,} points from the interrupted run"
        print(
            f"Exported {stats.points:,} points from {args.collection!r} to {stats.output} "
            f"({_human_size(stats.bytes_written)}, {stats.seconds:.1f}s{details})",
            file=sys.stderr,
        )
        print(f"Columns: {', '.join(stats.columns)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
