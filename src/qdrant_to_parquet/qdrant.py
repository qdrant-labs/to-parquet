from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import grpc
from qdrant_client import QdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_GRPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}
# Rate limits don't use up the retries, but a request gives up after waiting this long.
RATE_LIMIT_MAX_WAIT = 600


class ExportError(Exception):
    """A problem the user can fix."""


@dataclass(frozen=True)
class ClientConfig:
    """How to connect to Qdrant. Picklable, so worker processes can connect too."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    prefer_grpc: bool = True
    grpc_port: int = 6334
    timeout: int | None = None

    def make(self, check_compatibility: bool = True) -> QdrantClient:
        return QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            grpc_port=self.grpc_port,
            timeout=self.timeout,
            check_compatibility=check_compatibility,
        )


def point_id_key(point_id: Any) -> tuple[int, int]:
    """Qdrant's point order: integer ids first, then UUIDs by value."""
    if isinstance(point_id, int):
        return (0, point_id)
    return (1, uuid.UUID(str(point_id)).int)


def grpc_status(exc: grpc.RpcError) -> tuple[grpc.StatusCode | None, str]:
    code = exc.code() if callable(getattr(exc, "code", None)) else None
    details = exc.details() if callable(getattr(exc, "details", None)) else None
    return code, details or str(exc)


def describe_error(exc: BaseException, limit: int = 500) -> str:
    """One line, however long or multi-line the server's response was."""
    if isinstance(exc, UnexpectedResponse):
        content = (exc.content or b"").decode(errors="replace")
        text = f"Qdrant returned HTTP {exc.status_code} {exc.reason_phrase or ''}: {content}"
    elif isinstance(exc, grpc.RpcError):
        code, details = grpc_status(exc)
        text = f"Qdrant returned gRPC {code.name if code else 'error'}: {details}"
    elif isinstance(exc, ExportError):
        text = str(exc)
    else:
        text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split()).rstrip(" :")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, UnexpectedResponse):
        return (
            exc.status_code is None or exc.status_code >= 500 or exc.status_code == 429
        )
    if isinstance(exc, grpc.RpcError):
        return grpc_status(exc)[0] in RETRYABLE_GRPC_CODES
    return isinstance(exc, (ResponseHandlingException, ConnectionError, TimeoutError))


def call_with_retries(fn: Callable[[], T], *, retries: int, what: str) -> T:
    """Call ``fn``, retrying transient failures with backoff and waiting out rate limits."""
    attempt = 0
    rate_limited_for = 0
    while True:
        try:
            return fn()
        except ResourceExhaustedResponse as exc:
            delay = max(1, int(exc.retry_after_s or 0))
            if rate_limited_for + delay > RATE_LIMIT_MAX_WAIT:
                raise ExportError(
                    f"{what} still rate limited after waiting {rate_limited_for}s ({describe_error(exc)})"
                ) from exc
            rate_limited_for += delay
            if log.isEnabledFor(logging.WARNING):
                log.warning(
                    "%s rate limited (%s). Waiting %ss",
                    what,
                    describe_error(exc),
                    delay,
                )
            time.sleep(delay)
        except Exception as exc:
            if attempt >= retries or not _is_retryable(exc):
                code, details = (
                    grpc_status(exc) if isinstance(exc, grpc.RpcError) else (None, "")
                )
                if code == grpc.StatusCode.INTERNAL and "deserializ" in details:
                    raise ExportError(
                        "the gRPC client could not decode a page of points (payloads nested more than "
                        "~40 levels deep exceed its recursion limit). Use --rest"
                    ) from exc
                raise
            delay = 2**attempt
            attempt += 1
            if log.isEnabledFor(logging.WARNING):
                log.warning(
                    "%s failed (%s). Retrying in %ss (%d/%d)",
                    what,
                    describe_error(exc),
                    delay,
                    attempt,
                    retries,
                )
            time.sleep(delay)


def get_collection(
    client: QdrantClient, collection: str, retries: int
) -> models.CollectionInfo:
    try:
        return call_with_retries(
            lambda: client.get_collection(collection),
            retries=retries,
            what="get collection",
        )
    except (UnexpectedResponse, grpc.RpcError, ValueError) as exc:
        not_found = (
            # A 404 without "collection" in it means a wrong URL.
            (
                isinstance(exc, UnexpectedResponse)
                and exc.status_code == 404
                and b"ollection" in (exc.content or b"")
            )
            or (
                isinstance(exc, grpc.RpcError)
                and grpc_status(exc)[0] == grpc.StatusCode.NOT_FOUND
            )
            or (
                isinstance(exc, ValueError) and "not found" in str(exc)
            )  # qdrant-client local mode
        )
        if not_found:
            raise ExportError(f"collection {collection!r} does not exist") from None
        raise


def count_points(client: QdrantClient, collection: str, retries: int) -> int:
    return call_with_retries(
        lambda: client.count(collection, exact=True), retries=retries, what="count"
    ).count


def scroll_batches(
    client: QdrantClient,
    collection: str,
    *,
    batch_size: int,
    retries: int = 3,
    start: Any = None,
    end: Any = None,
) -> Iterator[tuple[list[models.Record], Any]]:
    """Yield pages of points in ``[start, end)``, each with the offset of the next page (None after the last)."""
    offset = start
    end_key = None if end is None else point_id_key(end)
    while True:
        records, next_offset = call_with_retries(
            lambda offset=offset: client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            ),
            retries=retries,
            what="scroll",
        )
        if end_key is not None:
            kept = [r for r in records if point_id_key(r.id) < end_key]
            if len(kept) < len(records) or (
                next_offset is not None and point_id_key(next_offset) >= end_key
            ):
                next_offset = None
            records = kept
        if not records:
            return
        yield records, next_offset
        if next_offset is None:
            return
        offset = next_offset
