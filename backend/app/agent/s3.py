"""The ``s3`` artifact backend: the agent's half (#791).

An assignment with ``storage.backend == "s3"`` is run exactly like a
``transfer`` one (``agent.transfer``: its own data directory, inputs in the
per-worker cache, outputs uploaded beside the run, the interrupt checkpoint
last), with one difference: the bytes go between this worker and the
bucket ``storage.s3`` names, never through the control plane.
:class:`S3Client` has ``TransferClient``'s shape (``session``, ``fetch``,
``upload``), so ``RunTransfer`` drives it unchanged:

- ``fetch`` downloads an input the cache lacks from its object
  (``TransferInput.key``, ``<prefix>inputs/<sha256>``) and checks its hash;
- ``upload`` puts a file under the run's prefix (``storage.s3.prefix``,
  ``.../runs/<run_id>/``): a reported artifact at ``<kind>/<name>``, a
  writer node's file at ``output/<path under the run's data folder>``
  (multipart for a large one, boto3's transfer manager), and answers the
  object key, which the ``artifact`` frame then carries
  (``backend: "s3"``). It retries like the transfer backend
  (``agent.transfer.retry_upload``): an unreachable endpoint, a 5xx or
  expired temporary credentials ride out the run's retry window, waiting
  for the endpoint or for the ``credentials`` frame that renews them
  (:meth:`S3Client.update_access`); a refusal (``AccessDenied``, a missing
  bucket) fails the file at once, which then stays on the worker with a
  ``transfer_failed`` warning.

The credentials stay in this process's memory: never in a file, a log or
the run subprocess's environment (#251). boto3 is optional (the ``s3``
extra): an agent without it does not list the ``s3`` capability, and the
control plane sends it ``transfer`` runs instead.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from ..core.s3 import (
    CHUNK,
    OUTPUT_DIRNAME,
    SHA256_METADATA,
    client_available,
    describe,
    error_code,
    make_client,
)
from ..core.transfer import PLACE_DATA, PLACE_RUN, clean_name, sha256_file
from .transfer import (
    ATTEMPTS,
    OUTAGE_STATUSES,
    RETRY_PAUSE_S,
    InputCache,
    RetryableUpload,
    TransferCancelled,
    TransferError,
    _sleep,
    retry_upload,
)

logger = logging.getLogger(__name__)

#: Error codes that mean "these credentials stopped working", which a
#: ``credentials`` frame on its way will fix: retried like an outage.
EXPIRED_CODES = frozenset({"ExpiredToken", "TokenRefreshRequired",
                           "RequestExpired", "ExpiredTokenException"})

__all__ = ["S3Client", "client_available"]


class _NoSession:
    """``TransferClient.session()``'s shape: boto3 pools connections in
    the client itself, so a batch needs nothing of its own."""

    def __enter__(self) -> "_NoSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _retryable(exc: BaseException) -> tuple[bool, bool]:
    """``(retry, outage)`` for a failed S3 call: a refusal is neither; an
    unreachable endpoint, a throttle or expired credentials an outage; any
    other 5xx a quick retry."""
    code, status = error_code(exc)
    if code is None and status is None:
        # No answer at all: a connection, timeout or DNS failure is the
        # endpoint being away; a local error (the file vanished) is not.
        return (True, True) if _botocore_error(exc) else (False, False)
    if code in EXPIRED_CODES or status in OUTAGE_STATUSES \
            or code in ("SlowDown", "RequestTimeout", "Throttling"):
        return True, True
    if status is not None and status >= 500:
        return True, False
    return False, False


def _botocore_error(exc: BaseException) -> bool:
    """Whether *exc*, or what caused it, is botocore's own (no HTTP answer:
    ``EndpointConnectionError``, a timeout ...)."""
    try:
        from botocore.exceptions import BotoCoreError
    except ImportError:  # pragma: no cover - no client, nothing to call
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BotoCoreError):
            return True
        current = current.__cause__ or current.__context__
    return False


class S3Client:
    """One ``s3`` run's access to its bucket (``wp.S3Access``)."""

    #: ``artifact.backend`` of what it uploads.
    backend = "s3"

    def __init__(self, access: Any, cache: InputCache,
                 factory: Callable[..., Any] = make_client) -> None:
        self.cache = cache
        self._factory = factory
        self._lock = threading.Lock()
        self._access = access
        self._client: Any = None

    @property
    def access(self) -> Any:
        return self._access

    def update_access(self, access: Any) -> None:
        """Use *access* from now on (a ``credentials`` frame): the next
        request builds a client with it. Thread-safe."""
        with self._lock:
            self._access = access
            self._client = None

    def _s3(self) -> tuple[Any, Any]:
        """``(client, access)``, built from the current access."""
        with self._lock:
            if self._client is None:
                access = self._access
                self._client = self._factory(
                    endpoint_url=access.endpoint_url, region=access.region,
                    addressing_style=access.addressing_style,
                    access_key_id=access.access_key_id,
                    secret_access_key=access.secret_access_key,
                    session_token=access.session_token)
            return self._client, self._access

    def session(self) -> _NoSession:
        return _NoSession()

    def key_for(self, kind: str, name: str, place: str = PLACE_RUN) -> str:
        """The object key a file the run uploads as *kind*/*name* gets."""
        from ..core.transfer import _part

        prefix = self._access.prefix
        try:
            relative = clean_name(name).as_posix()
        except ValueError:
            raise TransferError(f"refusing the output name {name!r}") \
                from None
        if place == PLACE_DATA:
            return f"{prefix}{OUTPUT_DIRNAME}/{relative}"
        if place != PLACE_RUN:
            raise TransferError(f"unknown place {place!r}")
        return f"{prefix}{_part(kind)}/{relative}"

    def fetch(self, entry: Any, cancelled: Callable[[], bool] = lambda: False,
              *, client: Any = None) -> tuple[Path, bool]:
        """``(cache path, fetched)`` for one input: from the cache when it
        holds it, else downloaded from ``entry.key`` and verified."""
        cached = self.cache.lookup(entry.sha256, entry.bytes)
        if cached is not None:
            return cached, False
        if not entry.key:
            raise TransferError(f"fetching {entry.name}: the assignment "
                                "names no S3 object for it")
        last: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            if cancelled():
                raise TransferCancelled("cancelled while fetching")
            try:
                s3, access = self._s3()
                body = s3.get_object(Bucket=access.bucket,
                                     Key=entry.key)["Body"]
                try:
                    return self.cache.add(entry.sha256,
                                          body.iter_chunks(CHUNK),
                                          cancelled), True
                finally:
                    body.close()
            except TransferError:
                raise
            except Exception as exc:  # noqa: BLE001 -- botocore, network
                error = TransferError(f"fetching {entry.name} from S3: "
                                      f"{describe(exc)}")
                retry, _ = _retryable(exc)
                if not retry:
                    raise error from None
                last = error
            _sleep(RETRY_PAUSE_S * attempt, cancelled)
        raise last or TransferError(f"fetching {entry.name} failed")

    def upload(self, run_id: str, kind: str, name: str, path: Path, *,
               place: str = PLACE_RUN, client: Any = None,
               cancelled: Callable[[], bool] = lambda: False,
               retry_until: Callable[[], float] | None = None,
               fail_fast: bool = False) -> str:
        """Put *path* in the bucket as *run_id*'s *kind*/*name*; its key.
        ``TransferError`` when it was not taken (see the module
        docstring). The object's metadata carries its sha256."""
        key = self.key_for(kind, name, place)
        checksum = sha256_file(path)

        def attempt() -> str:
            try:
                s3, access = self._s3()
                s3.upload_file(str(path), access.bucket, key, ExtraArgs={
                    "Metadata": {SHA256_METADATA: checksum}})
            except Exception as exc:  # noqa: BLE001 -- botocore / boto3
                error = TransferError(f"uploading {name} to S3: "
                                      f"{describe(exc)}")
                retry, outage = _retryable(exc)
                if not retry:
                    raise error from None
                raise RetryableUpload(error, outage=outage) from None
            return key

        return retry_upload(attempt, name, cancelled=cancelled,
                            retry_until=retry_until, fail_fast=fail_fast)
