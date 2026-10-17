"""The ``s3`` artifact backend: the control plane's half (#791).

The alternative to the ``transfer`` backend (#785) for a worker without the
shared mount, after Argo Workflows' S3 artifact repository: when
``CODEFYUI_S3_ENABLED`` is on and the worker's agent speaks it (capability
``s3``), the bytes go between the worker and an S3 / MinIO bucket and this
server stores only object keys.

- **Keys.** Under ``CODEFYUI_S3_PREFIX`` (:func:`key_prefix`):

  * ``inputs/<sha256>``: an upload a run reads, named by content
    (:func:`ensure_inputs` puts it there once, at assignment; the next run
    that reads the same file finds it). The agent keeps its own cache as
    for ``transfer``.
  * ``runs/<run_id>/<kind>/<name>`` (:func:`run_prefix`): what the run
    wrote. A reported artifact (a checkpoint, a TensorBoard folder file by
    file) under its kind; a writer node's file (``ModelSaver`` ...) under
    ``output/<models|media|output>/<path>``, kind ``output``, the path it
    has under the run's data folder.

  An ``exec_run_artifacts`` row records ``backend = "s3"`` and the key
  (:func:`owns_key` refuses one outside the run's prefix); listings show it
  as ``s3://<bucket>/<key>`` (:func:`uri`) and the download route proxies
  it or redirects to a pre-signed URL (``CODEFYUI_S3_DOWNLOAD``).
- **Credentials** (:func:`issue_access`). ``CODEFYUI_S3_CREDENTIALS``:
  ``sts`` asks the endpoint's STS for temporary ones (``AssumeRole`` with a
  session policy that allows writing the run's prefix and reading its
  inputs only), ``static`` hands the worker this server's own keys, and
  ``auto`` tries STS first. They travel in ``assign.storage.s3`` like a
  run's secret params (over TLS when the agent dials ``https://``); the
  agent keeps them in memory. ``WorkerExecutor`` sends fresh temporary
  ones (``credentials``) before the old ones expire. This server's keys are
  read from the environment and never written to SQLite or a log (#251).
- **Retention** (:func:`remove_run_objects`). ``RunStore.prune`` and
  ``DELETE /api/runs/{id}`` remove the run's objects, as #785 removes a
  transfer run's directory: an interrupted run keeps its ``checkpoint/``
  under retention, and a writer node's files (``output/``) stay, as they
  stay in this server's data folders with ``transfer``. Uploads a worker
  abandoned mid-way are the bucket's lifecycle rule's to clean
  (AbortIncompleteMultipartUpload; the docs show one).

boto3 is optional (the ``s3`` extra): imported lazily, and without it the
backend is off with a reason (:func:`problem`). Every function that talks
to the bucket blocks: call it in a thread.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

#: ``exec_run_artifacts.backend`` and ``artifact.backend`` for this backend.
BACKEND = "s3"
#: Under the prefix: content-addressed inputs, and every run's objects.
INPUTS_DIRNAME = "inputs"
RUNS_DIRNAME = "runs"
#: Under a run's prefix: a writer node's files (kept by retention).
OUTPUT_DIRNAME = "output"
CHECKPOINT_DIRNAME = "checkpoint"
#: ``run_warning`` kind: the run's worker cannot use S3, so its files go
#: through this server (the transfer backend) instead.
WARNING_STORAGE_FALLBACK = "storage_fallback"
#: Object metadata key holding the sha256 of what was uploaded.
SHA256_METADATA = "sha256"

#: Bytes read at a time when streaming an object.
CHUNK = 1024 * 1024

#: What ``pip`` installs to get the client.
INSTALL_HINT = ("install boto3 into CodefyUI's environment (the `s3` extra: "
                "`uv pip install boto3`)")


class S3Unavailable(RuntimeError):
    """The bucket could not be used; the message says why (no secrets)."""


@functools.lru_cache(maxsize=1)
def client_available() -> bool:
    """Whether boto3 can be imported here (looked up once)."""
    return importlib.util.find_spec("boto3") is not None


def problem() -> str | None:
    """Why the backend is off although ``CODEFYUI_S3_ENABLED`` is set, or
    None (it is on, or it was not asked for)."""
    from ..config import settings

    if not settings.S3_ENABLED:
        return None
    return unconfigured()


def unconfigured() -> str | None:
    """Why this server cannot reach a bucket, or None when it can."""
    from ..config import settings

    if not settings.S3_BUCKET:
        return "CODEFYUI_S3_BUCKET is not set"
    if not client_available():
        return f"boto3 is not installed; {INSTALL_HINT}"
    return None


def configured() -> bool:
    """A bucket is named and the client is installed: rows of this backend
    can be served and collected (even with ``S3_ENABLED`` turned off
    since)."""
    from ..config import settings

    return bool(settings.S3_BUCKET) and client_available()


def enabled() -> bool:
    """Whether new worker runs without the mount use this backend."""
    from ..config import settings

    return bool(settings.S3_ENABLED) and configured()


# ── keys ──────────────────────────────────────────────────────────────────


def key_part(value: str) -> str:
    """One key component from an id (a run id, an artifact kind): every
    character outside ``[A-Za-z0-9._-]`` becomes ``_``. Never shortened:
    ``checkpoints._safe_part`` cuts at 48 characters, which would make two
    long names one key (#838); an object key may be 1024 bytes."""
    cleaned = "".join(c if (c.isascii() and (c.isalnum() or c in "._-"))
                      else "_" for c in str(value))
    return cleaned.strip("._") or "unnamed"



def key_prefix() -> str:
    """``CODEFYUI_S3_PREFIX`` as a key prefix: no leading ``/``, one
    trailing ``/`` unless empty."""
    from ..config import settings

    prefix = str(settings.S3_PREFIX or "").strip().strip("/")
    return f"{prefix}/" if prefix else ""


def run_prefix(run_id: str) -> str:
    """``<prefix>runs/<run_id>/``: every object *run_id* writes."""
    return f"{key_prefix()}{RUNS_DIRNAME}/{key_part(run_id)}/"


def input_key(sha256: str) -> str:
    """``<prefix>inputs/<sha256>``: an upload a run reads, by content."""
    return f"{key_prefix()}{INPUTS_DIRNAME}/{sha256}"


def owns_key(run_id: str, key: str) -> bool:
    """Whether *key* is one of *run_id*'s objects: under :func:`run_prefix`,
    at least ``<kind>/<name>`` deep, with no part that is empty, ``.`` or
    ``..``. What a worker may record for the run, and what retention and
    the download route touch."""
    prefix = run_prefix(run_id)
    key = str(key)
    if not key.startswith(prefix):
        return False
    rest = key[len(prefix):]
    parts = rest.split("/")
    if len(parts) < 2:
        return False
    return all(part not in ("", ".", "..") for part in parts) \
        and "\\" not in rest


def uri(key: str) -> str | None:
    """``s3://<bucket>/<key>``, how a listing names an S3 row; None
    without a bucket."""
    from ..config import settings

    bucket = settings.S3_BUCKET
    return f"s3://{bucket}/{key}" if bucket else None


# ── clients ───────────────────────────────────────────────────────────────


def make_client(*, endpoint_url: str | None, region: str | None,
                addressing_style: str, access_key_id: str,
                secret_access_key: str, session_token: str | None = None,
                service: str = "s3") -> Any:
    """A boto3 client for *service*. ``S3Unavailable`` without boto3."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise S3Unavailable(f"boto3 is not installed; {INSTALL_HINT}") \
            from None
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": addressing_style},
        connect_timeout=10, read_timeout=60,
        retries={"max_attempts": 3, "mode": "standard"})
    session = boto3.session.Session()
    return session.client(
        service, endpoint_url=endpoint_url or None,
        region_name=region or None,
        aws_access_key_id=access_key_id or None,
        aws_secret_access_key=secret_access_key or None,
        aws_session_token=session_token or None, config=config)


_clients: dict[tuple[Any, ...], Any] = {}
_clients_lock = threading.Lock()


def _settings_key(service: str) -> tuple[Any, ...]:
    from ..config import settings

    return (service, settings.S3_ENDPOINT_URL, settings.S3_REGION,
            settings.S3_ADDRESSING_STYLE, settings.S3_ACCESS_KEY_ID,
            settings.S3_SECRET_ACCESS_KEY.get_secret_value())


def control_client(service: str = "s3") -> Any:
    """This server's client for *service*, from its settings (one per
    distinct configuration; boto3 clients are thread-safe)."""
    from ..config import settings

    key = _settings_key(service)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = make_client(
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region=settings.S3_REGION or None,
                addressing_style=settings.S3_ADDRESSING_STYLE,
                access_key_id=settings.S3_ACCESS_KEY_ID,
                secret_access_key=settings.S3_SECRET_ACCESS_KEY
                .get_secret_value(),
                service=service)
            # The settings changed: drop this service's stale client.
            for stale in [k for k in _clients if k[0] == service]:
                del _clients[stale]
            _clients[key] = client
        return client


def error_code(exc: BaseException) -> tuple[str | None, int | None]:
    """``(code, HTTP status)`` of a botocore ``ClientError`` (or one that
    caused *exc*: boto3's transfer manager wraps them); ``(None, None)``
    for anything else."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if isinstance(response, Mapping):
            error = response.get("Error") or {}
            meta = response.get("ResponseMetadata") or {}
            status = meta.get("HTTPStatusCode")
            return (str(error.get("Code")) if error.get("Code") else None,
                    int(status) if status else None)
        current = current.__cause__ or current.__context__
    return None, None


def _is_missing(exc: BaseException) -> bool:
    code, status = error_code(exc)
    return status == 404 or code in ("404", "NoSuchKey", "NotFound")


def describe(exc: BaseException) -> str:
    """A one-line reason for a failed S3 call: the error code and message,
    never the request's headers or credentials."""
    code, status = error_code(exc)
    if code is not None:
        response = getattr(exc, "response", None) or {}
        message = (response.get("Error") or {}).get("Message") \
            if isinstance(response, Mapping) else None
        return f"{code} (HTTP {status}){': ' + message if message else ''}"
    return f"{type(exc).__name__}: {exc}"


# ── credentials ───────────────────────────────────────────────────────────


def session_policy(bucket: str, prefix: str,
                   input_keys: Iterable[str]) -> str:
    """The STS session policy for one run: write and read under its
    *prefix*, read its inputs (by name; the whole ``inputs/`` folder when
    listing them would make the policy too long)."""
    keys = sorted(set(input_keys))
    inputs = [f"arn:aws:s3:::{bucket}/{key}" for key in keys]
    statements: list[dict[str, Any]] = [{
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload",
                   "s3:ListMultipartUploadParts"],
        "Resource": [f"arn:aws:s3:::{bucket}/{prefix}*"],
    }]
    if inputs:
        statements.append({"Effect": "Allow", "Action": ["s3:GetObject"],
                           "Resource": inputs})
    policy = json.dumps({"Version": "2012-10-17", "Statement": statements},
                        separators=(",", ":"))
    if len(policy) > 2000 and inputs:
        statements[1]["Resource"] = [
            f"arn:aws:s3:::{bucket}/{key_prefix()}{INPUTS_DIRNAME}/*"]
        policy = json.dumps({"Version": "2012-10-17",
                             "Statement": statements}, separators=(",", ":"))
    return policy


_sts_warned = False


def issue_access(run_id: str, input_keys: Iterable[str] = ()) -> Any:
    """``worker_protocol.S3Access`` for *run_id*'s worker: temporary
    credentials from STS, or this server's own (``CODEFYUI_S3_CREDENTIALS``,
    see the module docstring). ``S3Unavailable`` when ``sts`` was asked
    for and STS refused. Blocking."""
    global _sts_warned
    from . import worker_protocol as wp
    from ..config import settings

    prefix = run_prefix(run_id)
    common = dict(endpoint_url=settings.S3_ENDPOINT_URL or None,
                  region=settings.S3_REGION or None,
                  bucket=settings.S3_BUCKET, prefix=prefix,
                  addressing_style=settings.S3_ADDRESSING_STYLE)
    mode = settings.S3_CREDENTIALS
    if mode in ("sts", "auto"):
        try:
            answer = control_client("sts").assume_role(
                RoleArn=(settings.S3_STS_ROLE_ARN
                         or "arn:aws:iam::000000000000:role/codefyui"),
                RoleSessionName=f"codefyui-{key_part(run_id)}"[:64],
                Policy=session_policy(settings.S3_BUCKET, prefix,
                                      input_keys),
                DurationSeconds=int(settings.S3_STS_DURATION_S))
            creds = answer["Credentials"]
            expires = creds.get("Expiration")
            expires_at = expires.timestamp() if hasattr(expires, "timestamp") \
                else time.time() + int(settings.S3_STS_DURATION_S)
            return wp.S3Access(
                **common, access_key_id=creds["AccessKeyId"],
                secret_access_key=creds["SecretAccessKey"],
                session_token=creds.get("SessionToken"),
                expires_at=expires_at)
        except Exception as exc:  # noqa: BLE001 -- STS refused or absent
            if mode == "sts":
                raise S3Unavailable(
                    f"STS AssumeRole failed: {describe(exc)} "
                    "(CODEFYUI_S3_CREDENTIALS=sts)") from None
            if not _sts_warned:
                _sts_warned = True
                logger.warning(
                    "S3: STS AssumeRole failed (%s); workers get this "
                    "server's own S3 keys instead (CODEFYUI_S3_CREDENTIALS="
                    "auto). Set it to `static` to skip STS, or `sts` to "
                    "refuse such runs.", describe(exc))
    return wp.S3Access(
        **common, access_key_id=settings.S3_ACCESS_KEY_ID,
        secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value())


# ── inputs ────────────────────────────────────────────────────────────────


def ensure_inputs(manifest: Iterable[Mapping[str, Any]]
                  ) -> list[dict[str, Any]]:
    """*manifest* (``ContentIndex.manifest``) with each entry's object
    ``key``, uploading the stored file once when the bucket does not hold
    it yet. Blocking; ``S3Unavailable`` when the bucket cannot be used."""
    from ..config import settings
    from .transfer import store_path

    client = control_client()
    bucket = settings.S3_BUCKET
    entries: list[dict[str, Any]] = []
    for entry in manifest:
        key = input_key(str(entry["sha256"]))
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            present = int(head.get("ContentLength", -1)) == int(entry["bytes"])
        except Exception as exc:  # noqa: BLE001 -- botocore
            if not _is_missing(exc):
                raise S3Unavailable(f"bucket {bucket}: {describe(exc)}") \
                    from None
            present = False
        if not present:
            path = store_path(str(entry["kind"]), str(entry["name"]))
            if path is None:
                raise S3Unavailable(f"`{entry['name']}` is not an upload on "
                                    "this server any more")
            try:
                client.upload_file(
                    str(path), bucket, key,
                    ExtraArgs={"Metadata": {SHA256_METADATA:
                                            str(entry["sha256"])}})
            except Exception as exc:  # noqa: BLE001 -- botocore / boto3
                raise S3Unavailable(f"uploading `{entry['name']}` to bucket "
                                    f"{bucket}: {describe(exc)}") from None
            logger.info("S3: uploaded input %s as %s", entry["name"], key)
        entries.append({**entry, "key": key})
    return entries


# ── reading and removing a run's objects ──────────────────────────────────


def head(key: str) -> dict[str, Any] | None:
    """The object's metadata, or None when there is no such object."""
    from ..config import settings

    try:
        return control_client().head_object(Bucket=settings.S3_BUCKET,
                                            Key=key)
    except Exception as exc:  # noqa: BLE001 -- botocore
        if _is_missing(exc):
            return None
        raise S3Unavailable(describe(exc)) from None


def is_folder(key: str) -> bool:
    """Whether *key* names a folder: objects exist under ``<key>/``."""
    from ..config import settings

    try:
        answer = control_client().list_objects_v2(
            Bucket=settings.S3_BUCKET, Prefix=key.rstrip("/") + "/",
            MaxKeys=1)
    except Exception as exc:  # noqa: BLE001 -- botocore
        raise S3Unavailable(describe(exc)) from None
    return bool(answer.get("KeyCount") or answer.get("Contents"))


def open_stream(key: str) -> Iterator[bytes]:
    """The object's bytes, :data:`CHUNK` at a time (a sync iterator, for a
    ``StreamingResponse``). The object is opened here, so a missing one or
    a refusal raises now (``S3Unavailable``), not mid-response."""
    from ..config import settings

    try:
        body = control_client().get_object(Bucket=settings.S3_BUCKET,
                                           Key=key)["Body"]
    except Exception as exc:  # noqa: BLE001 -- botocore
        raise S3Unavailable(describe(exc)) from None

    def chunks() -> Iterator[bytes]:
        try:
            yield from body.iter_chunks(CHUNK)
        finally:
            body.close()

    return chunks()


def presigned_url(key: str, filename: str) -> str:
    """A pre-signed GET for *key*, valid ``CODEFYUI_S3_PRESIGN_TTL_S``, that
    downloads as *filename*."""
    from ..config import settings

    safe = filename.replace('"', "")
    return control_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key,
                "ResponseContentDisposition":
                    f'attachment; filename="{safe}"'},
        ExpiresIn=int(settings.S3_PRESIGN_TTL_S))


def _kept(relative: str, keep_checkpoints: bool) -> bool:
    first = relative.split("/", 1)[0]
    return first == OUTPUT_DIRNAME or (keep_checkpoints
                                       and first == CHECKPOINT_DIRNAME)


def remove_run_objects(run_id: str, *, keep_checkpoints: bool = False) -> int:
    """Delete *run_id*'s objects (retention, ``DELETE``): everything under
    :func:`run_prefix` but ``output/`` (a writer node's files) and, with
    *keep_checkpoints* (an interrupted run under retention),
    ``checkpoint/``; and abort the run's unfinished multipart uploads.
    Blocking. How many objects went."""
    from ..config import settings

    client = control_client()
    bucket = settings.S3_BUCKET
    prefix = run_prefix(run_id)
    doomed: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = str(item["Key"])
            if not _kept(key[len(prefix):], keep_checkpoints):
                doomed.append(key)
    removed = 0
    for start in range(0, len(doomed), 1000):
        batch = doomed[start:start + 1000]
        answer = client.delete_objects(
            Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch],
                                   "Quiet": True})
        failed = answer.get("Errors") or []
        removed += len(batch) - len(failed)
        for error in failed[:3]:
            logger.warning("S3: could not delete %s: %s", error.get("Key"),
                           error.get("Code"))
    try:
        uploads = client.list_multipart_uploads(Bucket=bucket, Prefix=prefix)
        for upload in uploads.get("Uploads") or []:
            client.abort_multipart_upload(Bucket=bucket, Key=upload["Key"],
                                          UploadId=upload["UploadId"])
    except Exception:  # noqa: BLE001 -- the lifecycle rule is the backstop
        logger.debug("S3: could not list %s's unfinished uploads", run_id,
                     exc_info=True)
    return removed


def remove_runs_objects(runs: Iterable[tuple[str, bool]]) -> list[str]:
    """:func:`remove_run_objects` for each ``(run_id, keep_checkpoints)``;
    the runs something was removed for. Never raises: a bucket that cannot
    be reached is logged and its objects stay (the next retention pass does
    not see the run again, so the bucket's lifecycle rule is the
    backstop). Blocking."""
    removed: list[str] = []
    for run_id, keep in runs:
        try:
            if remove_run_objects(run_id, keep_checkpoints=keep):
                removed.append(run_id)
        except Exception as exc:  # noqa: BLE001 -- botocore, network
            logger.warning("S3: could not remove the objects of run %s: %s",
                           run_id, describe(exc))
    return removed


def filename(key: str) -> str:
    """The last part of *key*, what a download is saved as."""
    return PurePosixPath(key).name or Path(key).name or "artifact"
