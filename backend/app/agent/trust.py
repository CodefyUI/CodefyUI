"""How the agent verifies the control plane over TLS (#786).

Assignments carry graphs with their SECRET parameters restored, so the
agent must know it is talking to its own control plane. Over ``https``,
one :class:`ControlPlaneTrust` decides, for the enrollment request
(``urllib``) and for every ``wss`` dial (``websockets``) alike: both take
the ``ssl.SSLContext`` from :meth:`ControlPlaneTrust.context`, whose socket
and BIO classes check the certificate at the end of the handshake.

Two modes, and nothing is ever learned silently:

- **Pinned**: the user passed ``--ca-fingerprint sha256:<hex>`` (now, or on
  an earlier start, which the credential file recorded). The SPKI sha256
  of the server's LEAF certificate (the one whose key the handshake
  proved) must equal it, on every Python version; a CA certificate in the
  chain does not count. The CA and the host name are not checked, which is
  what lets a self-signed certificate on a lab network work. A mismatch
  raises :class:`FingerprintMismatch` with both values; it is final.
- **System CAs** (no pin): the usual CA and host name checks, and nothing
  is recorded, so a renewal with a new key keeps working. A certificate
  the CAs reject fails the handshake; :func:`untrusted_message` adds the
  fingerprint it presents, for the user to check and pin.

The flag keeps its name from the issue (and kubeadm's habit); what it pins
is the server certificate's public key, not a CA's.

Plain ``http`` has no certificate; the CLI warns once instead.
"""

from __future__ import annotations

import socket
import ssl
from typing import Any
from urllib.parse import urlsplit

from ..core import tls_fingerprint as fp

#: Seconds :func:`probe_fingerprint` may take.
PROBE_TIMEOUT_S = 10.0


class TrustError(ssl.SSLCertVerificationError):
    """The control plane's certificate is not the one the user pinned.

    An ``ssl`` error so ``urllib`` and ``asyncio`` treat it as a failed
    handshake (the connection is closed before anything is sent)."""

    def __str__(self) -> str:  # ssl.SSLError prints its args tuple
        return str(self.args[0]) if self.args else ""


class FingerprintMismatch(TrustError):
    def __init__(self, expected: str, actual: str, source: str,
                 hint: str = "") -> None:
        self.expected = expected
        self.actual = actual
        self.source = source
        message = (f"the control plane's certificate does not match: "
                   f"expected {expected} ({source}), got {actual}")
        super().__init__(f"{message}. {hint}" if hint else message)


def is_https(url: str) -> bool:
    return urlsplit(url.strip()).scheme.lower() in ("https", "wss")


def first_use_context() -> ssl.SSLContext:
    """The context without a pin: this machine's CAs, host name checked
    (a function so a test can add its own CA)."""
    return ssl.create_default_context()


class ControlPlaneTrust:
    """The TLS verifier shared by enrollment and the worker socket.

    *expected* is the ``sha256:<hex>`` pin the user gave, or None (system
    CAs). *source* names where the pin came from and *hint* says how to
    re-trust, both for the mismatch message.
    """

    def __init__(self, expected: str | None = None, *,
                 source: str = "--ca-fingerprint", hint: str = "") -> None:
        self.expected = fp.normalize(expected) if expected else None
        self.source = source
        self.hint = hint
        #: The leaf fingerprint of the last completed handshake.
        self.observed: str | None = None

    def check(self, conn: Any) -> None:
        leaf = conn.getpeercert(binary_form=True)
        if not leaf:
            raise TrustError("the control plane presented no certificate")
        self.observed = fp.spki_fingerprint(leaf)
        if self.expected is not None and self.observed != self.expected:
            raise FingerprintMismatch(self.expected, self.observed,
                                      self.source, self.hint)

    def context(self) -> ssl.SSLContext:
        """A client context that runs :meth:`check` after each handshake."""
        if self.expected is not None:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        else:
            ctx = first_use_context()
        trust = self

        class PinnedSocket(ssl.SSLSocket):
            def do_handshake(self, *args: Any, **kwargs: Any) -> None:
                super().do_handshake(*args, **kwargs)
                trust.check(self)

        class PinnedObject(ssl.SSLObject):
            def do_handshake(self) -> None:
                super().do_handshake()
                trust.check(self)

        ctx.sslsocket_class = PinnedSocket
        ctx.sslobject_class = PinnedObject
        return ctx


def probe_fingerprint(url: str,
                      timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """The leaf fingerprint *url*'s server presents, checking nothing; None
    when it cannot be read. Only for telling the user what they would be
    pinning; never a reason to trust it."""
    parts = urlsplit(url.strip())
    host = parts.hostname
    if not host:
        return None
    port = parts.port or 443
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw, \
                ctx.wrap_socket(raw, server_hostname=host) as tls:
            der = tls.getpeercert(binary_form=True)
    except (OSError, ValueError):
        return None
    try:
        return fp.spki_fingerprint(der) if der else None
    except ValueError:
        return None


def untrusted_message(url: str, exc: BaseException) -> str:
    """Why a certificate this machine's CAs reject was refused, and how to
    pin it after checking it out of band."""
    seen = probe_fingerprint(url)
    reason = getattr(exc, "verify_message", None) or str(exc)
    message = (f"the certificate of {url} is not trusted by this machine "
               f"({reason})")
    if seen is None:
        return message
    return (f"{message}. It presents {seen}: if that equals the fingerprint "
            f"the Add worker dialog shows (or the one openssl prints on the "
            f"proxy host), start the agent with --ca-fingerprint {seen}")
