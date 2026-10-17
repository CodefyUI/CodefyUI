"""SPKI sha256 fingerprints of X.509 certificates (#786).

A certificate's fingerprint here is ``sha256:<hex>`` of its DER-encoded
SubjectPublicKeyInfo: the public key, not the certificate, so a certificate
re-issued for the same key (a renewal with ``--reuse-key``) keeps it. It is
what ``kubeadm join --discovery-token-ca-cert-hash`` pins, and what this
``openssl`` pipeline prints on the host that holds the certificate
(:data:`OPENSSL_COMMAND`).

Both ends compute it: the agent of the certificate the control plane
presents, the server of a certificate file it is told about
(``CODEFYUI_TLS_CERT_FILE``). No ``cryptography`` dependency: the few lines
of DER walking below find the SubjectPublicKeyInfo element, and its bytes
are hashed as they are.
"""

from __future__ import annotations

import base64
import hashlib
import re

PREFIX = "sha256:"

#: Computes the fingerprint on the host that holds the certificate (the TLS
#: proxy's), for the leaf certificate first in ``fullchain.pem``.
OPENSSL_COMMAND = (
    "openssl x509 -in /path/to/fullchain.pem -pubkey -noout"
    " | openssl pkey -pubin -outform der"
    " | openssl dgst -sha256"
    " | awk '{print \"sha256:\" $NF}'")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_PEM_CERT = re.compile(
    rb"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.S)


def _element(data: bytes, offset: int) -> tuple[int, int, int]:
    """The DER element at *offset*: ``(tag, content start, content end)``."""
    if offset + 2 > len(data):
        raise ValueError("truncated DER element")
    tag = data[offset]
    length = data[offset + 1]
    start = offset + 2
    if length & 0x80:
        count = length & 0x7F
        if count == 0 or count > 4 or start + count > len(data):
            raise ValueError("unsupported DER length")
        length = int.from_bytes(data[start:start + count], "big")
        start += count
    end = start + length
    if end > len(data):
        raise ValueError("truncated DER element")
    return tag, start, end


def spki_der(cert_der: bytes) -> bytes:
    """The DER SubjectPublicKeyInfo of a DER certificate.

    ``Certificate ::= SEQUENCE { tbsCertificate, ... }`` and
    ``TBSCertificate ::= SEQUENCE { [0] version OPTIONAL, serialNumber,
    signature, issuer, validity, subject, subjectPublicKeyInfo, ... }``.
    """
    tag, start, end = _element(cert_der, 0)
    if tag != 0x30:
        raise ValueError("not a DER certificate")
    tag, offset, tbs_end = _element(cert_der, start)
    if tag != 0x30:
        raise ValueError("not a DER certificate")
    index = 0
    while offset < tbs_end:
        tag, content_start, content_end = _element(cert_der, offset)
        if index == 0 and tag == 0xA0:  # explicit version; not counted
            offset = content_end
            continue
        if index == 5:
            if tag != 0x30:
                raise ValueError("no SubjectPublicKeyInfo")
            return cert_der[offset:content_end]
        index += 1
        offset = content_end
    raise ValueError("no SubjectPublicKeyInfo")


def spki_fingerprint(cert_der: bytes) -> str:
    """``sha256:<hex>`` of the certificate's SubjectPublicKeyInfo."""
    return PREFIX + hashlib.sha256(spki_der(cert_der)).hexdigest()


def pem_certificates(pem: bytes) -> list[bytes]:
    """The DER certificates in a PEM bundle, in file order."""
    return [base64.b64decode(b"".join(block.split()))
            for block in _PEM_CERT.findall(pem)]


def normalize(value: str) -> str:
    """``SHA256:AB:CD...`` / ``abcd...`` -> ``sha256:abcd...``.

    Raises ``ValueError`` for anything that is not 64 hex digits after an
    optional ``sha256:`` (colons and spaces between the digits are dropped,
    the way ``openssl -c`` prints them)."""
    text = value.strip().lower()
    if text.startswith(PREFIX):
        text = text[len(PREFIX):]
    digits = re.sub(r"[\s:]", "", text)
    if _HEX64.fullmatch(digits) is None:
        raise ValueError(
            f"a fingerprint is sha256:<64 hex digits>, got {value!r}")
    return PREFIX + digits
