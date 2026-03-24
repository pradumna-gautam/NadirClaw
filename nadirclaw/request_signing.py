"""
HMAC request signing/verification for internal service-to-service communication.

Signs outbound requests and verifies inbound signatures to prevent
request tampering and replay attacks.

Headers:
  X-Signature: HMAC-SHA256 hex digest
  X-Timestamp: Unix epoch seconds

Configuration:
  NADIRCLAW_INTERNAL_HMAC_SECRET: shared secret (required)
  NADIRCLAW_HMAC_CLOCK_SKEW_SECS: max allowed clock skew (default 300)
"""

import hashlib
import hmac as hmac_mod
import logging
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger("nadirclaw.request_signing")

_HMAC_SECRET: Optional[str] = os.getenv("NADIRCLAW_INTERNAL_HMAC_SECRET")
_MAX_CLOCK_SKEW_SECS: int = int(os.getenv("NADIRCLAW_HMAC_CLOCK_SKEW_SECS", "300"))


def sign_request(method: str, path: str, body: bytes) -> Optional[Tuple[str, int]]:
    """Compute HMAC-SHA256 signature for an outbound request.

    Returns (signature_hex, timestamp) or None if no secret is configured.
    """
    if not _HMAC_SECRET:
        return None

    timestamp = int(time.time())
    body_hash = hashlib.blake2b(body, digest_size=32).hexdigest()
    signing_string = f"{method}\n{path}\n{timestamp}\n{body_hash}"

    signature = hmac_mod.new(
        _HMAC_SECRET.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return (signature, timestamp)


def verify_request(
    method: str,
    path: str,
    body: bytes,
    signature_header: str,
    timestamp_header: str,
) -> Optional[str]:
    """Verify an inbound HMAC-SHA256 signature.

    Returns None if valid, or an error message string.
    """
    if not _HMAC_SECRET:
        return "HMAC verification failed: no secret configured"

    # Parse timestamp
    try:
        timestamp = int(timestamp_header)
    except (ValueError, TypeError):
        return "HMAC verification failed: invalid timestamp"

    # Check clock skew (replay prevention)
    now = int(time.time())
    if abs(now - timestamp) > _MAX_CLOCK_SKEW_SECS:
        return "HMAC verification failed: request too old (possible replay)"

    # Recompute signature
    body_hash = hashlib.blake2b(body, digest_size=32).hexdigest()
    signing_string = f"{method}\n{path}\n{timestamp}\n{body_hash}"

    expected = hmac_mod.new(
        _HMAC_SECRET.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison
    if hmac_mod.compare_digest(expected, signature_header):
        return None
    return "HMAC verification failed: signature mismatch"
