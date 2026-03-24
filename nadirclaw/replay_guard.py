"""
Opt-in replay attack prevention via request nonces.

When a client sends `X-Request-Nonce`, we track it in-memory for a
configurable window (default 300s). Duplicate nonces within the window
are rejected with 409 Conflict.

Clients that do NOT send the header are unaffected.

Configuration:
  NADIRCLAW_NONCE_WINDOW_SECS: dedup window (default 300)
  NADIRCLAW_NONCE_MAX_SIZE: max stored nonces before eviction (default 50000)
"""

import logging
import os
import time
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger("nadirclaw.replay_guard")

_WINDOW_SECS = int(os.getenv("NADIRCLAW_NONCE_WINDOW_SECS", "300"))
_MAX_SIZE = int(os.getenv("NADIRCLAW_NONCE_MAX_SIZE", "50000"))


class NonceStore:
    """Thread-safe in-memory nonce deduplication store with TTL eviction."""

    def __init__(self, window_secs: int = _WINDOW_SECS, max_size: int = _MAX_SIZE):
        self._lock = threading.Lock()
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self._window = window_secs
        self._max_size = max_size

    def check_and_store(self, nonce: str) -> bool:
        """
        Check if a nonce is fresh (not seen before within the window).

        Returns True if the nonce is fresh (request should proceed).
        Returns False if the nonce is a duplicate (request should be rejected).
        """
        now = time.time()

        with self._lock:
            # Evict expired entries (oldest first, thanks to OrderedDict)
            while self._nonces:
                oldest_key, oldest_time = next(iter(self._nonces.items()))
                if now - oldest_time > self._window:
                    self._nonces.pop(oldest_key)
                else:
                    break

            # Evict oldest if at capacity
            while len(self._nonces) >= self._max_size:
                self._nonces.popitem(last=False)

            # Check for duplicate
            if nonce in self._nonces:
                logger.warning("Replay detected: duplicate nonce %s", nonce[:32])
                return False

            # Store the nonce
            self._nonces[nonce] = now
            return True


# Global singleton
_store = NonceStore()


def check_nonce(nonce: Optional[str]) -> Optional[str]:
    """
    Check a request nonce for replay.

    Args:
        nonce: The X-Request-Nonce header value (None if not provided).

    Returns:
        None if the request should proceed.
        An error message string if the request is a replay.
    """
    if nonce is None:
        return None  # Opt-in: no nonce = no replay checking

    if not nonce or len(nonce) > 256:
        return "Invalid nonce format"

    if not _store.check_and_store(nonce):
        return "Duplicate request nonce (possible replay)"

    return None
