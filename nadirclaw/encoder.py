"""Shared SentenceTransformer singleton for NadirClaw.

The encoder is loaded lazily on first use — not at import time.
This avoids the ~500ms cold-start penalty when running commands that
don't need classification (e.g. ``nadirclaw serve`` before the first request).
"""

import logging
import os
import time
from threading import Lock

logger = logging.getLogger(__name__)

_shared_encoder = None  # type: ignore[assignment]
_encoder_lock = Lock()


def get_shared_encoder_sync():
    """
    Lazily initialize and return a shared SentenceTransformer instance.
    The first call loads the model (~80 MB download on first run).
    Uses double-checked locking to avoid redundant loads.

    The ``sentence_transformers`` import itself is deferred so that
    ``import nadirclaw`` does not trigger a heavy torch import chain.
    """
    global _shared_encoder
    if _shared_encoder is None:
        with _encoder_lock:
            if _shared_encoder is None:
                t0 = time.time()
                logger.info("Loading SentenceTransformer encoder: all-MiniLM-L6-v2")

                # Suppress noisy tokenizer parallelism warning
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

                from sentence_transformers import SentenceTransformer

                _shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                elapsed = int((time.time() - t0) * 1000)
                logger.info("SentenceTransformer encoder loaded (%dms)", elapsed)
    return _shared_encoder
