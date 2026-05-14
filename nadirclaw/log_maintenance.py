"""
Log rotation and pruning for NadirClaw.

Rotates requests.jsonl when it exceeds a size threshold and prunes
old rows from requests.db.  Designed to run once at server startup —
fast no-op when nothing needs work.
"""

import gzip
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("nadirclaw")


def rotate_jsonl(
    log_dir: Path,
    max_size_mb: int = 50,
    retention_days: int = 30,
    compress: bool = True,
) -> None:
    """Rotate requests.jsonl if it exceeds *max_size_mb*.

    The current file is renamed to ``requests.<timestamp>.jsonl[.gz]``
    and a fresh empty file takes its place.  Archived files older than
    *retention_days* are deleted.
    """
    jsonl_path = log_dir / "requests.jsonl"
    if not jsonl_path.exists():
        return

    # --- rotate if over threshold ---
    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    if size_mb >= max_size_mb:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if compress:
            archive = log_dir / f"requests.{stamp}.jsonl.gz"
            with open(jsonl_path, "rb") as f_in, gzip.open(archive, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            archive = log_dir / f"requests.{stamp}.jsonl"
            shutil.copy2(jsonl_path, archive)

        # Truncate the live file (preserves inode for any open handles)
        with open(jsonl_path, "w"):
            pass

        logger.info("Rotated requests.jsonl (%.1f MB) → %s", size_mb, archive.name)

    # --- prune old archives ---
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for p in log_dir.glob("requests.*.jsonl*"):
        if p.name == "requests.jsonl":
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                p.unlink()
                logger.info("Deleted old log archive: %s", p.name)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", p.name, exc)


def prune_sqlite(
    log_dir: Path,
    retention_days: int = 30,
) -> None:
    """Delete rows older than *retention_days* from requests.db."""
    db_path = log_dir / "requests.db"
    if not db_path.exists():
        return

    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "DELETE FROM requests WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        if deleted:
            # VACUUM must run outside a transaction
            conn.execute("VACUUM")
            logger.info("Pruned %d old rows from requests.db", deleted)
        conn.close()
    except sqlite3.OperationalError as exc:
        # Table may not exist yet on a fresh install
        logger.debug("SQLite prune skipped: %s", exc)


def run_maintenance(
    log_dir: Path,
    max_size_mb: int = 50,
    retention_days: int = 30,
    compress: bool = True,
) -> None:
    """Run all log maintenance tasks.  Safe to call on every startup."""
    try:
        rotate_jsonl(log_dir, max_size_mb, retention_days, compress)
    except Exception as exc:
        logger.warning("JSONL rotation failed: %s", exc)

    try:
        prune_sqlite(log_dir, retention_days)
    except Exception as exc:
        logger.warning("SQLite pruning failed: %s", exc)
