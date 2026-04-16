"""Tests for nadirclaw.log_maintenance."""

import gzip
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nadirclaw.log_maintenance import prune_sqlite, rotate_jsonl, run_maintenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, size_mb: float) -> None:
    """Write a JSONL file of approximately *size_mb* megabytes."""
    line = json.dumps({"msg": "x" * 200}) + "\n"
    target_bytes = int(size_mb * 1024 * 1024)
    with open(path, "w") as f:
        while f.tell() < target_bytes:
            f.write(line)


def _create_requests_db(db_path: Path, rows: list[tuple[str, str]]) -> None:
    """Create a minimal requests table with (timestamp, model) rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (timestamp TEXT, model TEXT)"
    )
    conn.executemany("INSERT INTO requests VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# rotate_jsonl
# ---------------------------------------------------------------------------

class TestRotateJsonl:
    def test_no_rotation_when_under_threshold(self, tmp_path: Path):
        jsonl = tmp_path / "requests.jsonl"
        jsonl.write_text('{"a":1}\n')

        rotate_jsonl(tmp_path, max_size_mb=50)

        assert jsonl.exists()
        assert list(tmp_path.glob("requests.*.jsonl*")) == []

    def test_rotation_with_gzip(self, tmp_path: Path):
        jsonl = tmp_path / "requests.jsonl"
        _write_jsonl(jsonl, size_mb=1.1)

        rotate_jsonl(tmp_path, max_size_mb=1, compress=True)

        # Live file should be empty now
        assert jsonl.stat().st_size == 0

        # Should have one .gz archive
        archives = list(tmp_path.glob("requests.*.jsonl.gz"))
        assert len(archives) == 1

        # Archive should be valid gzip containing JSONL
        with gzip.open(archives[0], "rt") as f:
            first_line = f.readline()
        assert json.loads(first_line)["msg"]

    def test_rotation_without_compression(self, tmp_path: Path):
        jsonl = tmp_path / "requests.jsonl"
        _write_jsonl(jsonl, size_mb=1.1)

        rotate_jsonl(tmp_path, max_size_mb=1, compress=False)

        assert jsonl.stat().st_size == 0
        archives = list(tmp_path.glob("requests.*.jsonl"))
        # Filter out the live file
        archives = [a for a in archives if a.name != "requests.jsonl"]
        assert len(archives) == 1

    def test_old_archives_deleted(self, tmp_path: Path):
        jsonl = tmp_path / "requests.jsonl"
        jsonl.write_text("")

        # Create a fake old archive with mtime 60 days ago
        old_archive = tmp_path / "requests.20250101T000000Z.jsonl.gz"
        old_archive.write_bytes(b"old")
        old_mtime = time.time() - (60 * 86400)
        import os
        os.utime(old_archive, (old_mtime, old_mtime))

        # Create a recent archive
        new_archive = tmp_path / "requests.20260401T000000Z.jsonl.gz"
        new_archive.write_bytes(b"new")

        rotate_jsonl(tmp_path, max_size_mb=999, retention_days=30)

        assert not old_archive.exists()
        assert new_archive.exists()

    def test_noop_when_no_file(self, tmp_path: Path):
        rotate_jsonl(tmp_path, max_size_mb=1)  # should not raise


# ---------------------------------------------------------------------------
# prune_sqlite
# ---------------------------------------------------------------------------

class TestPruneSqlite:
    def test_prune_old_rows(self, tmp_path: Path):
        db = tmp_path / "requests.db"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()

        _create_requests_db(db, [
            (old_ts, "gpt-4"),
            (old_ts, "claude-3"),
            (new_ts, "gpt-4"),
        ])

        prune_sqlite(tmp_path, retention_days=30)

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        conn.close()
        assert count == 1  # only the recent row remains

    def test_noop_when_all_recent(self, tmp_path: Path):
        db = tmp_path / "requests.db"
        new_ts = datetime.now(timezone.utc).isoformat()
        _create_requests_db(db, [(new_ts, "gpt-4")])

        prune_sqlite(tmp_path, retention_days=30)

        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        conn.close()
        assert count == 1

    def test_noop_when_no_db(self, tmp_path: Path):
        prune_sqlite(tmp_path, retention_days=30)  # should not raise

    def test_noop_when_no_table(self, tmp_path: Path):
        db = tmp_path / "requests.db"
        conn = sqlite3.connect(str(db))
        conn.close()

        prune_sqlite(tmp_path, retention_days=30)  # should not raise


# ---------------------------------------------------------------------------
# run_maintenance
# ---------------------------------------------------------------------------

class TestRunMaintenance:
    def test_orchestrates_both(self, tmp_path: Path):
        # Set up JSONL over threshold
        jsonl = tmp_path / "requests.jsonl"
        _write_jsonl(jsonl, size_mb=1.1)

        # Set up SQLite with old rows
        db = tmp_path / "requests.db"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _create_requests_db(db, [(old_ts, "gpt-4")])

        run_maintenance(tmp_path, max_size_mb=1, retention_days=30, compress=True)

        # JSONL rotated
        assert jsonl.stat().st_size == 0
        assert len(list(tmp_path.glob("requests.*.jsonl.gz"))) == 1

        # SQLite pruned
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        conn.close()
        assert count == 0

    def test_handles_missing_dir_gracefully(self, tmp_path: Path):
        empty = tmp_path / "nonexistent"
        empty.mkdir()
        run_maintenance(empty, max_size_mb=50, retention_days=30)  # no crash
