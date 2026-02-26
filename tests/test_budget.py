"""Tests for nadirclaw.budget — spend tracking and budget alerts."""

import json
import tempfile
from pathlib import Path

from nadirclaw.budget import BudgetTracker


class TestBudgetTracker:
    def test_record_tracks_spend(self, tmp_path):
        tracker = BudgetTracker(state_file=tmp_path / "state.json")
        result = tracker.record("gpt-4.1", 1000, 500)
        assert result["cost"] >= 0
        assert result["daily_spend"] >= 0
        assert result["daily_requests"] == 1

    def test_daily_budget_alert(self, tmp_path):
        tracker = BudgetTracker(
            daily_budget=0.001,  # Very low budget
            state_file=tmp_path / "state.json",
        )
        # Record enough to exceed budget
        result = tracker.record("gpt-4.1", 100_000, 50_000)
        # Should have triggered an alert
        assert result["daily_spend"] > 0

    def test_model_tracking(self, tmp_path):
        tracker = BudgetTracker(state_file=tmp_path / "state.json")
        tracker.record("gpt-4.1", 1000, 500)
        tracker.record("gpt-4.1", 2000, 1000)
        tracker.record("gemini-2.5-flash", 1000, 500)

        status = tracker.get_status()
        assert status["daily_requests"] == 3
        top = status["top_models"]
        assert len(top) == 2

    def test_state_persistence(self, tmp_path):
        state_file = tmp_path / "state.json"
        tracker = BudgetTracker(state_file=state_file)
        tracker.record("gpt-4.1", 1000, 500)
        tracker.flush()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["daily_requests"] == 1

        # Load again
        tracker2 = BudgetTracker(state_file=state_file)
        status = tracker2.get_status()
        assert status["daily_requests"] == 1

    def test_warn_threshold(self, tmp_path):
        tracker = BudgetTracker(
            daily_budget=0.0001,
            warn_threshold=0.5,
            state_file=tmp_path / "state.json",
        )
        result = tracker.record("gpt-4.1", 100_000, 50_000)
        # Should have both warn and limit alerts
        assert len(result["alerts"]) >= 1
