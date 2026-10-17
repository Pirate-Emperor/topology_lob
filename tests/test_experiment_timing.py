from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_saved_fi2010_sweep_has_valid_or_explicitly_missing_timing() -> None:
    report = json.loads((ROOT / "results" / "fi2010_ic_sweep.json").read_text())
    elapsed = report.get("elapsed_seconds")
    if elapsed is None:
        assert "timing_note" in report
        assert "not reliable" in report["timing_note"]
    else:
        assert isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        assert math.isfinite(elapsed)
        assert elapsed >= 0


def test_incomplete_main_png_is_absent() -> None:
    assert not (ROOT / "main.png").exists()
    assert not (ROOT / "assets" / "main.png").exists()

def test_elapsed_seconds_uses_passed_start_time(monkeypatch) -> None:
    from types import SimpleNamespace
    from scripts import run_fi2010_ic_sweep as sweep

    monkeypatch.setattr(sweep, "time", SimpleNamespace(perf_counter=lambda: 18.5))
    assert sweep.elapsed_seconds_since(10.0) == 8.5


def test_elapsed_seconds_rejects_negative_duration(monkeypatch) -> None:
    from types import SimpleNamespace
    import pytest
    from scripts import run_fi2010_ic_sweep as sweep

    monkeypatch.setattr(sweep, "time", SimpleNamespace(perf_counter=lambda: 1.0))
    with pytest.raises(RuntimeError, match="Invalid monotonic elapsed time"):
        sweep.elapsed_seconds_since(2.0)
