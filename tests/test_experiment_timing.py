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
