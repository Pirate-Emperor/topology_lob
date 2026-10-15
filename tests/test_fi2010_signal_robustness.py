import numpy as np
import pytest

from scripts.run_fi2010_benchmark import sequence_signal_diagnostics


def make_segments(seed=10, n_segments=5, n=1600):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n_segments):
        score = rng.normal(size=n)
        future = 0.45 * score + rng.normal(scale=0.8, size=n)
        result.append({
            "day": 8 + i // 5,
            "sequence": i % 5 + 1,
            "start": i * n,
            "end": (i + 1) * n,
            "probability": score,
            "future": future,
        })
    return result


def test_sequence_diagnostics_reports_bootstrap_and_shift_placebo():
    report = sequence_signal_diagnostics(
        make_segments(), block_size=100, bootstrap_reps=80, shift_reps=99, seed=30
    )
    assert report["sequence_count"] == 5
    assert report["positive_ic_sequences"] == 5
    assert report["pooled_pearson_ic"] > 0.35
    ci = report["moving_block_bootstrap"]["ci_95"]
    assert ci[0] < report["pooled_pearson_ic"] < ci[1]
    assert report["within_sequence_circular_shift_placebo"]["replicates"] == 99
    assert report["within_sequence_circular_shift_placebo"]["two_sided_p_value"] <= 0.05
    assert report["rows"] == 5 * 1600


def test_sequence_diagnostics_rejects_mismatched_arrays():
    row = make_segments(n_segments=1)[0]
    row["future"] = row["future"][:-1]
    with pytest.raises(ValueError, match="matching 1-D"):
        sequence_signal_diagnostics([row], block_size=100, bootstrap_reps=10, shift_reps=10)


def test_sequence_diagnostics_requires_multiple_sequences():
    row = make_segments(n_segments=1)[0]
    with pytest.raises(ValueError, match="at least three"):
        sequence_signal_diagnostics([row], block_size=100, bootstrap_reps=10, shift_reps=10)
