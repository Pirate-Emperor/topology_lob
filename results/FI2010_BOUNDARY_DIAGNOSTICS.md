# FI-2010 price-sequence boundary diagnostic

**Why this exists:** the public FI-2010 Decimal-Precision files contain abrupt price-scale changes but lack usable ticker IDs or exchange timestamps. Returns spanning these jumps can create false long-horizon labels. This report is a data-quality diagnostic, not proof of security identity.

The experiment uses a fixed 5% adjacent mid-price-jump rule before features or labels are constructed. It yields five inferred price sequences per file: 16 cuts across the train file and three test files. Rolling features, fractional differentiation, topology values and forward returns now respect these boundaries.

| File | Events | Jumps above 5% | Detected absolute jumps (approx. %) |
|---|---:|---:|---|
| Train CF_7 | 254,750 | 4 | 52.40, 35.80, 26.55, 173.91 |
| Test CF_7 / day 8 | 55,478 | 4 | 53.73, 32.30, 29.28, 174.52 |
| Test CF_8 / day 9 | 52,172 | 4 | 50.10, 34.16, 27.24, 181.84 |
| Test CF_9 / day 10 | 31,937 | 4 | 50.02, 32.46, 27.07, 175.40 |

The 5% rule is an assumption based on price inputs only. Verify it against a dataset with ticker IDs before treating inferred boundaries as definitive security identities.

At a 1,000-event return horizon, the absolute-return 99th percentile and count of returns with magnitude at least 10% were:

| File | 99th percentile before cuts | 99th percentile after cuts | Count ≥10% before / after |
|---|---:|---:|---:|
| Train CF_7 | 31.32% | 2.16% | 4,000 / 0 |
| Test CF_7 / day 8 | 101.69% | 1.67% | 4,000 / 0 |
| Test CF_8 / day 9 | 103.29% | 1.80% | 4,000 / 0 |
| Test CF_9 / day 10 | 101.70% | 1.35% | 4,000 / 0 |

The drop shows that the concatenated file must not be treated as one uninterrupted path for long-horizon labels. It does not prove that every jump is a stock transition, so the assumption remains visible in the results.

The diagnostic code is scripts/diagnose_fi2010_boundaries.py, with tests in tests/test_fi2010_boundary_diagnostics.py and a dedicated workflow at .github/workflows/topology-lob-fi2010-diagnostics.yml. Full corrected experiment JSON is in [Actions artifact 11986421250](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730/artifacts/11986421250).
