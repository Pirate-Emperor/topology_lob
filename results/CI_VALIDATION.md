# Supported-environment CI validation

This record captures the latest full CI validation from [GitHub Actions run #198](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894156) on `main`. It covers the new SVG dashboard checks and report-image links.

## Execution environment

| Item | Value |
|---|---|
| Python | 3.11.16 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.11.4 |
| scikit-learn | 1.3.2 |
| statsmodels | 0.14.4 |
| XGBoost | 2.1.4 |
| imbalanced-learn | 0.12.4 |
| Numba | 0.59.1 |
| Matplotlib | 3.8.4 |
| Giotto-TDA | 0.6.2 |
| Pillow | 12.3.0 |
| pytest | 8.4.2 |

## Test and integration gates

- Unit and integration test suite: **55 passed in 4.84s, 13 warnings**
- Synthetic demo with `--require-gtda`: **passed**
- TDA backend assertion: **Giotto-TDA**
- Extended evidence generation: **passed**
- Extended evidence validation: **passed**
- Evidence artifact upload: **passed**
- C++ CPU build: **passed**
- C++ benchmark: **passed**
- Raster asset validation: **passed**
- README asset refresh commit: **passed**

The extended evidence run covers five ablation variants, six synthetic seeds, five chronological walk-forward folds, a label-permutation placebo, Focal Loss gradient/Hessian finite-difference checks, malformed L2 rejection, common-price translation invariance, and causal fractional-differentiation prefix invariance.

## Supported-environment synthetic result

The current supported-environment run used 6,000 deterministic synthetic L2 events and produced:

| Item | Value |
|---|---:|
| Model rows | 5,735 |
| Train rows | 3,994 |
| Test rows | 1,721 |
| Purge gap | 20 |
| Topology clouds | 743 |
| Selected d | 0.1 |
| TDA backend | giotto-tda |
| FFD backend | CPU fallback (no CUDA on runner) |

| Metric | Focal | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.073292 | 0.095368 |
| Rank IC | 0.039856 | 0.036762 |
| ROC-AUC | 0.531075 | 0.528676 |
| PR-AUC | 0.244568 | 0.242809 |
| Log loss | 0.516779 | 0.561761 |

These are synthetic engineering results. They are not a historical or live-market performance claim and do not establish the resume target of 0.064.

## Standalone C++ benchmark

The supported CI runner measured:

```text
events=200000 width=256 d=0.450000
elapsed_s=0.032996
events_per_s=6053625.768610
checksum=0.390931
```

This is a CPU micro-benchmark of the standalone implementation, not a trading-latency result.

## CI artifact

The latest extended-evidence artifact was uploaded successfully from [workflow run #198](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894156), artifact [12034616792](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894156/artifacts/12034616792). It contains the experiment JSON, markdown record, analysis dashboards, and readable report preview.

