# Figures and report snapshot

This folder contains plots generated from experiment outputs. The README gallery uses seven SVG plots made by `scripts/generate_readme_plots.py`; the script reads the checked-in A100 and CPU FI-2010 result JSON files. It uses Matplotlib's built-in styles and makes ordinary charts with labels and axes. Synthetic test figures are called out as engineering checks, not market-data results.

## README gallery

| File | What it shows |
|---|---|
| figures/readme_daily_ic.svg | A100 fixed Focal Loss Pearson and Rank IC by held-out day, with the 0.064 target |
| figures/readme_daily_auc.svg | A100 fixed Focal Loss ROC-AUC by held-out day |
| figures/readme_ffd_throughput.svg | CPU vs A100 CUDA fractional-differentiation speed, with log-scaled throughput |
| figures/readme_feature_ablation.svg | CPU FI-2010 reference ablations for microstructure, FFD, topology, and the full feature stack |
| figures/readme_robustness.svg | Pooled A100 Pearson IC and its 95% moving-block bootstrap interval, plus the 0.064 target |
| figures/readme_return_regression.svg | A100 held-out IC by day for the separate training-selected continuous-return regressor |
| figures/readme_stationarity.svg | Training-only ADF p-value and price-correlation trade-off for d=0.2 and d=0.3 |

## Other research and validation figures

| File | What it shows |
|---|---|
| figures/lob_snapshot.svg | A deterministic synthetic L2 snapshot and its liquidity point cloud |
| figures/betti_curves.svg | H0/H1 Betti counts as the Vietoris-Rips radius changes |
| figures/ffd_stationarity.svg | Training-prefix ADF checks for the deterministic synthetic example |
| figures/model_comparison.svg | Focal Loss and log-loss control on a synthetic holdout |
| figures/cpp_benchmark.svg | Standalone C++ CPU fractional-differentiation throughput |
| figures/experimental_validation_dashboard.png | Synthetic feature/objective ablation, seed sensitivity, walk-forward, and placebo results |
| figures/robustness_dashboard.png | Numerical checks and metric sensitivity |
| figures/data_diagnostics_dashboard.png | Synthetic spread, imbalance, volatility, and Betti-1 diagnostics |
| figures/fi2010_validation_dashboard.png | Corrected CPU FI-2010 benchmark results |
| figures/a100_fi2010_validation_dashboard.svg | Detailed A100 FI-2010 validation, daily IC, CUDA/CPU parity, and throughput |
| figures/fi2010_return_regression_selection.png | Training-only selected regression model report |
| screenshots/report_preview.png | Readable summary of a measured synthetic experiment |

The plot gallery is regenerated in CI and validated before the assets are committed. There is intentionally no terminal screenshot and no `main.png` banner.

Maintainer: Pirate-Emperor
