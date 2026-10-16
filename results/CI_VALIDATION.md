# Supported-environment CI validation

The latest full CI run, [GitHub Actions run #199](https://github.com/SpryzenHell/topology_lob/actions/runs/38883483918), passed on the commit that added a regression test for the README's measured IC claims. All Python, package-build, extended-experiment, and image-refresh jobs succeeded. The current source and test files have not changed since that run; later commits were documentation-only.

## Execution environment

| Item | Value |
|---|---|
| Python | 3.11.17 |
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

- Unit and integration tests: **56 passed in 3.92 seconds; 13 warnings**.
- Python package build and wheel/source archive checks: **passed**.
- Synthetic end-to-end demo requiring Giotto-TDA: **passed**.
- Full TDA backend assertion: **passed**.
- Extended evidence generation and output validation: **passed**.
- Numerical checks for the custom Focal Loss gradient and Hessian: **passed**.
- C++ CPU build and benchmark: **passed**.
- Raster and SVG figure validation, required dashboard labels, title/description, and README image links: **passed**.
- Refresh of readable report images: **passed**.

The extended evidence run checks five ablation variants, six synthetic seeds, five chronological walk-forward folds, a label-permutation placebo, malformed L2 rejection, common-price translation invariance, and causal fractional-differentiation prefix invariance.

## Supported-environment synthetic result

The demo uses 6,000 generated L2 events and a chronological holdout with a purge gap. The saved example has 5,735 model rows, 3,994 training rows and 1,721 test rows. The selected fractional-differencing order is d = 0.1, with Giotto-TDA enabled.

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.073292 | 0.095368 |
| Rank IC | 0.039856 | 0.036762 |
| ROC-AUC | 0.531075 | 0.528676 |
| PR-AUC | 0.244568 | 0.242809 |
| Log loss | 0.516779 | 0.561761 |

These are engineering results on synthetic data, not market results and not evidence of the FI-2010 target.

## Standalone C++ benchmark

The supported CI runner measured:

~~~text
events=200000 width=256 d=0.450000
elapsed_s=0.038120
events_per_s=5239926.706871
checksum=0.390931
~~~

This is a CPU micro-benchmark of the standalone implementation, not an end-to-end trading-latency measurement. Runtime varies with the hosted runner.

## Dashboard and report checks

The A100 dashboard passed the asset validator with a 1600 × 1120 canvas, viewBox 0 0 1600 1120, 63 text nodes, a title and description, and all required labels present. The report uses high-contrast text rather than a terminal capture. The image-refresh job completed successfully.

## CI artifact

The run uploaded [extended evidence artifact 12038625972](https://github.com/SpryzenHell/topology_lob/actions/runs/38883483918/artifacts/12038625972). It contains synthetic experiment JSON, figures, and the readable report preview.

The independent real-data FI-2010 workflow is tracked separately; its result should be considered complete only after all benchmark, sweep, validation, artifact-upload, and publish steps finish.
