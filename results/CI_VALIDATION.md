# Supported-environment CI validation

The latest full CI run, [GitHub Actions run #215](https://github.com/SpryzenHell/topology_lob/actions/runs/38915647841), passed all three jobs on 16 October 2026. It generated the seven README plots, ran the test and numerical checks, built the Python package and C++ implementation, validated the figures, and committed the generated plot files.

## Test and build results

| Check | Result |
|---|---|
| Python unit and integration tests | **58 passed in 3.11 seconds** |
| Non-blocking dependency warnings | 13 |
| Seven-plot README gallery generation | Passed |
| Python package build | Passed |
| Synthetic end-to-end demo with Giotto-TDA required | Passed |
| TDA backend assertion | Passed |
| Extended numerical/data evidence | Passed |
| Evidence output validation and upload | Passed |
| C++ CPU build and benchmark | Passed |
| Detailed SVG/raster asset validation | Passed |
| All seven gallery SVG label checks | Passed |
| Asset refresh and commit to `main` | Passed |

## README plot gallery

The reproducible script `scripts/generate_readme_plots.py` writes seven SVG charts from the checked-in A100 and CPU FI-2010 result records:

1. Daily Pearson and Rank IC for the fixed Focal Loss classifier, with the 0.064 target line.
2. Daily ROC-AUC for the fixed classifier.
3. CPU versus A100 CUDA fractional-differentiation throughput.
4. CPU reference feature ablations.
5. Pooled Pearson IC and its moving-block bootstrap interval.
6. Per-day Pearson and Rank IC for the separately selected continuous-return regressor.
7. The training-only fractional-differencing stationarity/price-retention trade-off.

The charts use different built-in Matplotlib styles but a restrained plot-first layout: labels, bars or lines, standard axes, and clear legends. They are generated before tests run and regenerated in the image-refresh job. The asset validator checks each SVG's XML and required data labels. The CI run reported `Asset validation: PASS` and committed the seven files to the repository.

To reproduce them locally:

~~~bash
python -m pip install -e .
python scripts/generate_readme_plots.py
python scripts/validate_assets.py
python -m pytest -q
~~~

## Test coverage

The suite checks malformed L2 input, Focal Loss gradient and Hessian values against finite differences, common-price translation invariance, causal fractional-differentiation prefix behaviour, FI-2010 boundary handling, model-selection helpers, sweep timing, report images, local Markdown/HTML image links, the required seven-chart gallery, author metadata, removal of the obsolete single summary image, absence of `main.png`, and the single retained code-license file.

The extended synthetic evidence run includes five ablation variants, six deterministic seeds, five chronological walk-forward folds, a label-permutation placebo, malformed-book checks, translation invariance, and causal FFD-prefix checks. Custom Focal gradient and Hessian absolute errors were approximately `7.41e-12` and `1.06e-11`; translation and causal-prefix checks had zero recorded error.

## Synthetic engineering result

The CI demo uses 6,000 generated events, a chronological holdout, a 20-event purge gap, and 5,735 model rows (3,994 train and 1,721 test).

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.073292 | 0.095368 |
| Rank IC | 0.039856 | 0.036762 |
| ROC-AUC | 0.531075 | 0.528676 |
| PR-AUC | 0.244568 | 0.242809 |
| Log loss | 0.516779 | 0.561761 |

These are engineering checks on synthetic data. They are not FI-2010 results and do not establish that the real-data IC target has been met.

## C++ CPU benchmark

The latest CI run recorded the following for 200,000 events, width 256, and `d=0.45`:

```text
events=200000 width=256 d=0.450000
elapsed_s=0.044491
events_per_s=4489531.129053
checksum=0.390931
```

This hosted CPU micro-benchmark varies between runners and is not full-application or trading latency.

## Artifacts

- [Latest extended-evidence artifact 12052800538](https://github.com/SpryzenHell/topology_lob/actions/runs/38915647841/artifacts/12052800538)
- [Latest full CI run #215](https://github.com/SpryzenHell/topology_lob/actions/runs/38915647841)
- [A100 FI-2010 validation report](A100_VALIDATION.md)
- [FI-2010 validation report](FI2010_VALIDATION.md)

Maintainer: Pirate-Emperor
