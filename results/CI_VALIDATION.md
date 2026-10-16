# Supported-environment CI validation

The latest full CI run, [GitHub Actions run #204](https://github.com/SpryzenHell/topology_lob/actions/runs/38903367175), passed all three jobs on 16 October 2026. It checked the plain benchmark plot and expanded README along with the package, numerical, integration, C++, and image tests.

## Test and build results

| Check | Result |
|---|---|
| Python unit and integration tests | **58 passed in 5.69 seconds** |
| Non-blocking dependency warnings | 13 |
| Python package build | Passed |
| Synthetic end-to-end demo with Giotto-TDA required | Passed |
| TDA backend assertion | Passed |
| Extended numerical/data evidence | Passed |
| Evidence output validation | Passed |
| Evidence artifact upload | Passed |
| C++ CPU build | Passed |
| C++ fractional-differentiation benchmark | Passed |
| SVG/raster asset validation | Passed |
| Analysis-image refresh | Passed |

The workflow tested commit `06646b30ee6da9dc19a208fc5d6466d00b79b40c`. The only subsequent code-tree change was README wording that clarifies which CLI command supports `--require-cuda`; Python code and tests were unchanged.

## Test coverage

The suite checks malformed L2 input, Focal Loss gradient and Hessian values against finite differences, common-price translation invariance, causal fractional-differentiation prefix behaviour, FI-2010 boundary handling, model-selection helpers, sweep timing, report images, and README/repository requirements.

The new plot tests validate that `assets/figures/benchmark_summary.svg` is well-formed SVG, carries the measured per-day IC values and target line, shows CPU/A100 throughput, and includes a title and description. Other tests confirm local README image links resolve, `main.png` stays absent, only one code-license file remains, and author metadata reads `Pirate-Emperor`.

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
elapsed_s=0.038110
events_per_s=5241207.866915
checksum=0.390931
```

This hosted CPU micro-benchmark varies between runners and is not full-application or trading latency.

## Plot and asset checks

The README summary plot is a 1400 × 650 SVG with labelled daily Pearson/Rank IC bars, a dashed 0.064 reference line, and CPU/A100 FFD throughput. The detailed A100 dashboard remains available as a secondary figure. Asset validation completed with `Asset validation: PASS`; the refresh job found no generated image changes to commit.

## Artifacts

- [Latest extended-evidence artifact 12046718253](https://github.com/SpryzenHell/topology_lob/actions/runs/38903367175/artifacts/12046718253)
- [Latest full CI run #204](https://github.com/SpryzenHell/topology_lob/actions/runs/38903367175)
- [A100 FI-2010 validation report](A100_VALIDATION.md)
- [FI-2010 validation report](FI2010_VALIDATION.md)

Maintainer: Pirate-Emperor
