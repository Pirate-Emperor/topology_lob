# A100 validation — FI-2010

**Result: passed.** [GitHub Actions run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328) completed all 25 steps successfully on the self-hosted `integrator-1` runner with the `a100` label. The job verified at least two A100 40 GB GPUs, required CUDA rather than allowing CPU fallback, ran the corrected benchmark and train-only model selections, checked the held-out metrics, and uploaded the evidence.

- Full experiment bundle: [Actions artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300)
- Fixed-model dashboard: [A100 validation dashboard](../assets/figures/a100_fi2010_validation_dashboard.svg)
- Project code and reproduction steps: [README](../README.md), [real-data run guide](../docs/REAL_DATA_RUN.md)

## Dataset and split

The run used the FI-2010 Decimal-Precision benchmark from the [official data record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), licensed **CC BY 4.0**. Cite Ntakaris et al. (2018), *Journal of Forecasting* 37(8), 852–866, [DOI 10.1002/for.2543](https://doi.org/10.1002/for.2543).

| Split | Events |
|---|---:|
| Train CF_7 | 254,750 |
| Test CF_7 / day 8 | 55,478 |
| Test CF_8 / day 9 | 52,172 |
| Test CF_9 / day 10 | 31,937 |
| Total | 394,337 |

The public files lack usable ticker IDs and exchange timestamps. A boundary diagnostic found 16 adjacent mid-price jumps above 5%, so the pipeline applies a fixed, price-only 5% jump rule to infer five sequences in each file. Features and forward-return labels restart at each inferred sequence; a purge gap prevents adjacent train/test leakage. This is an explicit assumption, not recovered stock metadata.

## Fixed custom Focal Loss result

**The corrected A100 run cleared the IC target on each of the three held-out files.**

| Held-out file | Model rows | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|---:|
| Day 8 | 54,153 | 0.159861 | 0.162348 | 0.716959 |
| Day 9 | 50,847 | 0.140476 | 0.144745 | 0.694573 |
| Day 10 | 30,612 | 0.162768 | 0.159412 | 0.707603 |
| **Pooled Pearson IC** | **135,612** | **0.151939** | — | — |
| Pooled margin Rank IC | 135,612 | — | **0.154008** | — |

Configuration: custom binary Focal Loss, fixed 10-event forward log-return horizon, 0.0005 label threshold, 50-event purge, and fractional-differencing order (d=0.2) selected using training-only stationarity checks. The benchmark used `numba-cuda` for fractional differentiation and Giotto-TDA for H0/H1 topology. Giotto-TDA computed 403 topology clouds using Betti features across five radii.

### Robustness checks

- **Sequence check:** all 15 inferred held-out sequences had positive Pearson IC; mean 0.215726, median 0.218067. The pooled score remains lower than the sequence-average view, so both are reported.
- **Moving-block interval:** 500 resamples with 500-event blocks gave a 95% interval of **[0.135663, 0.167179]** for pooled Pearson IC.
- **Circular-shift placebo:** 499 within-sequence score shifts gave two-sided (p=0.002); the observed absolute IC exceeded the 95th percentile of the null absolute IC (0.013229).

These checks support a non-zero association on this benchmark. They do not establish profitability.

## GPU fractional differentiation

The dedicated warmed end-to-end benchmark processed 200,000 events at width 256 and (d=0.45).

| Measure | CPU | A100 CUDA |
|---|---:|---:|
| Elapsed time | 0.481428 s | 0.003257 s |
| Throughput | 0.415 million events/s | 61.34 million events/s |

The measured speedup was **147.83×**. Timing used one warm-up followed by three timed CUDA calls; CUDA timing includes transfers. The maximum absolute difference between CPU and CUDA results was **2.22 × 10⁻¹⁵**. This is a fractional-differentiation micro-benchmark, not end-to-end trading latency.

## Separate continuous-return model

A separate XGBoost regressor was selected using **32 candidates and five expanding chronological validation folds inside the training file**. The held-out files were scored only after selection and refitting.

| Selection / held-out metric | Result |
|---|---:|
| Selected horizon | 10 events |
| Selected feature set | Microstructure + topology |
| Features | 20 |
| Mean validation Pearson IC | 0.198856 ± 0.075433 |
| Mean validation Rank IC | 0.248313 ± 0.028288 |
| Validation folds with positive Pearson IC | 5 / 5 |
| Held-out rows | 138,687 |
| **Held-out Pearson IC** | **0.214955** |
| **Held-out Rank IC** | **0.228103** |
| Direction hit rate | 0.571741 |

| Held-out file | Rows | Pearson IC | Rank IC |
|---|---:|---:|---:|
| Day 8 | 55,178 | 0.214325 | 0.234260 |
| Day 9 | 51,872 | 0.201933 | 0.220425 |
| Day 10 | 31,637 | 0.249459 | 0.240102 |

The selected regressor also exceeds the 0.064 target for both held-out Pearson and Rank IC. Its objective is different from the custom Focal classifier, so these metrics are a separate result, not substituted for the Focal result.

## Horizon-selection warning

A distinct search over Focal horizons selected a 1,000-event horizon with pooled Pearson IC 0.066806 and Rank IC 0.097197. Day 10 was negative on both measures (Pearson −0.045135, Rank −0.042798). This search result is less stable across held-out days; the fixed 10-event result above remains the primary Focal target check.

## CI, images, and reproduction

The latest general CI run, [run #198](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894156), passed package build, all **55 Python tests** (13 library warnings), extended numerical/data experiments, the C++ CPU build and benchmark, and image refresh. The 200,000-event C++ CPU benchmark measured about 6.05 million events/s; do not compare this directly with the CUDA end-to-end FFD number without accounting for implementation and timing differences.

To reproduce the real-data run, follow [the run guide](../docs/REAL_DATA_RUN.md) and [the reproducibility checklist](../docs/REPRODUCIBILITY.md). Add `--require-cuda` and `--require-gtda` for GPU-evidenced results; a CPU fallback must not be described as an A100 run.

**Scope note:** FI-2010 is preprocessed benchmark data, not the original timestamped raw exchange feed. The target is met on this public proxy; this does not independently reproduce the original historical raw-feed experiment or prove a live trading edge.

Maintainer: Pirate-Emperor


## Independent CPU replay

[CPU workflow run #169](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177) passed all 23 steps and published [artifact 12035175429](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177/artifacts/12035175429). A second independent CPU rerun, [run #170](https://github.com/SpryzenHell/topology_lob/actions/runs/38883483885), also passed its benchmark and final model checks and uploaded [artifact 12039460485](https://github.com/SpryzenHell/topology_lob/actions/runs/38883483885/artifacts/12039460485). Both CPU runs reproduced the fixed Focal pooled Pearson IC (0.151939) and selected return-regression Pearson/Rank IC (0.214955 / 0.228103) to recorded precision. They used CPU FFD, so the GPU throughput and CPU/CUDA parity figures in this report come from the A100 workflow only.
