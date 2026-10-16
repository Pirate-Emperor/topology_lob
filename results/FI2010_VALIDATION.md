# FI-2010 benchmark validation — corrected sequence split

This report covers three separate records: the corrected A100 run #13, the earlier corrected CPU reference run #121, and the independent CPU replay #169. The per-day table, feature ablation, and threshold-control results below are explicitly from the earlier CPU reference run; the exact A100 measurements are in the [A100 validation report](A100_VALIDATION.md). Full bundles: [A100 artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300), [CPU reference artifact 11986421250](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730/artifacts/11986421250), and [independent CPU replay artifact 12035175429](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177/artifacts/12035175429).

## Data checks

- Dataset: FI-2010 Decimal-Precision, licensed CC BY 4.0.
- Training / held-out events: 254,750 / 55,478, 52,172 and 31,937.
- SHA-256 hashes and download source are recorded in the data manifest.
- The boundary diagnostic found four adjacent mid-price jumps above 5% in each file. A fixed, price-only 5% rule produces five inferred sequences per file and 16 cuts total.
- Feature windows, FFD, topology filling and forward-return targets restart at inferred sequence boundaries. Purge gaps are applied at sequence edges.

The sequence split is an explicit assumption: the public benchmark lacks ticker IDs. It avoids treating the concatenated data as one uninterrupted price path.

## Fixed custom Focal Loss result — CPU reference run

| Held-out file | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Day 8 | 0.159861 | 0.162500 | 0.716958 | 0.242248 | 0.631763 |
| Day 9 | 0.140476 | 0.145006 | 0.694572 | 0.224194 | 0.643348 |
| Day 10 | 0.162768 | 0.159560 | 0.707603 | 0.249822 | 0.634141 |
| Pooled Pearson IC | **0.151940** | | | | |
| Pooled margin Rank IC | | **0.154200** | | | |

Configuration: fixed custom binary Focal Loss, full feature stack, training-only d = 0.2, 10-event horizon, 0.0005 label threshold and 50-event purge. TDA used Giotto-TDA and evaluated 403 clouds. This CPU reference run used CPU FFD because that runner had no CUDA.

## Feature ablation — CPU reference run

| Feature set | Focal Pearson IC | Focal Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Microstructure only | 0.165493 | 0.170511 | 0.704763 |
| Microstructure + FFD | 0.149409 | 0.151836 | 0.706668 |
| Microstructure + topology | 0.167184 | 0.172098 | 0.702709 |
| Full stack | 0.150447 | 0.152992 | 0.706917 |

This confirms that the topology implementation runs, but the full feature stack does not maximize each metric in this test.

## Training-only controls — CPU reference run

The threshold/objective sweep selected standard log-loss with a zero return threshold inside the training-only validation split. Its held-out Pearson IC was 0.195903 and Rank IC 0.202290. This is a control model, not the Focal Loss claim.

The separate continuous-return XGBoost model selected by five training-only chronological folds scored held-out Pearson IC 0.214955 and Rank IC 0.228299 in this CPU reference report. It uses a different objective, so it is reported separately.

## Corrected A100 result

The corrected FI-2010 run completed all 25 steps on the A100 runner in [integrator run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328). It reported gpu_used=true, FFD backend numba-cuda, and Giotto-TDA. The fixed 10-event custom Focal model reached pooled Pearson IC 0.151939; per-day Pearson IC was 0.159861 / 0.140476 / 0.162768. The exact A100 pooled margin Rank IC was 0.154008. Each held-out file exceeded the 0.064 target on Pearson and Rank IC. See the [A100 report](A100_VALIDATION.md), [dashboard](../assets/figures/a100_fi2010_validation_dashboard.svg), and [machine-readable summary](a100_fi2010_validation_summary.json).

The 200,000-event FFD benchmark processed about 61.34 million events/s on CUDA versus 0.415 million events/s on CPU (147.83× in this measured run); maximum CPU/CUDA absolute error was 2.22 × 10⁻¹⁵. Timing was one warm-up plus three timed end-to-end calls including transfers.

## Boundary diagnostic

The earlier pooled-series setup created extreme long-horizon returns across apparent stock transitions. For held-out files, the absolute-return 99th percentile at a 1,000-event horizon was about 101.7%–103.3% before sequence cuts and 1.35%–1.80% afterwards. Returns of magnitude at least 10% fell from 4,000 to zero per file. See [the full boundary diagnostic](FI2010_BOUNDARY_DIAGNOSTICS.md).

## Independent CPU replay

[CPU workflow run #169](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177) passed all 23 steps. It independently reproduced the main Focal Pearson IC and selected return-regression ICs to the recorded precision. It used CPU FFD, so the CUDA speed and parity claims remain based only on A100 run #13.

The dataset is preprocessed, and inferred boundaries are based on a fixed price-only rule because usable ticker IDs are missing. These results meet the numeric target on this public proxy; they do not prove profitability or reproduce the original raw exchange feed in full.

Maintainer: Pirate-Emperor
