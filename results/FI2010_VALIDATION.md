# FI-2010 benchmark validation — corrected sequence split

This report records the corrected A100 experiment ([run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328)) and the independent CPU replay ([run #169](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177)). The complete A100 JSON and images are in [artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300); the latest CPU bundle is [artifact 12035175429](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177/artifacts/12035175429).

## Data checks

- Dataset: FI-2010 Decimal-Precision, licensed CC BY 4.0.
- Training / held-out events: 254,750 / 55,478, 52,172 and 31,937.
- SHA-256 hashes and download source are recorded in the data manifest.
- The boundary diagnostic found four adjacent mid-price jumps above 5% in each file. A fixed, price-only 5% rule produces five inferred sequences per file and 16 cuts total.
- Feature windows, FFD, topology filling and forward-return targets restart at inferred sequence boundaries. Purge gaps are applied at sequence edges.

The sequence split is an explicit assumption: the public benchmark lacks ticker IDs. It avoids treating the concatenated data as one uninterrupted price path.

## Fixed custom Focal Loss result

| Held-out file | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Day 8 | 0.159861 | 0.162500 | 0.716958 | 0.242248 | 0.631763 |
| Day 9 | 0.140476 | 0.145006 | 0.694572 | 0.224194 | 0.643348 |
| Day 10 | 0.162768 | 0.159560 | 0.707603 | 0.249822 | 0.634141 |
| Pooled Pearson IC | **0.151940** | | | | |
| Pooled Rank IC | | **0.154200** | | | |

Configuration: fixed custom binary Focal Loss, full feature stack, training-only d = 0.2, 10-event horizon, 0.0005 label threshold and 50-event purge. TDA used Giotto-TDA and evaluated 403 clouds. The corrected run used CPU FFD because the hosted runner had no CUDA.

## Feature ablation

| Feature set | Focal Pearson IC | Focal Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Microstructure only | 0.165493 | 0.170511 | 0.704763 |
| Microstructure + FFD | 0.149409 | 0.151836 | 0.706668 |
| Microstructure + topology | 0.167184 | 0.172098 | 0.702709 |
| Full stack | 0.150447 | 0.152992 | 0.706917 |

This confirms that the topology implementation runs, but the full feature stack does not maximize each metric in this test.

## Training-only controls

The threshold/objective sweep selected standard log-loss with a zero return threshold inside the training-only validation split. Its held-out Pearson IC was 0.195903 and Rank IC 0.202290. This is a control model, not the Focal Loss claim.

The separate continuous-return XGBoost model selected by five training-only chronological folds scored held-out Pearson IC 0.214955 and Rank IC 0.228299. It uses a different objective, so it is reported separately.

The fixed Focal scores above are the main comparison for the custom-objective target. The 0.064 target was not used as a fitting value.

## GPU status

The corrected FI-2010 run completed on A100 in [integrator run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328). It reported `gpu_used=true`, FFD backend `numba-cuda`, and Giotto-TDA. The 200,000-event FFD benchmark processed about 61.34 million events/s on CUDA versus 0.415 million events/s on CPU (147.83× in this measured run); maximum CPU/CUDA absolute error was 2.22 × 10⁻¹⁵. Timing was one warm-up plus three timed end-to-end calls including transfers.

For the fixed custom Focal result, pooled Pearson IC was 0.151939 and each of the three held-out files exceeded 0.064. See the [full A100 report](A100_VALIDATION.md) and [dashboard](../assets/figures/a100_fi2010_validation_dashboard.svg).

## Boundary diagnostic

The earlier pooled-series setup created extreme long-horizon returns across apparent stock transitions. For held-out files, the absolute-return 99th percentile at a 1,000-event horizon was about 101.7%–103.3% before sequence cuts and 1.35%–1.80% afterwards. Returns of magnitude at least 10% fell from 4,000 to zero per file. See [the full boundary diagnostic](FI2010_BOUNDARY_DIAGNOSTICS.md).

Full experiment JSON outputs are in [Actions artifact 11986421250](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730/artifacts/11986421250). Data attribution: Ntakaris et al. (2018), [official FI-2010 record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), CC BY 4.0.


## Corrected A100 result

The exact corrected sequence-safe pipeline passed all 25 workflow steps. The fixed 10-event Focal model reached pooled Pearson IC 0.151939, with 0.159861 / 0.140476 / 0.162768 on days 8–10. The separately selected continuous-return model passed five train-only chronological folds and reached held-out Pearson IC 0.214955 and Rank IC 0.228103. These are separate objectives and should remain labelled separately. The preprocessed dataset and inferred stock boundaries limit what can be claimed about the original raw-feed experiment.


## Independent CPU replay

A later CPU run completed all 23 workflow steps successfully. It reproduced the fixed Focal pooled Pearson IC (0.151939), per-day Focal ICs (0.159861 / 0.140476 / 0.162768), and selected continuous-return regression ICs (Pearson 0.214955; Rank 0.228103) to the precision recorded in the A100 artifact. This confirms the score calculations and train-only selection are reproducible. It used CPU FFD, so the CUDA speed and parity claims remain based on A100 run #13. See [run #169](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177) and [artifact 12035175429](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177/artifacts/12035175429).
