# FI-2010 benchmark validation — corrected sequence split

This report summarizes [GitHub Actions run 38760696730](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730). The download, tests, experiments, validation checks, and artifact upload succeeded. The final automatic commit step alone failed because another workflow updated main while the results were being published; artifact **11986421250** contains the full experiment JSON and result image.

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

A prior A100 kernel test matched CPU FFD to about 2.22e-15 maximum absolute error. This corrected FI-2010 run used cpu-fallback:no-cuda. The updated A100 workflow reruns the corrected sequence-safe pipeline while requiring CUDA; until that run completes, do not describe these particular FI-2010 scores as measured on GPU.

## Boundary diagnostic

The earlier pooled-series setup created extreme long-horizon returns across apparent stock transitions. For held-out files, the absolute-return 99th percentile at a 1,000-event horizon was about 101.7%–103.3% before sequence cuts and 1.35%–1.80% afterwards. Returns of magnitude at least 10% fell from 4,000 to zero per file. See [the full boundary diagnostic](FI2010_BOUNDARY_DIAGNOSTICS.md).

Full experiment JSON outputs are in [Actions artifact 11986421250](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730/artifacts/11986421250). Data attribution: Ntakaris et al. (2018), [official FI-2010 record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), CC BY 4.0.
