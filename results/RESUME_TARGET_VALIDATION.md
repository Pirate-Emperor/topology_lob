# Resume-target validation — corrected FI-2010 run

**Scope:** test the stated Topology LOB objectives against a licensed public limit-order-book benchmark. Pearson IC and Rank IC are both reported. FI-2010 is a proxy for the original raw-feed experiment, not a substitute for it.

## Dataset, license, and split

The run used the FI-2010 Decimal-Precision files from the [official data record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Cite Ntakaris et al. (2018), *Journal of Forecasting* 37(8), 852–866, [DOI 10.1002/for.2543](https://doi.org/10.1002/for.2543).

| File | Events |
|---|---:|
| Train CF_7 | 254,750 |
| Test CF_7 / day 8 | 55,478 |
| Test CF_8 / day 9 | 52,172 |
| Test CF_9 / day 10 | 31,937 |
| Total | 394,337 |

The public files do not provide usable exchange timestamps or ticker IDs. A diagnostic found four adjacent mid-price jumps above 5% in each file (roughly 27%–182%), consistent with five stock streams concatenated together. The pipeline uses a fixed, price-only 5% jump rule to infer sequence boundaries: five sequences per file and 16 cuts across the four files. This rule does not read labels or future returns. It is a documented assumption because security IDs are missing.

Rolling features, fractional differentiation, topology values, and forward-return labels now restart at inferred sequence boundaries. Purge gaps are applied at sequence edges. The fixed 10-event Focal benchmark uses a 0.0005 forward log-return label threshold and a 50-event purge.

## Target 1 — Topological liquidity features

**Result: implemented and exercised on FI-2010.** Giotto-TDA ran Vietoris–Rips persistent homology for H0/H1 on normalized L2 liquidity point clouds. The output includes Betti-0 and Betti-1 counts at five radii and H1 persistence features.

| Item | Corrected CPU benchmark |
|---|---:|
| TDA backend | Giotto-TDA |
| Topology clouds | 403 |
| Homology | H0 and H1 |
| Selected fractional-differencing order | d = 0.2 |

Feature ablation confirms the topology path was exercised, but not that every feature improves every score. Pooled Focal Pearson IC was 0.16549 for microstructure-only, 0.16718 for microstructure plus topology, 0.14941 for microstructure plus FFD, and 0.15045 for the full stack.

## Target 2 — GPU fractional differentiation

**Result: GPU kernel parity was previously verified on A100 hardware; the corrected FI-2010 score in this record used CPU.**

The prior A100 kernel check reported maximum absolute difference of approximately 2.22e-15 between CUDA and CPU output. A separate older A100 demo reported gpu_used=true and numba-cuda, but it preceded the stock-sequence correction.

The corrected FI-2010 benchmark reports gpu_used=false and cpu-fallback:no-cuda because it ran on a hosted CPU runner. Therefore the results below validate the corrected split and model on CPU, not the exact corrected benchmark on A100. The separate A100 workflow requires CUDA, checks GPU hardware and parity, and reruns FI-2010.

## Target 3 — Out-of-sample Information Coefficient of 0.064

**Result: the fixed custom Focal Loss model exceeds the 0.064 target for Pearson IC on the corrected FI-2010 holdout.**

The fixed model uses the custom binary Focal Loss objective, full feature stack, training-only d = 0.2, a 10-event horizon, and a 0.0005 label threshold. The setup was not chosen by searching the held-out files.

| Held-out metric | Measured result | Target |
|---|---:|---|
| Pooled Pearson IC (score vs continuous forward return) | **0.151940** | Reached |
| Pooled Rank IC | 0.154200 | Reached |
| Mean daily Pearson IC | 0.154368 | Reached |
| Daily Pearson IC standard deviation | 0.012118 | — |
| Mean daily Rank IC | 0.155689 | Reached |

| Held-out file | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Day 8 | 0.159861 | 0.162500 | 0.716958 |
| Day 9 | 0.140476 | 0.145006 | 0.694572 |
| Day 10 | 0.162768 | 0.159560 | 0.707603 |

The target is exceeded on each of the three held-out files for both Pearson and Rank IC. This is benchmark evidence, not proof of profitability or a direct reproduction of the original raw-feed experiment.

### Separate controls — not the Focal Loss claim

A training-only threshold/objective sweep selected standard log-loss with a zero return threshold. Held-out Pearson IC was 0.195903 and Rank IC was 0.202290. This is a control model with a different objective.

A separate continuous-return XGBoost model, selected using five chronological validation folds confined to training sequences, obtained held-out Pearson IC 0.214955 and Rank IC 0.228299. Its selected horizon was 10 events and its feature set was microstructure plus topology. It is a regression model, not the Focal Loss classifier.

A separate Focal horizon/feature sweep selected a 1,000-event horizon during training validation and scored Pearson IC 0.067445 overall, but day 10 was negative. The fixed 10-event Focal result and the per-day scores above remain the main metric evidence.

## Data-boundary correction

An earlier setup let long-horizon forward returns cross large price-scale jumps between concatenated stock streams. The diagnostic showed that these jumps created extreme forward-return labels that disappeared when returns were restricted to the inferred sequence. Scores from the older pooled-series setup are superseded. The current run resets features and labels per sequence and uses purge gaps. See [FI-2010 boundary diagnostics](FI2010_BOUNDARY_DIAGNOSTICS.md).

## Decision

| Objective | Status |
|---|---|
| Betti-number liquidity features | **Met on FI-2010**: Giotto-TDA H0/H1 features were computed |
| GPU fractional differentiation | **Kernel parity verified in a prior A100 run**; corrected FI-2010 A100 rerun still pending |
| Custom Focal Loss out-of-sample IC ≥ 0.064 | **Met on corrected FI-2010 CPU run**: Pearson IC 0.151940; Rank IC 0.154200 |
| Original timestamped raw-feed claim | **Not independently reproduced here**: FI-2010 is preprocessed and lacks usable exchange timestamps/ticker IDs |

The corrected FI-2010 benchmark meets the numeric target. It does not establish that the original historical raw-feed experiment was reproduced exactly. The A100 workflow must complete on the corrected code before these exact scores can be described as GPU-measured.

## Reproduction

~~~bash
python -m pip install -e ".[tda,dev]"
python -m pytest -q
python scripts/fetch_fi2010.py --out data/external/fi2010.zip --extract

python scripts/run_fi2010_benchmark.py \
  --train data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --test \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt \
  --require-gtda --tda-stride 1000 \
  --out results/fi2010_benchmark.json
~~~

Add --require-cuda on a supported CUDA device. A run counts as GPU evidence only if it reports gpu_used=true and fractional_diff_backend=numba-cuda.

Full JSON experiment outputs were uploaded as [Actions artifact 11986421250](https://github.com/SpryzenHell/topology_lob/actions/runs/38760696730/artifacts/11986421250).
