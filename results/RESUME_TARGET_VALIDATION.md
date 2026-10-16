# Resume-target validation — corrected FI-2010 CPU and A100 runs

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

**Result: met on the corrected A100 FI-2010 run.** [Workflow run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328) completed all 25 checks on `integrator-1`, required CUDA, and confirmed `gpu_used=true` with backend `numba-cuda`. The 200,000-event FFD test measured 61.34 million events/s against 0.415 million events/s on CPU (147.83×) with maximum CPU/CUDA absolute error 2.22 × 10⁻¹⁵. The full run is preserved in [artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300).

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
| GPU fractional differentiation | **Met on corrected A100 run**: numba-cuda, 147.83× in the measured FFD micro-benchmark, max error 2.22 × 10⁻¹⁵ |
| Custom Focal Loss out-of-sample IC ≥ 0.064 | **Met on corrected A100 run**: pooled Pearson IC 0.151939; pooled margin Rank IC 0.154008; all three held-out days pass both IC thresholds |
| Original timestamped raw-feed claim | **Not independently reproduced here**: FI-2010 is preprocessed and lacks usable exchange timestamps/ticker IDs |

The corrected FI-2010 benchmark meets the numeric target on A100. It does not establish that the original historical raw-feed experiment was reproduced exactly or prove profitability. FI-2010 has no usable ticker IDs or exchange timestamps, so a fixed price-jump rule infers sequence boundaries.

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

Full A100 JSON experiment outputs are in [Actions artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300); the corrected CPU bundle is in [artifact 12031221876](https://github.com/SpryzenHell/topology_lob/actions/runs/38865530026/artifacts/12031221876). See also [the A100 validation report](A100_VALIDATION.md).


## A100 fixed-model robustness

The fixed custom Focal result had pooled Pearson IC 0.151939 and 95% moving-block bootstrap interval [0.135663, 0.167179] (500 resamples; block size 500). All 15 inferred held-out sequences had positive Pearson IC, and a 499-replicate within-sequence circular-shift placebo gave two-sided p = 0.002. A separate training-only 5-fold XGBoost return regression selected a 10-event horizon and microstructure + topology features; its held-out Pearson IC was 0.214955 and Rank IC 0.228103. It is a different objective from Focal Loss.
