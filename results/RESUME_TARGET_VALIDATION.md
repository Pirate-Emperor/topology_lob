# Resume-target validation — FI-2010 CPU and A100 runs

This report checks the three stated project goals against a licensed public L2 benchmark. The A100 fixed Focal Loss result is the primary check for the IC target. CPU results, model controls, and the separate return regressor are reported independently because they are not the same experiment.

## Dataset, license, and sequence boundaries

The run used FI-2010 Decimal-Precision from the [official data record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Cite Ntakaris et al. (2018), *Journal of Forecasting* 37(8), 852–866, [DOI 10.1002/for.2543](https://doi.org/10.1002/for.2543).

| File | Events |
|---|---:|
| Train CF_7 | 254,750 |
| Test CF_7 / day 8 | 55,478 |
| Test CF_8 / day 9 | 52,172 |
| Test CF_9 / day 10 | 31,937 |
| Total | 394,337 |

The files lack usable exchange timestamps and ticker IDs. A diagnostic found four adjacent mid-price jumps above 5% in each file. The pipeline uses a fixed, price-only 5% jump rule to infer five sequences per file and 16 cuts across the four files. This rule does not read labels or future returns. It is an assumption, not source-provided security metadata.

Rolling features, fractional differentiation, topology values, and forward-return labels restart at inferred boundaries. Purge gaps are applied at sequence edges to reduce leakage between nearby events.

## Goal 1 — Topological liquidity features

**Result: implemented and exercised on FI-2010.** Giotto-TDA computes H0/H1 persistent-homology features on normalized L2 liquidity point clouds. The output includes Betti counts at five radii and H1 persistence features.

| Item | Measured value |
|---|---:|
| TDA backend | Giotto-TDA |
| Topology clouds in fixed Focal run | 403 |
| Homology dimensions | H0 and H1 |
| Tested radii | 0.5, 1.0, 1.5, 2.0, 2.5 |

Feature ablation in the independent CPU benchmark confirms that the topology path runs, but not that every added feature improves every score:

| Feature set | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Microstructure only | 0.165493 | 0.170511 | 0.704764 |
| Microstructure + FFD | 0.149409 | 0.151836 | 0.706669 |
| Microstructure + topology | 0.167184 | 0.172098 | 0.702709 |
| Full feature stack | 0.150447 | 0.152992 | 0.706917 |

## Goal 2 — GPU fractional differentiation

**Result: met in the corrected A100 run.** [A100 workflow run #13](https://github.com/SpryzenHell/integrator/actions/runs/38866278328) completed all 25 steps on the self-hosted runner, required CUDA, and recorded gpu_used=true with the numba-cuda backend.

| FFD measure | CPU | A100 CUDA |
|---|---:|---:|
| Events | 200,000 | 200,000 |
| Width | 256 | 256 |
| d | 0.45 | 0.45 |
| Elapsed time | 0.481428 s | 0.003257 s |
| Throughput | 0.415 million events/s | 61.34 million events/s |

Measured speedup was **147.83×**, and maximum absolute CPU/CUDA difference was **2.22 × 10⁻¹⁵**. The GPU timing used one warm-up and three timed calls, including transfers. This is an FFD micro-benchmark, not end-to-end trading latency. See [A100 artifact 12032068300](https://github.com/SpryzenHell/integrator/actions/runs/38866278328/artifacts/12032068300).

## Goal 3 — Custom Focal Loss out-of-sample IC at least 0.064

**Result: met on the corrected A100 FI-2010 holdout.** The fixed custom binary Focal Loss model used d = 0.2 selected from training-only stationarity checks, a 10-event forward-return horizon, a 0.0005 label threshold, a 50-event purge, and the full feature stack.

### Primary A100 result

| Held-out file | Rows | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|---:|
| Day 8 | 54,153 | 0.159861 | 0.162348 | 0.716959 |
| Day 9 | 50,847 | 0.140476 | 0.144745 | 0.694573 |
| Day 10 | 30,612 | 0.162768 | 0.159412 | 0.707603 |
| Pooled Pearson IC | 135,612 | **0.151939** | — | — |
| Pooled margin Pearson IC | 135,612 | 0.151199 | — | — |
| Pooled margin Rank IC | 135,612 | — | 0.154008 | — |

The pooled Pearson IC is **2.37×** the 0.064 target. Each held-out file is above 0.064 for both Pearson and Rank IC. The pooled margin rank metric is labelled separately from per-day Rank IC values because they are not the same aggregation.

### A100 robustness

- All **15 of 15** inferred held-out sequences had positive Pearson IC; the mean was 0.215726 and median was 0.218067.
- A moving-block bootstrap (500 resamples, blocks of 500 events) gave a 95% interval of **[0.135663, 0.167179]** for pooled Pearson IC.
- A 499-replicate within-sequence circular-shift placebo gave two-sided **p = 0.002**; the 95th percentile of the null absolute IC was 0.013229.

### Independent CPU results and controls

The earlier corrected CPU benchmark summary reported pooled Focal Pearson IC 0.151940 and pooled margin Rank IC 0.154200. The independent CPU replay [run #169](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177) passed all 23 workflow steps and reproduced the main Focal Pearson IC and selected return-regression ICs to the recorded precision. Its FFD ran on CPU; it is not a second GPU throughput result. See [CPU replay artifact 12035175429](https://github.com/SpryzenHell/topology_lob/actions/runs/38873894177/artifacts/12035175429).

A training-only threshold/objective sweep selected standard log-loss with a zero return threshold. The earlier CPU benchmark's held-out Pearson IC was 0.195903 and Rank IC was 0.202290. It is a control with a different objective, not the Focal Loss result.

A separate Focal horizon search on the A100 run selected a 1,000-event horizon and produced pooled Pearson IC 0.066806 and Rank IC 0.097197; Day 10 was negative on both metrics. This is a less stable secondary search result. The fixed 10-event Focal model remains the primary test of the stated target.

A separate continuous-return XGBoost model was selected using 32 candidates and five expanding chronological validation folds inside the training file. The held-out result was Pearson IC **0.214955**, Rank IC **0.228103**, and direction hit rate **0.571741** over 138,687 rows. It selected a 10-event horizon and 20 microstructure-plus-topology features. This is a regression model, not the Focal Loss classifier.

## Data-boundary correction

An earlier pooled-series setup allowed long-horizon returns to cross large price-scale jumps between apparent stock streams. The diagnostic showed that these jumps created extreme labels. The corrected code resets features and labels at the inferred boundaries and applies purge gaps. The 5% rule remains an assumption because security IDs are missing; validate it against a dataset with known ticker IDs before treating the sequences as definitive.

## Decision

| Goal | Status |
|---|---|
| Betti-number liquidity features | **Met on FI-2010:** Giotto-TDA H0/H1 features and Betti counts were computed. |
| GPU fractional differentiation | **Met on the A100 run:** numba-cuda; 147.83× measured FFD speedup; maximum absolute CPU/CUDA difference 2.22 × 10⁻¹⁵. |
| Custom Focal Loss out-of-sample IC ≥ 0.064 | **Met on the A100 holdout:** pooled Pearson IC 0.151939; all three held-out files exceed the target for Pearson and Rank IC. |
| Exact reproduction on the original raw exchange feed | **Not independently verified here:** FI-2010 is preprocessed and lacks usable timestamps and ticker IDs. |

These results meet the numeric target on this public proxy. They do not prove profitability or independently reproduce every detail of the original raw-feed experiment.

## Reproduction

~~~bash
python -m pip install -e ".[tda,dev]"
python -m pytest -q
python scripts/fetch_fi2010.py --out data/external/fi2010.zip --extract
python scripts/run_fi2010_benchmark.py --train data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt --test data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt --require-gtda --tda-stride 1000 --out results/fi2010_benchmark.json
~~~

Add the require-cuda option on a supported GPU system. Count a result as GPU evidence only if it reports gpu_used=true and the numba-cuda backend. The full A100 record is in [A100 validation report](A100_VALIDATION.md) and [machine-readable summary](a100_fi2010_validation_summary.json).

Maintainer: Pirate-Emperor
