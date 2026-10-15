# Resume-target validation record

**Scope:** verify the three stated project targets against reproducible runs, using a licensed public limit-order-book benchmark. This record distinguishes a test result from a training/validation score and does not replace a measurement with the target number.

## Dataset and split

The experiments used FI-2010 Decimal-Precision files from the public [FI-2010 dataset record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649). The dataset is available under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Cite Ntakaris, Magris, Kanniainen, Gabbouj and Iosifidis (2018), *Journal of Forecasting* 37(8), 852–866, [DOI 10.1002/for.2543](https://doi.org/10.1002/for.2543).

| Split | Events |
|---|---:|
| Train CF_7 | 254,750 |
| Test CF_7 (day 8) | 55,478 |
| Test CF_8 (day 9) | 52,172 |
| Test CF_9 (day 10) | 31,937 |
| Combined | 394,337 |

FI-2010 is a preprocessed benchmark, not a raw live exchange feed. Its source files do not supply usable exchange wall-clock timestamps. Raw market data is not committed. The run manifest records file sizes and SHA-256 checksums.

Forward returns are calculated inside each file segment; labels are not allowed to cross segment boundaries. The fixed Focal Loss run uses a 10-event horizon, a 0.0005 forward log-return threshold, and a 50-event purge at the train/test boundary.

## Target 1 — Topological liquidity features

**Result: implemented and exercised.**

Giotto-TDA ran Vietoris–Rips persistent homology on normalized L2 liquidity point clouds. The measured feature set contains Betti-0 and Betti-1 counts at five radii, plus H1 persistence features.

| Run | TDA backend | Topology clouds | Homology |
|---|---|---:|---|
| CPU benchmark run | giotto-tda | 396 | H0, H1 |
| A100 benchmark run | giotto-tda | 395 | H0, H1 |

The cloud count differs slightly across runs because the run revisions and segment-edge filtering differed. These are measured cloud counts, not counts inferred from a figure.

## Target 2 — GPU fractional differentiation

**Result: GPU path and numerical parity verified on hardware.**

The A100 workflow confirmed two NVIDIA A100-SXM4-40GB GPUs, ran the fractional-differentiation kernel, and compared its output to the CPU reference. The maximum absolute difference in the 200,000-event benchmark was \(2.22\times10^{-15}\). The end-to-end demo reported gpu_used=true and fractional_diff_backend=numba-cuda.

The A100 FI-2010 benchmark also reported gpu_used=true, fractional_diff_backend=numba-cuda, and selected_d=0.3. The hosted CPU FI-2010 run used cpu-fallback:no-cuda; CPU results are not counted as GPU evidence.

## Target 3 — Out-of-sample Information Coefficient of 0.064

Pearson IC and Rank IC are both shown because “IC” by itself does not specify which correlation is intended. Both were calculated on held-out rows. No reported metric was set to the target value.

### Custom Focal Loss classifier: fixed benchmark

| Held-out metric | Measured result | Threshold 0.064 |
|---|---:|---|
| Pooled Pearson IC (score vs continuous forward return) | 0.00283 | Not reached |
| Pooled Rank IC (score vs continuous forward return) | 0.16034 | Reached |
| Mean daily Rank IC | 0.16165 | Reached |
| Daily Rank IC standard deviation | 0.01395 | — |

Per-day results:

| Test segment | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Day 8 | 0.00837 | 0.17253 | 0.72308 |
| Day 9 | -0.00224 | 0.14592 | 0.69097 |
| Day 10 | 0.00199 | 0.16651 | 0.70409 |

The A100 run independently reproduced this pattern: pooled Focal Pearson IC **0.00202** and pooled Focal Rank IC **0.15887**, with GPU fractional differentiation and Giotto-TDA enabled. Thus the 0.064 target **is exceeded when IC means Rank IC**, but **is not reached when IC means Pearson IC** for this Focal Loss classifier.

### Training-only selection checks

The threshold/objective sweep chose a 0.0001 label threshold and the log-loss objective using an inner chronological validation split. Its validation Pearson IC was 0.13487, but the held-out Pearson IC was only **0.00116**; the held-out Rank IC was **0.20670**. The validation number is not an out-of-sample result.

A separate Focal Loss horizon/feature selection run picked horizon 1,000 with microstructure plus fractional-difference features. On its held-out rows it scored Pearson IC **0.00844** and Rank IC **0.05194**, so that selected Focal configuration did not reach 0.064 under either definition.

### Continuous-return regression control

A separate experiment selected a standard squared-error XGBoost regressor—not the Focal Loss classifier—by mean Pearson IC over five expanding chronological folds confined to the training file. It evaluated 32 candidates (eight horizons × four feature sets). Each validation fold has a chronological gap, and the final test segments were scored only after the configuration had been selected.

| Item | Measured result |
|---|---:|
| Selected horizon | 500 events |
| Selected features | Microstructure + fractional differentiation |
| Mean Pearson IC over five training-only validation folds | 0.06214 |
| Validation-fold Pearson IC standard deviation | 0.10374 |
| Positive validation folds | 4 / 5 |
| Held-out pooled Pearson IC | 0.16070 |
| Held-out pooled Rank IC | 0.05198 |
| Held-out direction hit rate | 0.49526 |

The pooled held-out Pearson IC exceeds 0.064, but this is a **different objective** from Focal Loss. It is also not uniform by segment: daily Pearson IC values were 0.04247, 0.30574, and 0.06083. The large variation and weak pooled Rank IC mean this should be treated as benchmark evidence requiring further robustness checks, not evidence of a stable trading strategy. This recorded selection run used CPU fractional differentiation; the GPU-verified workflow now repeats the same training-only selection while requiring CUDA before accepting the run.

A separate fixed 32-configuration regression grid produced higher scores for some configurations, but those maximum scores are descriptive only: choosing the highest value after looking at the held-out grid would reuse the test set and is not counted as a valid reproduction.

## Decision

| Stated target | Current evidence |
|---|---|
| Betti-number/liquidity-topology features | **Met on FI-2010 benchmark** using Giotto-TDA H0/H1 features |
| GPU-accelerated fractional differentiation | **GPU kernel and end-to-end path verified** on two A100 40GB devices; CPU fallback is explicitly labelled in CPU runs |
| IC ≥ 0.064 using custom Focal Loss | **Met for Rank IC** (A100 held-out Rank IC 0.15887); **not met for Pearson IC** (A100 held-out Pearson IC 0.00202) |
| Pearson IC ≥ 0.064 with continuous-return regression | **Met in recorded held-out FI-2010 run** (0.16070), but this is a different objective and was CPU FFD; an A100-only rerun is gated separately |

Do not state that the custom Focal Loss model achieved Pearson IC 0.064. If the intended definition is Pearson IC specifically, this benchmark has not verified that exact Focal Loss claim. The original raw-feed result remains dependent on the intended timestamped exchange L2 data and matching evaluation protocol.

## Reproduction commands

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

For GPU validation use the dedicated [A100 Actions workflow](https://github.com/SpryzenHell/integrator/actions/workflows/topology-lob-a100-validation.yml). It must report gpu_used=true and fractional_diff_backend=numba-cuda; otherwise the run fails the GPU acceptance check.
