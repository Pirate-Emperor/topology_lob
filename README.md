# Topology LOB

A reproducible research pipeline for Level-2 limit-order-book data. It combines liquidity-shape features, fractional differentiation, and rare-event prediction with XGBoost.

![Experimental validation dashboard](assets/figures/experimental_validation_dashboard.png)

## Pipeline

1. Validate L2 snapshots, timestamps, prices, and depth sizes.
2. Build spread, imbalance, microprice, order-flow, and volatility features.
3. Turn each book snapshot into a normalized liquidity point cloud.
4. Extract H0/H1 persistent-homology and Betti-count features.
5. Select the fractional-differencing order using only the training prefix; apply the filter causally on CPU or CUDA.
6. Fit a custom binary Focal Loss XGBoost model and a standard log-loss control.
7. Score chronological held-out data after a purge gap and report Pearson IC, Rank IC, ROC-AUC, PR-AUC, and log loss.

## Current evidence

The repository separates synthetic engineering checks from market-data results. The target IC of **0.064** is a reproduction target, not a measured result for the runs below.

### Deterministic synthetic experiment

The supported CI experiment uses 6,000 generated events, a chronological holdout, a purge gap, five feature/objective variants, six random seeds, five walk-forward folds, and a label-permutation placebo. Numerical checks compare the Focal Loss gradient and Hessian against finite differences and test malformed books, price-translation invariance, and causal fractional differentiation.

The stored example produced Pearson IC **0.073292** for Focal Loss and **0.095368** for the log-loss control. These figures describe synthetic data only; they are not evidence of market alpha.

![Robustness dashboard](assets/figures/robustness_dashboard.png)

![Synthetic data diagnostics](assets/figures/data_diagnostics_dashboard.png)

### FI-2010 benchmark — corrected stock-sequence evaluation

The public FI-2010 Decimal-Precision benchmark contains 254,750 training events and three held-out files with 139,587 events combined. The files do not expose security IDs. The diagnostic found four abrupt mid-price jumps above 5% in each file (about 27%–182%). To avoid creating labels and features across apparent stock switches, the current experiment splits at a fixed 5% relative mid-price jump. This produces five inferred sequences per file and 16 cuts across the four files. The rule uses price inputs only, not labels or future returns; it remains an assumption because ticker IDs are absent.

The corrected experiment selects fractional-differencing order **d = 0.2** from training sequences only, extracts H0/H1 features with Giotto-TDA, and applies a fixed 10-event horizon, a 0.0005 label threshold, and a 50-event purge at each sequence edge.

| Fixed custom Focal Loss model | Held-out result |
|---|---:|
| Pooled Pearson IC (probability vs continuous forward return) | **0.15194** |
| Pooled Rank IC | 0.15420 |
| Mean daily Pearson IC | 0.15437 |
| Daily Pearson IC standard deviation | 0.01212 |
| Mean daily Rank IC | 0.15569 |
| ROC-AUC, days 8 / 9 / 10 | 0.7170 / 0.6946 / 0.7076 |
| Topology clouds evaluated | 403 |
| FFD backend in this run | CPU fallback; hosted runner had no CUDA |

The fixed Focal Loss Pearson IC exceeds the **0.064** target on all three held-out days: 0.15986, 0.14048, and 0.16277. This is a measured result; the target value is not substituted into any metric.

![FI-2010 validation dashboard](assets/figures/fi2010_validation_dashboard.png)

### Additional controls and limits

A training-only threshold/objective sweep selected standard log-loss with a zero return threshold. Its held-out Pearson IC was 0.19590 and Rank IC was 0.20229. This is a separate control, not the custom Focal result. A separate continuous-return XGBoost regressor, selected using five chronological folds inside the training sequences, reached Pearson IC 0.21495 and Rank IC 0.22830. This is also a different objective and is reported separately.

The corrected CPU rerun completed in [Actions run #167](https://github.com/SpryzenHell/topology_lob/actions/runs/38865530026). The run passed its unit tests, real-data TDA and feature checks, training-only sweeps, and final report validation. Its fresh result files are available in [the run's experiment artifact](https://github.com/SpryzenHell/topology_lob/actions/runs/38865530026/artifacts/12031221876).

![Walk-forward-selected return regression](assets/figures/fi2010_return_regression_selection.png)

The chart shows training-only walk-forward selection on the left and the separate held-out return-regression score by test day on the right. These are CPU-run values; the A100 workflow is being checked separately.

Feature ablation measured pooled Focal Pearson IC of 0.16549 using microstructure alone, 0.16718 using microstructure plus topology, 0.14941 using microstructure plus FFD, and 0.15045 with the full feature stack. TDA runs and yields Betti features, but this experiment does not show that adding every feature improves every metric.

FI-2010 is a licensed, preprocessed benchmark, not the original raw exchange feed. The fixed Focal result supports the numerical target on this benchmark under the stated 5% sequence-split rule; it does not by itself prove the historical raw-feed result. The corrected FI-2010 run used CPU FFD. GPU FFD passed a prior A100 kernel-parity check, but the corrected stock-split FI-2010 result must still be rerun on A100 before claiming this exact result was measured on GPU.

The dataset is under **CC BY 4.0**. Credit Ntakaris et al. (2018), *Journal of Forecasting* 37(8), 852–866, [DOI 10.1002/for.2543](https://doi.org/10.1002/for.2543), and link the [official FI-2010 record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) when sharing results.

See the [resume-target validation record](results/RESUME_TARGET_VALIDATION.md), [FI-2010 experiment details](results/FI2010_VALIDATION.md), and [boundary diagnostic](results/FI2010_BOUNDARY_DIAGNOSTICS.md). The latest completed CPU experiment JSON files and selection image are available in [Actions artifact 12031221876](https://github.com/SpryzenHell/topology_lob/actions/runs/38865530026/artifacts/12031221876).

## Quick start

Python 3.11 is recommended.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tda,dev]"
python -m pytest -q
~~~

Run the deterministic end-to-end demo with the full Giotto-TDA backend:

~~~bash
python run.py demo --config configs/demo.json --out results/demo --require-gtda
~~~

The output includes metrics.json, test_scores.csv, Betti curves, a fractional-stationarity plot, and index.html. Add --require-cuda to require CUDA fractional differentiation; the command exits if it falls back to CPU.

Run the extended experiments and C++ benchmark:

~~~bash
python scripts/generate_evidence.py --events 6000 --out results/evidence --require-gtda
python run.py ablation --config configs/demo.json --out results/ablation
python run.py walk-forward --config configs/demo.json --out results/walk_forward.json
python run.py ffd-benchmark --events 200000 --width 256 --d 0.45
cmake -S . -B build
cmake --build build --config Release
./build/fractional_diff_cpu
~~~

## Real L2 input

The CSV loader expects one row per snapshot with a timestamp and columns for every level:

~~~text
timestamp,bid_price_1,bid_size_1,ask_price_1,ask_size_1,...,bid_price_N,bid_size_N,ask_price_N,ask_size_N
~~~

The loader sorts by timestamp and rejects duplicate or invalid timestamps, missing or non-finite values, negative sizes, invalid prices, and crossed/locked or incorrectly ordered books.

~~~bash
python run.py demo --config configs/demo.json --data /absolute/path/to/l2.csv --out results/real_l2 --require-gtda
~~~

Use raw, timestamped L2 snapshots for any claim about a live-feed result. FI-2010 uses an event index in place of exchange timestamps because its benchmark files do not provide usable wall-clock timestamps.

## FI-2010 benchmark reproduction

The fetch script downloads the public Decimal-Precision benchmark files to a local directory; dataset files are not committed. FI-2010 is CC BY 4.0; credit Ntakaris et al. (2018) and link the [official data record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) when sharing results.

~~~bash
python scripts/fetch_fi2010.py --out data/external/fi2010.zip --extract
python scripts/run_fi2010_benchmark.py \
  --train data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --test \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt \
  --tda-stride 1000 --require-gtda --out results/fi2010_benchmark.json
~~~

Review the terms on the [official FI-2010 data record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) before using or redistributing the dataset. The repository stores scripts and result summaries, not raw dataset files.

### IC sensitivity experiment

The real-data workflow also runs `scripts/run_fi2010_ic_sweep.py`. It compares five return thresholds and both Focal Loss and log-loss using a chronological validation period inside the training data. Only the selected configuration is then evaluated on held-out segments. Pearson IC and Rank IC are recorded independently; the target value is never inserted into the measurements.

## Figures and reports

These images show experiment results and data analysis, not terminal captures. The report snapshot is rendered as a high-contrast raster image so its text stays readable on GitHub.

- [Experiment dashboard](assets/figures/experimental_validation_dashboard.png)
- [Robustness checks](assets/figures/robustness_dashboard.png)
- [Data diagnostics](assets/figures/data_diagnostics_dashboard.png)
- [FI-2010 results](assets/figures/fi2010_validation_dashboard.png)
- [Run-report snapshot](assets/screenshots/report_preview.png)
- [Figure notes](assets/README.md)

## Author

**Pirate-Emperor**

## License

The project code is released under the MIT License; see [LICENSE](LICENSE). Dataset terms are separate from the code license.
