# Real-data run guide

## FI-2010 benchmark

FI-2010 is a public high-frequency limit-order-book benchmark for five Finnish equities over ten trading days. It provides ten levels of bid/ask prices and sizes and is widely used for short-horizon price-movement research.

- Dataset record: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Original paper: A. Ntakaris, M. Magris, J. Kanniainen, M. Gabbouj and A. Iosifidis, “Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods,” Journal of Forecasting, 37(8), 852–866 (2018), DOI: https://doi.org/10.1002/for.2543.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as listed for the dataset/paper. Cite the authors and link the original dataset when sharing results or derived data.
- Data scope: the published benchmark is preprocessed and normalized. It is not a raw live exchange feed, and the text files do not provide usable exchange wall-clock timestamps.

The TDA experiments use the Decimal-Precision variant because price/depth geometry is more meaningful there than in z-score-normalized files. Do not describe FI-2010 metrics as results from a raw live feed.

## Download and convert

From the repository root:

~~~bash
python scripts/fetch_fi2010.py --out data/external/fi2010.zip --extract
python scripts/convert_fi2010.py \
  --input data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --output data/FI2010_DecPre_train.csv
~~~

The fetch helper prints the archive SHA-256. Keep that checksum, data record, selected variant, and split details with experiment output. Do not commit the raw archive or extracted data.

## Reproduce the benchmark

~~~bash
python scripts/run_fi2010_benchmark.py \
  --train data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --test \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt \
  --tda-stride 1000 --require-gtda \
  --out results/fi2010_benchmark.json
~~~

The three held-out segments are evaluated separately as well as pooled. Forward-return labels stop at each file boundary, and the training end is purged before fitting.

## Tune without using the final test

The IC sensitivity sweep selects a label threshold and objective using a chronological validation part of the training file, then refits on the training file and evaluates the held-out segments once:

~~~bash
python scripts/run_fi2010_ic_sweep.py \
  --train data/external/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --test \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt \
    data/external/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt \
  --require-gtda --tda-stride 1000 \
  --out results/fi2010_ic_sweep.json
~~~

The sweep compares Pearson IC and Rank IC and reports both against the 0.064 target. Neither metric is replaced with the target value. Threshold selection is based on internal validation, not the final test segments.

## GPU fractional differentiation

Add --require-cuda to the benchmark or sweep only on a machine with a working CUDA runtime. A GPU claim is valid only if the output reports gpu_used = true and fractional_diff_backend = numba-cuda. CPU fallback is allowed for ordinary CI but does not count as GPU verification.

## Scope limits

FI-2010 contains ten trading days of preprocessed data. It is useful for reproducible benchmark tests but cannot establish long-term stability or live trading performance. Confirm findings on a separate raw L2 feed before describing a real-feed result.
