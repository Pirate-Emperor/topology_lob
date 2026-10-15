# FI-2010 benchmark validation

This report records the real-data FI-2010 experiment run in GitHub Actions [run 117](https://github.com/SpryzenHell/topology_lob/actions/runs/38710254690). The run downloaded the public Decimal-Precision files, ran the Giotto-TDA pipeline, tested feature ablations, completed the leakage-aware IC sweep, and uploaded the result artifact.

## Data and environment

| Item | Value |
|---|---|
| Dataset | FI-2010 Decimal-Precision |
| License | CC BY 4.0 |
| Attribution | Ntakaris et al. (2018), Journal of Forecasting 37(8), 852–866 |
| Data record | https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 |
| Training events | 254,750 |
| Held-out segments | 55,478 / 52,172 / 31,937 |
| Total events | 394,337 |
| Topology clouds | 395 |
| TDA backend | Giotto-TDA |
| FFD order, fixed reference | d = 0.3 |
| FFD backend | CPU fallback; this hosted runner had no CUDA |
| Horizon / purge | 10 / 50 events |

FI-2010 is a public, preprocessed benchmark, not a raw live exchange feed. The files use event order rather than usable exchange timestamps. It is a valid reproducibility test, but it cannot by itself prove results on a raw production feed.

## Fixed Focal Loss benchmark

The fixed reference uses the full feature stack and a positive-return threshold of 0.0005.

| Held-out segment | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| 8 | 0.007156 | 0.173629 | 0.723816 | 0.261587 | 0.636420 |
| 9 | -0.002349 | 0.144605 | 0.691614 | 0.224680 | 0.645734 |
| 10 | 0.000786 | 0.161694 | 0.709854 | 0.253142 | 0.641049 |
| Pooled | 0.002023 | 0.159088 | — | — | — |

The mean daily Pearson IC is 0.001864 (standard deviation 0.004843). Mean daily Rank IC is 0.159976 (standard deviation 0.014588). Pearson IC and Rank IC are reported separately because they measure different things.

## Feature ablation

All variants use the same held-out segments and fixed training setup.

| Features / objective | Pearson IC | Rank IC | ROC-AUC |
|---|---:|---:|---:|
| Microstructure only, Focal | 0.003538 | 0.170967 | 0.702435 |
| Microstructure + FFD, Focal | 0.002793 | 0.158005 | 0.705547 |
| Microstructure + topology, Focal | 0.002955 | 0.174857 | 0.703799 |
| Full stack, Focal | 0.001826 | 0.160033 | 0.709805 |
| Full stack, log-loss | 0.001184 | 0.162941 | 0.707911 |

The full Focal stack gave the highest ROC-AUC in this comparison, but it did not have the highest Pearson IC or Rank IC. Fractional differentiation and topology should not be claimed to improve every metric based on this run.

## Threshold/objective sweep with internal validation

The new sweep used a chronological split inside the training file. It chose the fractional-differencing order from the first 60% of the training stream, fit candidate models on 190,757 rows, and selected among five label thresholds and two objectives using 63,628 validation rows. The held-out segments were not used to select the threshold or objective.

The selected setting was threshold = 0.0001 with standard log-loss. Its validation Pearson IC was 0.134872 and Rank IC was 0.211841. That high validation result did not carry over to the held-out data.

| Held-out pooled signal | Pearson IC | Rank IC | Rows |
|---|---:|---:|---:|
| Fixed Focal, threshold 0.0005 | 0.001933 | 0.156736 | 139,557 |
| Validation-selected log-loss, threshold 0.0001 | 0.001105 | 0.203677 | 139,557 |

| Segment | Fixed Focal Pearson | Fixed Focal Rank | Selected Pearson | Selected Rank |
|---|---:|---:|---:|---:|
| 8 | 0.006876 | 0.171880 | 0.007464 | 0.221199 |
| 9 | -0.002158 | 0.143406 | -0.002475 | 0.186737 |
| 10 | 0.000501 | 0.157393 | -0.002600 | 0.210339 |

**The validation-to-test drop is large.** The selected model's Pearson IC fell from 0.134872 in validation to 0.001105 on held-out data. This is a clear generalization warning; the validation result must not be described as out-of-sample performance.

## Does this reach the 0.064 target?

The original target does not state whether IC means Pearson correlation or Spearman Rank IC, so both results must be kept visible.

- **Pearson IC:** not reached. The best held-out Pearson IC in the validation-selected sweep was 0.001105, and the fixed Focal reference measured 0.001933.
- **Rank IC:** exceeded. The fixed Focal reference measured 0.156736 and the validation-selected model measured 0.203677 on the held-out data.

Therefore, the 0.064 target is supported on FI-2010 only if the intended metric is Rank IC. This experiment does not support a 0.064 Pearson IC claim. Do not present the ambiguous target as achieved without stating the IC definition and using a separate raw L2 dataset for confirmation.

## GPU status

This hosted run used CPU fallback for fractional differentiation. It verifies the numerical and model pipeline on public data, but **does not verify GPU acceleration**. The separate A100 workflow requires CUDA to be exercised and compares CUDA results with the CPU reference; its real-data result must be read separately.

## Artifact and reproduction

- Workflow: https://github.com/SpryzenHell/topology_lob/actions/runs/38710254690
- Result artifact: topology-lob-fi2010-results (ID 11960366159)
- Archive source: public FI-2010 Decimal-Precision mirror; raw data are not committed.

To rerun the core benchmark, use the commands in DATA_GUIDE.md. To repeat the validation-selected sweep, use scripts/run_fi2010_ic_sweep.py and keep threshold selection inside the training data.
