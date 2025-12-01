# Comparison: expanded dataset (synonym_youtoxic_english_1000.csv) vs original results

This document compares the key results from the expanded-data run (`eda/exp_synonym_1000/summary_synonym_1000.json`) against the previous best/representative results saved in the repository.

Summary of files used for comparison
- New (expanded-data) summary: `eda/exp_synonym_1000/summary_synonym_1000.json`
- Old conservative (OOF-guided best): `eda/logreg_spacy_oof_gap_search_record.json`
- Old XGBoost tuned overfit summary: `eda/xgboost_tuned_overfit_gap.json`

## Side-by-side metrics

Conservative model (TruncatedSVD + LogisticRegression)
- Old (original dataset):
  - OOF F1: 0.336884
  - Test F1: 0.331718
  - Overfit gap: 0.015334 (≈ 1.53%)
- New (expanded dataset):
  - OOF F1: 0.484243
  - Test F1: 0.475684
  - Overfit gap: 0.017676 (≈ 1.77%)

XGBoost model (per-label training / averaged folds)
- Old (threshold-tuned XGBoost, note: test-set thresholds caused leakage):
  - Train macro F1 (thresholded): 0.751868
  - Test macro F1 (thresholded): 0.540496
  - Overfit gap (thresholded): 0.281129 (≈ 28.11%)
- New (expanded dataset, fold-averaged per-label XGBoost):
  - OOF F1: 0.780716
  - Test F1: 0.783138
  - Overfit gap: -0.003102 (test slightly higher than OOF)

## Interpretation
- Both pipelines (conservative logistic+SVD and XGBoost) improved when we trained on the expanded dataset `synonym_youtoxic_english_1000.csv`.
- The conservative model's absolute F1 increased from ≈0.33 → ≈0.48 while maintaining a very small overfit gap (~1.5–1.8%). That is an excellent gain in generalization while keeping the desired stability.
- The XGBoost pipeline shows a dramatic improvement: OOF and test F1 ≈0.78, with no sign of overfitting (test slightly higher than OOF). This suggests the added data substantially reduced variance and allowed the stronger model to learn useful patterns without memorizing.
- The old XGBoost numbers that reported test F1 ≈0.53 and a large gap were partly influenced by threshold tuning on the test set (leakage) and smaller training data; the new fold-averaged per-label XGBoost results are more reliable because they use KFold OOF probabilities and averaged test predictions.

## Recommendation
1. Treat the expanded-data XGBoost run as the best-performing candidate so far (highest test F1 and no overfit gap). Validate it further by:
   - Recomputing per-label calibration and OOF-based threshold selection (to ensure thresholds generalize),
   - Running a final evaluation on a held-out set if available, or using cross-validated metrics with different random seeds.

2. Keep the conservative logistic+SVD model as a fallback / production-safe model because it still provides a small gap and improved F1 (≈0.48). Useful if you need a simpler, faster model or want redundancy.

3. Document and version the models and their exact training artifacts under `eda/exp_synonym_1000/` (already saved). Add a short README there describing how each artifact was produced and the random seed used.

## Quick numeric table

| Model | Data | OOF F1 | Test F1 | Overfit gap |
|---|---:|---:|---:|---:|
| Conservative (LogReg+SVD) — old | original | 0.336884 | 0.331718 | 0.015334 (~1.53%) |
| Conservative (LogReg+SVD) — new | synonym_1000 | 0.484243 | 0.475684 | 0.017676 (~1.77%) |
| XGBoost (old, threshold-tuned) | original | 0.751868 (train) | 0.540496 (test) | 0.281129 (~28.11%) |
| XGBoost (new, fold-averaged) | synonym_1000 | 0.780716 | 0.783138 | -0.003102 (~-0.31%) |

> Note: The old XGBoost entry mixes a thresholded train value and a test value derived after test-set threshold tuning (this leaks information). Use the new XGBoost fold-averaged OOF/test values for fair comparison.

## Next actions I can do for you
- Re-run an OOF-based threshold search for the new XGBoost to produce final per-label thresholds without test leakage.
- Produce a CSV that lists all artifacts (old and new) and their key metrics for archival.
- Create a short notebook cell snippet that loads both the conservative and XGBoost models from `eda/exp_synonym_1000/` and prints per-label metrics.

File generated: `eda/doc/comparison_synonym_vs_original.md`
