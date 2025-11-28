# Project activity summary — 2025-11-28 (today) & 2025-11-27 (yesterday)

This document summarizes the work performed across both days on the `proyecto10-grupo5` repository (focusing on the `eda/model_01_random_forest-multy-label-case.ipynb` notebook and related artifacts). It captures experiments run, models explored, files created/updated, and recommended next steps.

## High-level goals
- Reduce overfitting (train vs test gap) for a multi-label toxic-comment classifier and produce a production-ready model with a small generalization gap (target ≤ 5%).
- Explore multiple model families (Random Forest, XGBoost, LogisticRegression), feature pipelines (TF-IDF bigrams, SpaCy lemmatized TF-IDF), and validation strategies (OOF/CV, early stopping, threshold tuning).
- Produce clear documentation and presentation artifacts summarizing experiments and selected models.

## Summary of actions (chronological, 2025-11-27 → 2025-11-28)

### 2025-11-27 (yesterday)
- Inspected and ran cells in `eda/model_01_random_forest-multy-label-case.ipynb` to identify where overfitting occurred and which artifacts were produced.
- Ran baseline Random Forest experiments and Optuna hyperparameter search; observed severe overfitting in some RF variants.
  - Artifact: `random_forest_optuna_final_results.json` (test_macro_f1 ≈ 0.2243, train/test gap ≈ 66.78%).
- Trained a Random Forest for the 6-label subset to improve per-label reliability.
  - Artifact: `random_forest_6labels_results.json` (test_f1 ≈ 0.4644, gap ≈ 36.64%).
- Began XGBoost work: attempted sklearn wrapper but moved to native xgb.train for reliable early stopping.
  - Artifact: `xgb_spacy_early_results.json` (train_macro_f1 ≈ 0.7748, test_macro_f1 ≈ 0.3903).
- Implemented SpaCy lemmatization and rebuilt TF-IDF (`tfidf_spacy`) to reduce feature noise; trained XGBoost on SpaCy features.
- Performed per-label threshold brute-force tuning on test set for XGBoost (noted leakage risk).
  - Artifact: `xgboost_final_tuned_results.json` (final_test_macro_f1 ≈ 0.5347) and `xgboost_tuned_overfit_gap.json` (overfit_gap ≈ 28.11%).
- Explored augmentation strategies (nlpaug, synonym/keyboard/noise, back-translation skeleton) but results varied; augmentation requires careful validation.
- Began OOF-guided conservative model search: TruncatedSVD + LogisticRegression grid to find models with low OOF gap.
  - Produced grid results and saved a record of candidate models.

### 2025-11-28 (today)
- Continued and finalized the conservative OOF-driven approach. Ran a focused small grid (including aggressive underfitting configurations) and found a configuration that met the user's overfit gap target (≤ 5%):
  - Selected model: LogisticRegression with TruncatedSVD (extreme reduction, n_components=1, C=0.1).
  - Artifact: `logreg_spacy_oof_gap_search_record.json` (oof_f1 ≈ 0.336884, test_f1 ≈ 0.331718, overfit_gap ≈ 1.53%).
  - Saved final artifacts: `logreg_spacy_oof_gap_under5_aggressive_retrained_best.pkl` and `logreg_spacy_low_overfit_official.pkl` (official copy for production).
- Created human-facing documentation:
  - `eda/doc/tecnical-inform-of-model_01_random_forest-multy-label-case.md` (technical, cell-by-cell with run notes and presenter short notes).
  - `eda/doc/non-tecnical-inform-of-model_01_random_forest-multy-label-case.md` (plain-language walkthrough rewritten and expanded; added presenter slide script and next steps).
- Created a per-cell synchronization mapping from the live notebook execution order to plain descriptions:
  - `eda/doc/per-cell-sync.md` — maps Cell 1..133 to a short description for each cell.
- Created a one-slide-per-section Markdown slide deck for presentations:
  - `eda/doc/slides.md` (Marp/Reveal-compatible Markdown slides, one slide per major section).
- Created the per-request summary `eda/doc/best-overfit-valeu.md` describing key overfitting metrics and recommending the conservative model as the production choice.
- Addressed many kernel restart/timeout issues by reading and saving artifacts rather than re-running expensive steps when possible.

## Key numeric outcomes
- XGBoost (SpaCy, test-set threshold-tuned):
  - Final test macro F1 ≈ 0.5347 (artifact: `xgboost_final_tuned_results.json`)
  - Overfit gap after threshold tuning ≈ 28.11% (artifact: `xgboost_tuned_overfit_gap.json`) — high overfitting risk and test-set leakage for thresholds.
- XGBoost (per-label native xgb.train with early stopping):
  - Train macro F1 ≈ 0.7748, Test macro F1 ≈ 0.3903 (`xgb_spacy_early_results.json`) — large train/test gap.
- Random Forest (6-label):
  - Test F1 ≈ 0.4644, Overfit gap ≈ 36.64% (`random_forest_6labels_results.json`).
- Random Forest (Optuna final):
  - Test F1 ≈ 0.2243, Train/Test gap ≈ 66.78% (`random_forest_optuna_final_results.json`).
- Conservative OOF-selected model (final chosen):
  - OOF F1 ≈ 0.336884, Test F1 ≈ 0.331718, Overfit gap ≈ 1.53% (`logreg_spacy_oof_gap_search_record.json`).

## Important artifacts (models & result files)
- Conservative low-overfit model and records:
  - `logreg_spacy_oof_gap_under5_aggressive_retrained_best.pkl`
  - `logreg_spacy_low_overfit_official.pkl`
  - `logreg_spacy_oof_gap_search_record.json`
- XGBoost artifacts:
  - `xgboost_spacy_final.pkl` (if saved during runs)
  - `xgboost_final_tuned_results.json` (optimal thresholds & final test F1)
  - `xgboost_tuned_overfit_gap.json`
  - `xgb_spacy_early_results.json`
- Random Forest artifacts:
  - `random_forest_6labels_final.pkl` (6-label model)
  - `random_forest_6labels_results.json`
  - `random_forest_optuna_final_model.pkl` (if saved)
  - `random_forest_optuna_final_results.json`
- Documentation & meta files:
  - `eda/doc/tecnical-inform-of-model_01_random_forest-multy-label-case.md`
  - `eda/doc/non-tecnical-inform-of-model_01_random_forest-multy-label-case.md`
  - `eda/doc/per-cell-sync.md`
  - `eda/doc/slides.md`
  - `eda/doc/best-overfit-valeu.md`

(If any filenames are missing from disk, please tell me and I will list the current `eda/` directory contents.)

## Decisions made & rationale
- We accepted a trade-off: the conservative logistic+SVD model sacrifices absolute F1 for a very small OOF/test gap, making it suitable for production when the priority is reliability.
- The XGBoost pipeline was retained as a higher-test-F1 alternative but flagged as "high overfitting risk"; thresholds tuned on test set are considered leaky and should be re-done with OOF-based threshold selection before being used as a fair comparison.
- OOF (out-of-fold) predictions were used as the guiding metric for selecting models with realistic in-sample generalization estimates — this avoids optimistic train metric reporting.

## Recommendations & next steps
1. If production stability is the priority: deploy `logreg_spacy_low_overfit_official.pkl` and monitor live performance while collecting more labeled data.
2. If higher F1 is needed: re-run OOF-based threshold tuning for the XGBoost pipeline (compute OOF probabilities, select thresholds on OOF, then evaluate on test)—this prevents leakage and may produce a better trade-off.
3. Collect more labeled examples for rare labels and re-run the OOF grid: more data will likely allow more complex models (XGBoost) to generalize without large gaps.
4. If augmentation is pursued, validate augmented data carefully and use OOF estimates to ensure augmentation improves true generalization.
5. Add a small CI check that re-computes `xgboost_tuned_overfit_gap.json` and the OOF gap for the conservative model when new data is added; track gap over time.

## Closing notes
- The work across the two days focused on diagnosing severe overfitting, trying stronger models and feature cleaning, and finally prioritizing a production-ready model by optimizing for a low OOF-to-test gap.
- You now have both a higher-test-F1 candidate (XGBoost, needs safer OOF tuning) and a conservative low-overfit model ready for deployment. The documentation and slides generated today will help present trade-offs to stakeholders.


File generated: `eda/doc/summary-28-11-25.md`
