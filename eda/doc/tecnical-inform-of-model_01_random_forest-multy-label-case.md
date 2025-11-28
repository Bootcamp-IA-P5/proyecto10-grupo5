# Technical documentation: model_01_random_forest-multy-label-case.ipynb

This document walks through the notebook `model_01_random_forest-multy-label-case.ipynb` cell-by-cell (in execution order) and explains, in technical detail, what each cell does, which variables it reads/writes, expected inputs/outputs, assumptions, and potential pitfalls. Use this as a script to explain each notebook cell in presentations or code reviews.

Notes about conventions used below:
- "Cell N" refers to the N-th cell in the notebook starting at 1.
- For code cells we list: purpose, main libraries/objects used, key variables produced/consumed, and edge-case notes.
- For markdown cells we summarise the intent and what to say when presenting.

---

Cell 1 — Markdown (Title)
- Purpose: Notebook title and high-level objective. Explains that the notebook trains and optimizes a multi-label Random Forest classifier for toxic comment detection.
- Presentation tip: state the objective and dataset used.

Cell 2 — Markdown (Import Libraries header)
- Purpose: Section header for imports.

Cell 3 — Code (Imports & basic setup)
- Purpose: Import core scientific and ML libraries and set up a few utilities.
- Key libraries: pandas, numpy, matplotlib, seaborn, joblib, datetime, sklearn.ensemble.RandomForestClassifier, sklearn.model_selection utilities, sklearn.metrics functions, warnings.
- Effects: Defines functions/classes in memory; prints start timestamp.
- Variables created: None persistent beyond imports, aside from the imported modules themselves.
- Pitfalls: Importing heavy libs can take time; warnings are suppressed which can hide benign deprecation notices.

Cell 4 — Markdown (Load Preprocessed Data header)
- Purpose: Section header indicating data loading.

Cell 5 — Code (Load train/test + TF-IDF vectorizers)
- Purpose: Read CSVs `train_data_2.csv` and `test_data_2.csv` into DataFrames and load pre-fitted TF-IDF vectorizers from disk using `joblib.load`.
- Inputs: files expected in working directory: `train_data_2.csv`, `test_data_2.csv`, `tfidf_vectorizer_unigram_2.pkl`, `tfidf_vectorizer_bigram_2.pkl`.
- Outputs/side effects: `train_df`, `test_df`, `tfidf_unigram`, `tfidf_bigram` in memory.
- Pitfalls: If vectorizer files are missing or paths are wrong, joblib.load will raise FileNotFoundError.

Cell 6 — Markdown (Prepare Features)
- Purpose: Section header for feature preparation.

Cell 7 — Code (Feature matrix X and target y preparation)
- Purpose: Transform raw text into TF-IDF bigram features and create target matrices for 10 labels.
- Key operations: X_train = tfidf_bigram.transform(train_df['text']); X_test = tfidf_bigram.transform(test_df['text']); define `target_cols` list of 10 column names; y_train/y_test are DataFrames sliced from train/test.
- Variables produced: `X_train`, `X_test` (sparse matrices), `y_train`, `y_test` (DataFrames), `target_cols` (list).
- Notes: prints shapes and class counts; `X_train.shape[1]` gives number of features (bigram vocabulary size).

Cell 8 — Markdown (Train Random Forest - Baseline)
- Purpose: Introduce baseline training cell.

Cell 9 — Code (Train baseline RandomForestClassifier)
- Purpose: Instantiate and fit a RandomForestClassifier (n_estimators=100) to multi-label targets using scikit-learn's native multi-output behavior (when passing 2D y) — sklearn will create one estimator per label internally.
- Key ops: rf_baseline.fit(X_train, y_train)
- Variables produced: `rf_baseline` (fitted RandomForestClassifier), training time printed.
- Pitfalls: Training a forest with multi-output for high-dimensional sparse X can be slow and memory heavy; be sure n_jobs is set and sufficient RAM exists.

Cell 10 — Markdown (Evaluate Baseline Model)
- Purpose: Introduce evaluation block.

Cell 11 — Code (Predict and get probabilities for baseline)
- Purpose: Compute hard predictions and stack probabilities from the multi-output estimators.
- Key ops:
  - y_train_pred = rf_baseline.predict(X_train)
  - y_test_pred = rf_baseline.predict(X_test)
  - y_train_proba/y_test_proba = stack positive-class probabilities for each label
- Variables produced: `y_train_pred`, `y_test_pred`, `y_train_proba`, `y_test_proba`.
- Notes: rf_baseline.predict_proba returns a list (one array per label); the code extracts the positive-class column.

Cell 12 — Markdown (Calculate Multi-Label Metrics (ROBUST FIX))
- Purpose: Header for robust metric calculations and overfitting diagnostics.

Cell 13 — Code (Metrics, CV baseline, ROC-AUC robust calculation)
- Purpose: Compute macro/micro F1, precision, recall, ROC-AUC while dealing with labels that have no variance. Compute a baseline CV score using the first estimator (IsToxic) and calculate an overfit gap.
- Key ops:
  - test_f1_macro, test_precision_macro, test_recall_macro, test_f1_micro
  - train_f1_macro
  - robust AUC computation: filter labels with variance >1 and compute roc_auc_score on the corresponding columns only
  - cross_val_score on rf_baseline.estimators_[0] for IsToxic (StratifiedKFold)
  - overfit_gap computed: (train_f1_macro - test_f1_macro) / train_f1_macro
- Variables produced: `test_auc_macro`, `test_auc_weighted`, `cv_scores`, `overfit_gap`, `overfit_diff_absolute`.
- Pitfalls: Using a single estimator (estimators_[0]) for CV is a simplification — it only measures IsToxic; multi-label CV requires per-label scoring or specialized wrappers.

Cell 14 — Markdown (Overfitting Check (Multi-Label))
- Purpose: Introduce quick overfitting check cell.

Cell 15 — Code (Overfitting check and qualitative bands)
- Purpose: Compute train/test Macro F1 difference and categorize gap levels (<5% good, 5-15% moderate, >=15% high).
- Inputs: relies on `y_train`, `y_train_pred`, `y_test`, `y_test_pred` computed earlier.

Cell 16 — Markdown (Multi-Label Classification Report)
- Purpose: Header for classification report.

Cell 17 — Code (Print classification_report)
- Purpose: Print sklearn.metrics.classification_report for multi-label predictions (test set) using `y_test` and `y_test_pred`.

Cell 18 — Markdown (Multi-Label Confusion Matrices)
- Purpose: Header

Cell 19 — Code (Generate and print confusion matrices per label)
- Purpose: Compute `multilabel_confusion_matrix` and print counts (TN, FP, FN, TP) per label; useful to inspect support and error types per label.
- Outputs: console text for each label showing the 2x2 confusion matrix.

Cell 20 — Code (Heatmap visualization of confusion matrices)
- Purpose: Use seaborn and matplotlib to present the 10 confusion matrices as heatmaps arranged in a 5x2 grid. Uses `matrix_array` computed previously.
- Variables used: `matrix_array`, `target_cols`.
- Notes: Cell hides unused subplots when fewer than 10 labels.

Cell 21 — Markdown (Feature Importance Analysis)
- Purpose: Header

Cell 22 — Code (Print top-5 feature importances per label)
- Purpose: For each estimator in rf_baseline.estimators_, obtain `feature_importances_`, pair with `feature_names = tfidf_bigram.get_feature_names_out()`, create DataFrame, sort and print top 5 features per label.
- Key variables: `feature_names`, `rf_baseline.estimators_`.
- Pitfalls: Feature importances of tree ensembles on high-dimensional sparse TF-IDF vectors are hard to interpret; correlated tokens/bigrams and IDF scaling can skew importance.

Cell 23 — Code (Plot top-30 feature importances per label)
- Purpose: Create a 5x2 grid where each subplot shows top 30 features for a label as a bar chart. Uses seaborn.barplot.
- Side effects: Display plots inline; may take time for 10 plots with many labels.

Cell 24 — Markdown (ROC Curve Analysis per label)

Cell 25 — Code (ROC curves per label)
- Purpose: For each label, calculate fpr, tpr, auc and plot ROC curve per label in a grid. Uses `y_test_proba` as probabilities.

Cell 26 — Markdown (Precision-Recall Curve Analysis)

Cell 27 — Code (Precision-Recall curves per label)
- Purpose: Plot precision-recall curves, plot baseline (positive class rate) and average precision score when available. Handles labels without positive examples via try/except.

Cell 28 — Markdown (ROC Curve Visualization — duplicate/variant)

Cell 29 — Code (Alternate ROC plotting)
- Purpose: Duplicate/variant of earlier ROC plotting cell with small stylistic changes (grid, legend). Safe to skip if previously produced identical plots.

Cell 30 — Markdown (Precision-Recall Curve Visualization — duplicate/variant)

Cell 31 — Code (Alternate Precision-Recall plotting)

Cell 32 — Markdown (Cross-Validation)

Cell 33 — Code (Cross-validation macro-F1 on multi-label)
- Purpose: Perform 5-fold CV over all 10 labels using `cross_val_score` with `make_scorer(f1_score, average='macro')` and KFold (since StratifiedKFold is incompatible with multi-label); compute CV mean, overfit gap between CV and test.
- Variables produced: `cv_scores_macro`, `cv_f1_mean`, `overfit_gap_cv`.
- Pitfalls: cross_val_score with a 2D y uses estimator's multi-output behavior; some scorers/wrappers may not support multi-output directly — using macro f1 scorer with multi-output is an accepted simplification.

Cell 34 — Markdown (Optimized Random Forest header)

Cell 35 — Code (Train and evaluate optimized Random Forest — hand-tuned params)
- Purpose: Train a new RandomForestClassifier with more aggressive parameters (n_estimators=200, max_depth=30, min_samples_leaf=2, class_weight='balanced') to attempt to improve recall for minority labels.
- Variables produced: `rf_optimized`, `y_train_pred_opt`, `y_test_pred_opt`, `y_train_proba_opt`, `y_test_proba_opt`, metrics like `test_f1_macro_opt`.
- Notes: This is a direct hand-tuned run (not Optuna). Class weighting is used to penalize errors on minority classes.

Cell 36 — Markdown (Final evaluation optimized RF)

Cell 37 — Code (Comprehensive evaluation & overfitting check for optimized RF)
- Purpose: Compute detailed test/train metrics, ROC-AUC robustly, overfit gap percentage, and print classification report for `rf_optimized`.
- Inputs: `rf_optimized`, `X_train`, `X_test`, `y_train`, `y_test`.

Cell 38 — Markdown (Compare baseline vs optimized)

Cell 39 — Code (Create comparison table and insights)
- Purpose: Build a pandas DataFrame comparing baseline metrics and optimized model metrics (macro F1, recall, precision, ROC AUC, F1 gap). Displays textual insights and automatic improvement flags.

Cell 40 — Markdown (Save Model and Results)

Cell 41 — Code (Save optimized model + JSON results)
- Purpose: Persist `rf_optimized` to `random_forest_optimized_multi_label_model.pkl` using joblib, prepare and save a results JSON `random_forest_optimized_multi_label_results.json` with parameters and evaluation metrics.
- Pitfalls: Some numeric placeholders are hard-coded in the cell; ensure to update them if re-running.

Cell 42 — Markdown (Optuna hyperparameter optimization header)

Cell 43 — Code (Optuna objective definition for RandomForest)
- Purpose: Define `objective(trial)` for Optuna that searches for RF hyperparameters (n_estimators, max_depth, max_features, min_samples_leaf) and evaluates with cross_val_score on multi-label target using KFold and macro F1 scorer. Returns mean CV score for Optuna to maximize.
- Notes: Uses `trial.suggest_int()` and `trial.suggest_categorical()` to build search space. Forces `class_weight='balanced'`.

Cell 44 — Markdown (Run the Optuna study)

Cell 45 — Code (Execute Optuna study)
- Purpose: Create `study = optuna.create_study(direction='maximize')` and run `study.optimize(objective, n_trials=N_TRIALS)` with 100 trials; then print best trial value and params.
- Side effects: Depending on compute, this can take long; uses parallel CV via n_jobs=-1 in cross_val_score.

Cell 46 — Code (print best results) — duplicate display

Cell 47 — Markdown (Final Model Training and Evaluation)

Cell 48 — Code (Train final model using study.best_params or fallback + apply safety caps)
- Purpose: Build `bp` from study.best_params if available; apply conservative safety caps (cap n_estimators to 300, cap max_depth to 12, set min_samples_leaf >=3) to avoid overfitting; train `rf_optuna_final` with `class_weight='balanced'`.
- Outputs: `rf_optuna_final` trained model and timing info.

Cell 49 — Markdown (Final Evaluation and Comparison)

Cell 50 — Code (Evaluate rf_optuna_final)
- Purpose: Compute final metrics for `rf_optuna_final` (train/test macro F1, ROC AUC robustly) and print overfit gap percent; print classification report.

Cell 51 — Markdown (Save the Optuna final model)

Cell 52 — Code (Save optuna final model and results JSON)
- Purpose: Save `rf_optuna_final` to disk as `random_forest_optuna_final_model.pkl` and write results file `random_forest_optuna_final_results.json` with metrics and hyperparameters. Several numeric values are present as hard-coded placeholders — update if rerunning.

Cell 53 — Markdown (Final conclusion summary)

Cell 54 — Markdown (Notes on reducing labels to 6 to improve overfitting)

Cell 55 — Markdown (Predictions about rare labels and justification for using 6 labels)

Cell 56 — Code (Define training with 6 labels: load and show distributions)
- Purpose: Subset the dataset to 6 labels (dropping extremely rare labels) to stabilize macro metrics. Produces `y_train_6`, `y_test_6` and prints per-label counts.

Cell 57 — Code (Train optimized RF on 6 labels with safety caps)
- Purpose: Similar to earlier RF training but applied to `y_train_6` with conservative caps pulled from studies if available; variable `rf_6_labels` is produced.

Cell 58 — Code (Evaluate 6-label model)
- Purpose: Assess `rf_6_labels` on X_test using macro metrics, compute overfitting gap percent and classification report.

Cell 59 — Code (Comparison: 10 labels vs 6 labels)
- Purpose: Build a comparison table summarizing metrics for 10-label and 6-label models to demonstrate trade-offs and motivate using 6-label setup.

Cell 60 — Code (Save 6-label model + results)
- Purpose: Persist `random_forest_6labels_final.pkl` and `random_forest_6labels_results.json` with metrics.

Cell 61 — Markdown (Suggestions to improve overfitting and plan for hyper-regularized optuna)

Cell 62 — Markdown (Optuna hyper-regularization for 6 labels)

Cell 63 — Code (Define objective_6_labels with stricter regularization constraints)
- Purpose: Construct Optuna objective focused on generalization for 6-label problem: restrict search space to smaller max_depth (10-20), larger min_samples_leaf, smaller n_estimators, smaller max_features.

Cell 64 — Markdown (Run hyper-regularized Optuna)

Cell 65 — Code (Execute hyper-regularized Optuna study — study_6_labels)
- Purpose: Run 100 trials of the regularized Optuna study and print best params/value.

Cell 66 — Markdown (Train Final Regularized Model)

Cell 67 — Code (Train rf_regularized_final using best_params_regularized)
- Purpose: Train `rf_regularized_final` using parameters found by the regularized Optuna and class_weight balanced.

Cell 68 — Markdown (Final Evaluation and Overfitting Gap Calculation)

Cell 69 — Code (Evaluate `rf_regularized_final`)
- Purpose: Compute metrics, overfit gap percent; print classification report; decide if overfitting is acceptable.

Cell 70 — Markdown (Goal not achieved marker)

Cell 71 — Markdown (Optuna Optimization: Multi-Label XGBoost header)

Cell 72 — Code (Install XGBoost)
- Purpose: Run `!pip install xgboost` in the notebook to install the package. Side effect: package installed in kernel.

Cell 73 — Code (Define objective_xgboost for Optuna with XGBoost hyperparams)
- Purpose: Define XGBoost parameter search space (n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree, reg_alpha, reg_lambda) and wrap base XGBClassifier in MultiOutputClassifier; evaluate with KFold CV macro-F1.

Cell 74 — Markdown (Run XGBoost Optuna study)

Cell 75 — Code (Execute XGBoost Optuna study)
- Purpose: Create and run the Optuna study for XGBoost `study_xgboost` for 100 trials.

Cell 76 — Markdown (Train Final XGBoost Model)

Cell 77 — Code (Train rf_xgboost_final using best_params_xgb)
- Purpose: Instantiate base_xgb_final with best_params_xgb and wrap with MultiOutputClassifier; fit on X_train and y_train_6. Saves `rf_xgboost_final`.

Cell 78 — Markdown (Final Evaluation and Overfitting Gap Calculation)

Cell 79 — Code (Evaluate XGBoost final)
- Purpose: Compute predictions/probabilities, compute macro metrics, overfit gap percent for XGBoost final; print classification report.

Cell 80 — Markdown (Rerun XGBoost Optuna with balanced weighting)

Cell 81 — Code (objective_xgboost_balanced and run study)
- Purpose: Compute global scale_pos_weight from data imbalance and define a new Optuna objective that forces scale_pos_weight and restricts depth; run final XGBoost balanced study.

Cell 82 — Markdown (Final XGBoost Training and Evaluation)

Cell 83 — Code (Train final balanced xgboost rf_xgboost_final and evaluate)
- Purpose: Use final best_params and scale_pos_weight to set up base_xgb_final, wrap as MultiOutputClassifier, fit and evaluate.

Cell 84 — Markdown (SpaCy feature engineering header)

Cell 85 — Code (Install SpaCy and download en_core_web_sm)
- Purpose: Notebook pip commands to install spaCy and download the small English model.

Cell 86 — Code (Load spaCy and define spacy_lemmatizer function)
- Purpose: Load spaCy model `en_core_web_sm` into `nlp` and define `spacy_lemmatizer(text)` that cleans text via regex, lowercases, runs nlp(text), and returns joined lemmas excluding stopwords and very short tokens.
- Output: `spacy_lemmatizer` function and `nlp` object in memory.

Cell 87 — Markdown (Rerun vectorization and retrain header)

Cell 88 — Code (Apply lemmatization, create tfidf_spacy, retrain xgboost on SpaCy features)
- Purpose: Apply `spacy_lemmatizer` across train/test texts, fit TfidfVectorizer (unigram only, max_features=2000), obtain `X_train_spacy`, `X_test_spacy`, instantiate `base_xgb_final_spacy` and `rf_xgboost_spacy` (MultiOutput), fit on `X_train_spacy` and `y_train_6`.
- Key variables: `train_df_lemmatized`, `tfidf_spacy`, `X_train_spacy`, `rf_xgboost_spacy`.
- Notes: Switching to lemmatized unigrams reduces feature sparsity and acts as regularization.

Cell 89 — Markdown (Final evaluation and overfitting victory header)

Cell 90 — Code (Evaluate rf_xgboost_spacy on SpaCy features)
- Purpose: Generate predictions and compute test/train macro F1 and overfit gap percent for model trained on SpaCy TF-IDF features.

Cell 91 — Code (Alternate spaCy load and robust lemmatizer redefinition)
- Purpose: Robustly load spaCy (download model if missing) and redefine `spacy_lemmatizer` with NaN handling; useful if kernel was restarted.

Cell 92 — Code (Apply lemmatization and re-vectorize with slightly different max_features)
- Purpose: Re-run the lemmatization and TF-IDF fit with `max_features=1800`, recreate `X_train_spacy`, `X_test_spacy`. This is a repeated/iterative refinement of the previous SpaCy vectorization.

Cell 93 — Code (Train final XGBoost model with SpaCy features)
- Purpose: Fit `rf_xgboost_spacy` with best_params (n_estimators=200, max_depth=4, learning_rate etc.) computed earlier; compute training time.

Cell 94 — Code (Final evaluation on SpaCy features — duplicate/variant)

Cell 95 — Markdown (Per-Label Threshold Optimization)

Cell 96 — Code (Per-label threshold brute force on test set)
- Purpose: For each of the 6 labels, stack test probabilities and brute-force thresholds from 0.05 to 0.95 to pick threshold maximizing F1 per label; produce `best_thresholds` and `final_macro_f1_tuned`.
- Data used: `rf_xgboost_spacy.predict_proba(X_test_spacy)`.
- Outputs: `best_thresholds` dictionary and prints tuned macro/micro F1.
- Notes: This method uses test set for threshold selection — it can leak information and artificially inflate test F1. Prefer OOF-based thresholding if you must tune thresholds.

Cell 97 — Code (Save final results and thresholds JSON)
- Purpose: Persist `xgboost_final_tuned_results.json` which contains `optimal_thresholds` and sample final test macro/micro F1 placeholders.

Cell 98 — Markdown (Reflection on data limitations and augmentation proposals)

Cell 99 — Markdown (Data Augmentation suggestions — high level)

Cell 100 — Code (Install augmentation libraries and initialize augmenters)
- Purpose: Install `nlpaug` and set up augmenters `AUG_SYN` (SynonymAug) and `AUG_CHAR` (KeyboardAug) for later augmentation functions.

Cell 101 — Code (Define revised augmentation function augment_toxic_samples_revised)
- Purpose: Implement a function that augments toxic samples using synonym replacement and keyboard noise, with safeguards to avoid identical outputs. Returns a concatenated DataFrame of original + augmented rows.

Cell 102 — Code (Back-translation helper skeleton)
- Purpose: Provide a helper `back_translate` for more meaningful augmentations using a translation model; the function uses a transformer model/tokenizer (T5-style) — placeholder, potentially requiring heavy dependencies (torch, transformers).

Cell 103 — Code (Complete augmentation function with back-translation option)
- Purpose: More robust augmentation combining synonym replacement and back-translation; uses `back_translate` when available. Builds `train_df_augmented`.

Cell 104 — Code (Rerun vectorization & final training with augmented data)
- Purpose: Take `train_df_augmented`, lemmatize augmented texts, vectorize with TF-IDF, and retrain XGBoost on the augmented dataset. Evaluate final model on test set and produce metrics.

Cell 105 — Code (Augmentation final attempt with fixes and resource downloads)
- Purpose: Ensure NLTK resources exist, configure `NLTK_DATA_PATH`, download 'averaged_perceptron_tagger' & 'wordnet', define augmenters globally and provide a robust augmentation function.

Cell 106 — Code (Execute augmentation, vectorization, and training pipeline)
- Purpose: Execute the overall pipeline: augment, lemmatize, vectorize, train rf_xgboost_augmented on augmented data using SpaCy+TFIDF.

Cell 107 — Code (Final evaluation of augmented model)
- Purpose: Compute train/test metrics for the augmented model and assess whether overfitting improved.

Cell 108 — Code (Prepare data for early stopping split)
- Purpose: Create a small validation set via train_test_split from augmented training set to be used as eval_set for early stopping in xgboost training.

Cell 109 — Code (Retrain XGBoost with manual early stopping per label using XGBClassifier and MultiOutputClassifier wrapper)
- Purpose: For each label, clone base XGB classifier and call `fit(..., eval_set=..., early_stopping_rounds=10)` to allow early stopping per label; store trained estimators in a list and pack into a wrapper-like object. This manual loop ensures compatibility with varying xgboost/sklearn API versions.

Cell 110 — Code (Execute final threshold tuning again using rf_xgboost_spacy)
- Purpose: Recompute per-label thresholds on probabilities from `rf_xgboost_spacy` (or other recent model) for more stable tuned thresholds.

Cell 111 — Code (Per-label XGBoost training with xgb.train and stronger regularization)
- Purpose: Train one native xgboost booster per label using xgb.train with large num_boost_round and early_stopping_rounds to obtain per-label boosters; stack predictions for evaluation.

Cell 112 — Code (Heavily regularized LogisticRegression grid search with SVD to minimize overfit gap)
- Purpose: Search combinations of TruncatedSVD dimensionality and LogisticRegression regularization `C` to find configurations that minimize overfit gap (train-vs-test difference). Uses OneVsRestClassifier and preserves `tfidf_spacy` features when available.
- Outputs: saves `logreg_spacy_best.pkl` or None if not found.

Cell 113 — Code (Compute OOF and final test predictions for a chosen config; compute real overfit gap)
- Purpose: Use cross_val_predict (KFold) to compute OOF predictions for model and compare OOF train F1 vs final test F1; persist `logreg_spacy_oof_final.pkl`.

Cell 114 — Code (OOF grid search for gap < 5%)
- Purpose: Grid-search across combinations of `n_components` and `C` to identify a model with OOF-based overfit gap < 5%. Saves found model as `logreg_spacy_oof_gap_under5.pkl` and grid results to JSON.

Cell 115 — Code (Aggressive underfitting grid search: small n_components [5,3,2,1] and very small C values)
- Purpose: More aggressive search to force underfitting (reduce capacity) so OOF train and test come closer; if found, saves `logreg_spacy_oof_gap_under5_aggressive.pkl` and small grid JSON.

Cell 116 — Code (Helper imports / kernel restore messages)

Cell 117 — Code (Rebuild feature matrices fallback from CSVs with small TF-IDF)
- Purpose: Fallback to re-create `X_train_spacy` and `X_test_spacy` from CSVs using a compact TF-IDF (max_features=1000) if previous SpaCy artifacts are missing.

Cell 118 — Code (Per-label threshold optimization using OOF probabilities)
- Purpose: Compute OOF probabilities using cross_val_predict on per-label base logistic estimators; choose per-label thresholds that maximize OOF F1. Apply thresholds to test set using fitted base classifiers to get test predictions; save `per_label_thresholds.json` and `test_pred_with_thresholds.csv`.
- Importance: This cell fixes the data-leakage issue of earlier threshold tuning by selecting thresholds using OOF probabilities (train-side CV), rather than tuning thresholds directly on the test set.

Cell 119 — Code (Kernel health check)

Cell 120 — Code (Compute tuned train F1 and overfit gap)
- Purpose: Compute train F1 after applying per-label thresholds (using training probabilities) and compare with test tuned F1 — saves `xgboost_tuned_overfit_gap.json` for record.

Cell 121 — Code (A2 improved: grid-search to find config with OOF gap <= 5% using small search)
- Purpose: Small and focused grid search over `n_components` in [1,2,5,10] and C in [0.001, 0.01, 0.1, 1.0] to find a conservative logistic+SVD configuration with OOF gap <= 5%. If found, saves artifacts `logreg_spacy_oof_gap_under5_aggressive_retrained_best.pkl` and a JSON record `logreg_spacy_oof_gap_search_record.json`.

Cell 122 — Markdown (Official low-overfit model — load & predict)

Cell 123 — Code (Load local artifact and example predict_texts function)
- Purpose: Show how to load the saved conservative model artifact and run predictions on raw text. Exposes `tfidf_spacy`, `svd` and per-label `classifiers` loaded from the artifact and provides a `predict_texts(texts, threshold=0.5)` helper that returns per-label probabilities and binary preds. Copies artifact to `logreg_spacy_low_overfit_official.pkl` as a stable name.

---

Appendix: common recommendations and pitfalls (technical)
- When re-running: re-create TF-IDF and SpaCy artifacts if kernel restarted; saved pickles/JSONs can be loaded but verify versions (sklearn/xgboost) match.
- Threshold tuning should be done on OOF probabilities to avoid leakage — Cell 118 implements that improved approach.
- When using multi-output wrappers, be careful with predict_proba return types (lists) and the required stacking order.
- Be cautious with cross_val_score on multi-label y — using a macro scorer and KFold is a pragmatic but approximate approach.

If you want, I can also produce a compact per-cell speaker script for presenting this notebook (one-paragraph speaker note per cell), or extract the exact source lines for each cell into an annotated file. Tell me which format you prefer.

---

Presenter slide script (technical short notes — one paragraph per major section)

- Setup & Imports
  - Purpose: Import required libraries, set random seed and define helper utilities. Verify environment dependencies (scikit-learn, xgboost, spaCy). Ensure TF‑IDF and SpaCy artifacts are present or re-create them when kernel restarts.

- Data loading & feature pipelines
  - Purpose: Load CSVs and pre-fitted TF‑IDF vectorizers or re-fit them after SpaCy lemmatization. Notes: check vectorizer file paths and versions when re-using pickled objects.

- Baseline RF training & evaluation
  - Purpose: Train a RandomForest baseline, compute predictions/probabilities, and calculate robust metrics (macro/micro F1, ROC AUC for labels with variance). Watch for memory consumption on high-dimensional sparse matrices.

- Cross-validation & Optuna tuning
  - Purpose: Run K‑fold CV to get reliable estimates and optionally run Optuna studies. For multi-label problems, use macro F1 over folds and guard against overly-large trees with safety caps.

- XGBoost experiments & SpaCy features
  - Purpose: Train XGBoost variants on original and SpaCy TF‑IDF features. Use native xgb.train or manual early stopping loops if sklearn wrapper early stopping is incompatible. Save per-label boosters and tuned thresholds.

- Thresholding and OOF corrections
  - Purpose: Select per-label thresholds using OOF probabilities (cross_val_predict) to avoid test leakage. Save `per_label_thresholds.json` and recompute tuned overfit summary for transparent reporting.

- Augmentation & robustness experiments
  - Purpose: Provide functions for synonym and keyboard noise augmentation, with optional back-translation. Note heavy dependencies (transformers, torch) if back-translation is enabled.

- Conservative OOF-guided model selection
  - Purpose: Grid-search TruncatedSVD + LogisticRegression for configurations with small OOF-vs-test gap (target ≤ 5%). Prefer OOF-based metrics as the decision criterion for selecting a production model.

- Final model saving & inference example
  - Purpose: Save final artifact (`logreg_spacy_low_overfit_official.pkl`) containing the TF‑IDF, SVD and per-label classifiers and provide a `predict_texts` helper. Include versioning notes for scikit-learn/xgboost compatibility.

If you want I can now:
- Generate a one‑paragraph speaker note per notebook cell (a full per‑cell script),
- Export these technical notes into a PDF or slide deck, or
- Re-run a light sync that validates the exact `Cell N` mapping against the current notebook and update both docs to include exact cell text snippets.
