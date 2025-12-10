---
marp: true
---

# Model: Random Forest multi-label — Slides

Presentation slides (Markdown) — one slide per major section. Use a slide tool that understands Markdown slides (Marp, Reveal, or GitHub Pages).

---

# Data & Setup

- Load train/test CSVs and precomputed TF‑IDF vectorizers (or re-fit after SpaCy cleaning).
- Inspect dataset size and per-label counts to understand class imbalance.

Notes: Mention dataset size (800 train / 200 test) and why imbalance matters.

---

# Baseline Model (Random Forest)

- Train a Random Forest baseline using TF‑IDF features.
- Purpose: get a simple, explainable point of comparison.

Speaker note: "This baseline is our starting benchmark." 

---

# Evaluation & Overfitting Check

- Compute per-label precision/recall/F1, macro/micro F1 and compare train vs test.
- Overfitting gap = (train - test) / train.

Speaker note: "We track overfitting as our main risk metric." 

---

# Feature Inspection

- Show top words/bigrams the model relies on.
- Helps stakeholders trust the model's signals.

---

# Cross-validation & Conservative Tuning

- Run K‑fold CV for robust estimates and run Optuna searches with safety caps.
- Use conservative caps to prevent models from overfitting.

Speaker note: "We prioritize generalization over raw in-sample performance." 

---

# Label Reduction (6-label experiment)

- Drop/merge very rare labels to stabilize metrics and focus on reliably-predicted toxicity types.

---

# XGBoost Experiments

- Train XGBoost variants, use early stopping and per-label boosters when needed.
- Compare XGBoost's accuracy vs overfitting risk.

---

# SpaCy Lemmatization & Re-vectorization

- Clean and lemmatize text with SpaCy; rebuild TF‑IDF on lemmas to reduce noise.
- Often improves generalization by consolidating word forms.

---

# Threshold Tuning (Caution)

- Brute-force per-label thresholds can increase reported F1 but may leak test information.
- Prefer OOF-based threshold selection to avoid leakage.

---

# Data Augmentation (Optional)

- Augment rare toxic samples (synonym swaps, keyboard noise, back-translation) to increase sample diversity.
- Use carefully to avoid label noise.

---

# Early Stopping & Per-label Training

- Use explicit validation sets or native xgb.train with early stopping per label.
- Practical defense against overfitting for complex models.

---

# OOF-based Conservative Model Search (Final)

- Use Out-Of-Fold predictions and grid-search (TruncatedSVD + LogisticRegression) to find models with OOF gap ≤ 5%.
- Choose an extreme regularization to guarantee low overfitting gap for production.

---

# Final Artifact & Inference

- Save and version the conservative model `logreg_spacy_low_overfit_official.pkl`.
- Provide a simple `predict_texts` helper (preprocess → TF‑IDF → SVD → per-label predict_proba).

---

# Recommendations & Next Steps

- Deploy the low-overfit model for production stability.
- Keep the XGBoost model as a high-performance comparison if more data becomes available.
- Consider collecting more labeled data and re-running OOF grid searches periodically.

---

# Questions

- Which trade-off would your stakeholders prefer: higher absolute F1 or lower overfitting risk?

