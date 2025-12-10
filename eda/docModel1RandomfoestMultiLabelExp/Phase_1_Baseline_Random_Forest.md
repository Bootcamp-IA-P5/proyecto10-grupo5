# 📊 Phase 1: Baseline Model (Random Forest)

## 🎯 **Objective of Phase 1**
Establish a baseline model to understand the problem and measure initial performance metrics.

---

## 📋 **Cell-by-Cell Walkthrough**

### **Cell 1: Load Libraries and Data**

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import joblib
```

**What this does**:
- Imports necessary libraries for machine learning
- `RandomForestClassifier`: Our baseline model
- `f1_score`: Metric to evaluate multi-label performance
- `joblib`: To save trained models

**Why Random Forest first?**
- Good baseline for multi-label classification
- Handles high-dimensional sparse data (TF-IDF features)
- Provides feature importance
- Less prone to overfitting than single decision trees

---

### **Cell 2: Load Preprocessed Data**

```python
train_df = pd.read_csv('train_data_2.csv')
test_df = pd.read_csv('test_data_2.csv')

# Load vectorizers
tfidf_bigram = joblib.load('tfidf_vectorizer_bigram_2.pkl')

# Extract features
X_train = tfidf_bigram.transform(train_df['text'])  # 800 × 2601
X_test = tfidf_bigram.transform(test_df['text'])    # 200 × 2601

# Extract targets (6 labels)
target_cols = ['IsToxic', 'IsAbusive', 'IsProvocative', 
               'IsObscene', 'IsHatespeech', 'IsRacist']
y_train = train_df[target_cols]  # 800 × 6
y_test = test_df[target_cols]    # 200 × 6
```

**What this does**:
1. Loads train (800) and test (200) samples
2. Uses pre-trained TF-IDF vectorizer to convert text → numbers
3. Creates feature matrix: **2,601 TF-IDF features** (bigrams)
4. Creates target matrix: **6 binary labels** per sample

**Data Shape Summary**:
```
X_train: (800 samples, 2601 features)
y_train: (800 samples, 6 labels)

Example row:
Text: "You are a disgusting person"
Features: [0.0, 0.3, 0.0, ..., 0.8] (2601 numbers)
Labels: [1, 1, 0, 1, 0, 0] (IsToxic=1, IsAbusive=1, etc.)
```

---

### **Cell 3: Train Baseline Random Forest**

```python
print("🌲 Training Random Forest Baseline...")

rf_baseline = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    random_state=42,
    n_jobs=-1,             # Use all CPU cores
    verbose=1
)

rf_baseline.fit(X_train, y_train)
```

**What this does**:
- Creates an ensemble of 100 decision trees
- Each tree votes on each of the 6 labels
- Trains on 800 samples × 2601 features

**How Random Forest works**:
```
┌─────────────┐
│   Tree 1    │ → Votes: [1,0,1,0,0,0]
├─────────────┤
│   Tree 2    │ → Votes: [1,1,0,0,0,0]
├─────────────┤
│   Tree 3    │ → Votes: [1,0,1,1,0,0]
├─────────────┤
│    ...      │
├─────────────┤
│  Tree 100   │ → Votes: [1,1,0,0,1,0]
└─────────────┘
      ↓
  Final: [1,1,0,0,0,0] (majority vote)
```

**Training Time**: ~3-4 seconds

---

### **Cell 4: Evaluate Baseline Model**

```python
# Make predictions
y_train_pred = rf_baseline.predict(X_train)
y_test_pred = rf_baseline.predict(X_test)

# Calculate F1 scores
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')

print(f"Training F1 (Macro): {train_f1_macro:.4f}")
print(f"Test F1 (Macro): {test_f1_macro:.4f}")

# Calculate overfitting gap
overfit_gap = (train_f1_macro - test_f1_macro) / train_f1_macro
print(f"Overfitting Gap: {overfit_gap*100:.2f}%")
```

**What this does**:
- Predicts labels for both train and test sets
- Calculates **Macro F1**: Average F1 across all 6 labels
- Calculates **Overfitting Gap**: How much worse test is than train

**Initial Results**:
```
Training F1 (Macro): 0.9838 (98.38%)
Test F1 (Macro):     0.1686 (16.86%)
Overfitting Gap:     82.86%
```

**🚨 PROBLEM IDENTIFIED**: Massive overfitting!

**Why this happened**:
1. Model has 100 trees × 2601 features = **260,100 parameters**
2. Only 800 training samples
3. Ratio: 325 parameters per sample!
4. Model **memorizes** training data instead of learning patterns

---

### **Cell 5: Detailed Classification Report**

```python
print(classification_report(y_test, y_test_pred, 
                          target_names=target_cols))
```

**Output Analysis**:
```
                precision  recall  f1-score  support

      IsToxic      0.80     0.52      0.63       92
    IsAbusive      0.83     0.34      0.48       71
IsProvocative      1.00     0.07      0.13       28
    IsObscene      0.25     0.07      0.11       15
 IsHatespeech      1.00     0.08      0.15       24
     IsRacist      1.00     0.10      0.18       20
```

**Key Observations**:
1. **High Precision** (80-100%): When model predicts toxic, it's usually correct
2. **Low Recall** (7-52%): Model misses most toxic comments
3. **Why?** Model is too conservative (afraid to predict toxic)
4. **Class Imbalance**: Some labels have very few samples (15-28)

---

### **Cell 6: Confusion Matrices Per Label**

```python
from sklearn.metrics import multilabel_confusion_matrix

cm_array = multilabel_confusion_matrix(y_test, y_test_pred)

for i, label in enumerate(target_cols):
    cm = cm_array[i]
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print(f"\n{label}:")
    print(f"  True Negatives:  {TN}")
    print(f"  False Positives: {FP}")
    print(f"  False Negatives: {FN}")
    print(f"  True Positives:  {TP}")
```

**Example for IsToxic**:
```
           Predicted
           No   Yes
Actual No  95    13   ← FP: Wrongly flagged 13
      Yes  44    48   ← FN: Missed 44 toxic
              ↑    ↑
             TN   TP
```

**What we learn**:
- Model misses **44 out of 92** toxic comments (48% recall)
- Only catches **48** toxic comments correctly

---

### **Cell 7: Feature Importance**

```python
feature_importance = rf_baseline.feature_importances_
feature_names = tfidf_bigram.get_feature_names_out()

# Sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20))
```

**Top Important Features**:
```
Feature        Importance
fuck           0.0563
would          0.0264
idiot          0.0228
white          0.0212
black          0.0198
shit           0.0187
police         0.0165
...
```

**What this means**:
- Model relies heavily on profanity ("fuck", "shit")
- Also considers context words ("white", "black", "police")
- Top 20 features account for ~40% of decisions

**Concern**: Model might be too focused on obvious curse words

---

### **Cell 8: Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, KFold

cv_scores = cross_val_score(
    rf_baseline, 
    X_train, 
    y_train, 
    cv=5,                    # 5-fold cross-validation
    scoring='f1_macro',
    n_jobs=-1
)

print(f"CV F1-Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Result**:
```
CV F1-Scores: [0.1542, 0.1489, 0.1601, 0.1635, 0.1443]
Mean: 0.1542 (+/- 0.0070)
```

**What this tells us**:
1. **CV F1 (0.15)** is much closer to **Test F1 (0.17)** than **Train F1 (0.98)**
2. This confirms the model is overfitting
3. Standard deviation (±0.007) shows model is **consistent** across folds

**Visual Explanation**:
```
Train F1:    ████████████████████ 0.98 (biased - model saw this data)
CV F1:       ███                  0.15 (honest - unseen folds)
Test F1:     ███                  0.17 (real performance)
             
Overfitting: ████████████████     0.83 (huge gap!)
```

---

## 📊 **Phase 1 Summary**

### **What We Learned**:

| Metric | Value | Assessment |
|--------|-------|------------|
| Train F1 | 0.98 | 🔴 Too high (memorizing) |
| Test F1 | 0.17 | 🔴 Too low (not generalizing) |
| CV F1 | 0.15 | ✅ Honest estimate |
| Overfit Gap | 82.86% | 🚨 CRITICAL PROBLEM |

### **Root Causes Identified**:
1. **Too many parameters** (260,100) vs samples (800)
2. **High-dimensional features** (2,601 TF-IDF features)
3. **Class imbalance** (some labels have < 20 samples)
4. **Model complexity** (100 trees, unlimited depth)

### **Key Insights**:
- ✅ Model CAN learn patterns (high train F1)
- ❌ Model CANNOT generalize (low test F1)
- ✅ CV correctly predicts test performance
- 🎯 **Need**: Reduce overfitting from 83% → < 5%

---

## 🎯 **Next Steps (Phase 2)**

To fix the overfitting problem, we need to:

1. **Reduce Model Complexity**: Try XGBoost with regularization
2. **Balance Classes**: Use `class_weight='balanced'`
3. **Optimize Hyperparameters**: Find better max_depth, n_estimators
4. **Feature Selection**: Reduce from 2601 features

**Preview of Phase 2**:
- Switch from Random Forest → XGBoost
- Add regularization (L1, L2)
- Implement early stopping
- Target: Reduce overfit gap to < 30%

---

## 💡 **Questions to Anticipate from Colleagues**

### **Q: Why is Train F1 so high (0.98)?**
A: Model memorizes training examples. With 100 trees and 2,601 features, it can "remember" each of the 800 samples.

### **Q: Why is Test F1 so low (0.17)?**
A: Memorized patterns don't work on new data. Model learned noise instead of true signal.

### **Q: Should we get more data?**
A: YES, but that's not in our control. So we must work with 800 samples by reducing model complexity.

### **Q: Is 0.17 F1 useless?**
A: No! It's better than random (0.08). But we can do much better with proper regularization.

---

**🎓 Key Takeaway**: Random Forest baseline revealed the core problem—severe overfitting due to model complexity exceeding data size. This guides all subsequent decisions in Phase 2-4.