# XGBoost Multi-Label Toxicity Classification Pipeline

## Project Overview

**Objective:** Build and compare machine learning models to classify toxic comments across six different toxicity dimensions.

**Dataset:** `synonym_youtoxic_english_1000.csv`
- Total samples: 3,751 comments
- Target labels: 6 binary classification tasks
- Text column: `Text`

**Target Categories:**
1. `IsToxic` - General toxicity
2. `IsAbusive` - Abusive language
3. `IsProvocative` - Provocative content
4. `IsObscene` - Obscene language
5. `IsHatespeech` - Hate speech
6. `IsRacist` - Racist content

---

## Complete Pipeline

### Phase 1: Data Loading and Preparation

**Step 1.1: Import Dependencies**
```python
Libraries Used:
- pandas: Data manipulation
- spacy: NLP preprocessing (en_core_web_sm model)
- re: Regular expressions for text cleaning
- xgboost: XGBClassifier for model training
- sklearn: Train/test split, metrics, vectorization
- gensim: Word2Vec embeddings
```

**Step 1.2: Load Dataset**
- Loaded CSV file with 3,751 rows and 16 columns
- Verified presence of target columns
- Identified text column: `Text`

---

### Phase 2: Text Preprocessing Pipeline

**Step 2.1: URL Removal**
```python
Method: Regular Expression Pattern Matching
Pattern: r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|edu|gov|io|co|ai|ly)\S*'
Result: Removed 12 URLs from the dataset
```

**Step 2.2: Special Character Removal**
```python
Method: Regular Expression Substitution
Pattern: r'[^\w\s\'"!?.,-]'
Process:
- Remove emojis and unusual symbols
- Keep: letters, numbers, basic punctuation
- Collapse multiple spaces
Result: Removed 1,690 special characters
```

**Step 2.3: Text Lowercasing**
```python
Method: String method .lower()
Purpose: Normalize all text to lowercase
Result: Uniform text case for better feature matching
```

**Step 2.4: Tokenization and Lemmatization (spaCy)**
```python
Model: en_core_web_sm (parser disabled for speed)
Process:
1. Tokenize text into individual words
2. Apply lemmatization (convert to base form)
3. Remove stop words (is, the, a, etc.)
4. Remove punctuation and whitespace
5. Convert to lowercase

Output: New column 'tokens' containing list of lemmas
Example: "running quickly" → ["run", "quickly"]
```

**Step 2.5: High-Frequency Token Removal**
```python
Method: Frequency Analysis + Custom Filtering
Process:
1. Count all token occurrences across dataset
2. Identify top 100 most frequent tokens
3. Remove these tokens from all documents

Top tokens removed: 'black', 'people', 'white', 't', 's', 'police', etc.
Rationale: Domain-specific high-frequency words don't distinguish toxicity
Result: More discriminative vocabulary
```

---

### Phase 3: Feature Engineering

#### Approach A: TF-IDF Vectorization

**Step 3A.1: Text Reconstruction**
```python
Method: Join tokenized lists back into strings
Input: ['token1', 'token2', 'token3']
Output: 'token1 token2 token3'
Column created: 'cleaned_text'
```

**Step 3A.2: Train/Test Split**
```python
Method: train_test_split (sklearn)
Parameters:
- test_size: 0.2 (20% test data)
- random_state: 42
- stratify: yes (maintains class balance)

Split per target (example for IsToxic):
- Training set: 3,000 samples
- Test set: 751 samples
```

**Step 3A.3: TF-IDF Transformation**
```python
Method: TfidfVectorizer (sklearn)
Parameters:
- max_df: 0.8 (ignore words in >80% of docs)
- min_df: 5 (ignore words in <5 docs)
- ngram_range: (1, 2) (unigrams + bigrams)

Process:
1. Fit vectorizer on training data ONLY
2. Transform both training and test data

Output:
- Vocabulary size: 3,909 features
- X_train_tfidf: (3,000, 3,909) sparse matrix
- X_test_tfidf: (751, 3,909) sparse matrix
```

#### Approach B: Word2Vec Embeddings

**Step 3B.1: Train Word2Vec Model**
```python
Method: gensim.models.Word2Vec
Parameters:
- vector_size: 100
- window: 5
- min_count: 5
- workers: 4
- seed: 42

Training corpus: df['tokens'] (list of token lists)
Result: 2,488 unique words in vocabulary
```

**Step 3B.2: Document Vector Averaging**
```python
Method: Average word vectors for each document
Process:
1. For each token in document:
   - Look up its 100-dimensional vector
   - Add to running sum
2. Divide by count of valid tokens
3. Return averaged 100-dimensional vector

Output: X_w2v matrix (3,751, 100)
```

---

### Phase 4: Model Training and Evaluation

**XGBoost Configuration (Both Feature Sets)**
```python
Model: XGBClassifier
Hyperparameters:
- objective: 'binary:logistic'
- n_estimators: 100
- learning_rate: 0.1
- random_state: 42
- eval_metric: 'logloss'
- n_jobs: -1 (use all CPU cores)
```

**Training Process (Per Target)**
```python
For each of 6 target labels:
1. Split data with stratification
2. Vectorize/transform features
3. Train XGBoost model
4. Make predictions on train and test sets
5. Calculate metrics:
   - Weighted F1 Score (train)
   - Weighted F1 Score (test)
   - Overfit Margin (train F1 - test F1)
   - Per-class F1 scores
   - Training time
```

---

## Results Comparison

### TF-IDF Feature Set Performance

| Target | Test F1 (Weighted) | Minority Class F1 | Overfit Margin | Training Time |
|--------|-------------------|-------------------|----------------|---------------|
| IsToxic | 0.8453 | 0.3101 ⚠️ | 0.0323 ✓ | 4.27s |
| IsAbusive | 0.7712 | 0.5963 | 0.0496 ✓ | 3.29s |
| IsProvocative | 0.7462 | 0.8664 | 0.0823 ✓ | 2.16s |
| IsObscene | 0.8571 | 0.9283 | 0.0441 ✓ | 1.85s |
| IsHatespeech | 0.8340 | 0.9107 | 0.0319 ✓ | 2.43s |
| IsRacist | 0.8413 | 0.9165 | 0.0394 ✓ | 2.17s |
| **Average** | **0.8159** | **0.7564** | **0.0466** | **2.70s** |

### Word2Vec Feature Set Performance

| Target | Test F1 (Weighted) | Minority Class F1 | Overfit Margin | Training Time |
|--------|-------------------|-------------------|----------------|---------------|
| IsToxic | 0.8270 | 0.2308 | 0.1535 ⚠️ | 7.71s |
| IsAbusive | 0.6892 | 0.4834 | 0.2838 ⚠️ | 4.94s |
| IsProvocative | 0.7289 | 0.8536 | 0.2522 ⚠️ | 2.76s |
| IsObscene | 0.8205 | 0.9184 | 0.1704 ⚠️ | 3.66s |
| IsHatespeech | 0.7837 | 0.8849 | 0.2073 ⚠️ | 2.57s |
| IsRacist | 0.8012 | 0.8969 | 0.1911 ⚠️ | 2.11s |
| **Average** | **0.7751** | **0.7113** | **0.2097** | **3.96s** |

---

## Key Findings

### Winner: TF-IDF Features

**Advantages:**
- ✅ Higher F1 scores across all targets
- ✅ Excellent generalization (low overfitting: 0.0466)
- ✅ 3,909 features capture fine-grained patterns
- ✅ Better minority class performance

**Disadvantages:**
- ⚠️ High-dimensional sparse matrices
- ⚠️ Slightly slower training for some targets

### Word2Vec Features

**Advantages:**
- ✅ Lower dimensionality (100 features)
- ✅ Captures semantic relationships

**Disadvantages:**
- ❌ Severe overfitting (margin: 0.2097)
- ❌ Lower test F1 scores
- ❌ Poor generalization to unseen data
- ❌ Averaging loses context information

---

## Critical Issues Identified

### ⚠️ Class Imbalance Problem

**IsToxic Category:**
- Test set distribution: 108 False, 643 True
- Minority class F1: **0.3101** (critically low)
- Model heavily biased toward majority class

**Recommendation:**
```python
Next steps to address imbalance:
1. Use scale_pos_weight in XGBoost
2. Apply SMOTE (Synthetic Minority Oversampling)
3. Adjust class weights
4. Collect more minority class samples
```

---

## Deployment Artifacts

### Saved Models and Components

**Location:** `../resources/models/`

**Files Created:**
```
tfidf_vectorizer.pkl                    # Feature extractor
xgb_tfidf_model_IsToxic.json           # Model 1
xgb_tfidf_model_IsAbusive.json         # Model 2
xgb_tfidf_model_IsProvocative.json     # Model 3
xgb_tfidf_model_IsObscene.json         # Model 4
xgb_tfidf_model_IsHatespeech.json      # Model 5
xgb_tfidf_model_IsRacist.json          # Model 6
```

**Loading for Inference:**
```python
import joblib
from xgboost import XGBClassifier

# Load vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load model
model = XGBClassifier()
model.load_model('xgb_tfidf_model_IsToxic.json')

# Predict new text
new_text = ["Your comment here"]
features = vectorizer.transform(new_text)
prediction = model.predict(features)
```

---

## Technical Stack Summary

**NLP Processing:**
- spaCy (en_core_web_sm)
- Regular expressions (re)

**Feature Engineering:**
- TF-IDF Vectorizer (sklearn)
- Word2Vec (gensim)

**Machine Learning:**
- XGBoost Classifier
- Train/test split with stratification

**Evaluation Metrics:**
- Weighted F1 Score
- Per-class F1 Score
- Overfitting margin (Train F1 - Test F1)

**Data Manipulation:**
- pandas
- numpy
