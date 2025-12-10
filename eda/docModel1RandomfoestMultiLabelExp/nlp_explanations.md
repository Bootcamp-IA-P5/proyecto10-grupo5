# NLP Project Libraries Explained (YouTube Hate Speech Detection)

## 1. SpaCy

### a. What is SpaCy?

SpaCy is an **industrial‑grade Natural Language Processing (NLP)
library** designed for speed, efficiency, and production use. Unlike
NLTK (educational) or HuggingFace (deep-learning focused), SpaCy focuses
on **fast linguistic preprocessing** essential for classical ML models.

### b. Why use SpaCy?

For this project, SpaCy helps to:\
- Clean text linguistically\
- Tokenize words\
- Lemmatize words\
- Remove stopwords\
- Detect POS tags if needed\
- Normalize text for TF‑IDF and ML models

### c. Benefits

-   Very fast\
-   Accurate lemmatizer\
-   Built‑in stopword list\
-   Easy integration with scikit‑learn pipelines\
-   Optimized for production systems (what YouTube wants)

------------------------------------------------------------------------

### d. Step-by-Step Example (How SpaCy Works)

Input:

    "You're a disgusting racist!!!"

**Step 1 -- Remove URLs**\
No URLs → text stays the same.

**Step 2 -- Remove special characters**

    Youre a disgusting racist

**Step 3 -- Lowercase**

    youre a disgusting racist

**Step 4 -- SpaCy processing** SpaCy splits text into tokens:

    [Token("youre"), Token("a"), Token("disgusting"), Token("racist")]

**Step 5 -- Lemmatization + Stopword Removal** - "youre" → lemma="you",
stopword=True → removed\
- "a" → stopword=True → removed\
- "disgusting" → lemma="disgust" → kept\
- "racist" → lemma="racist" → kept

**Final Output:**

    "disgust racist"

This transforms messy human language into **clean ML features**.

------------------------------------------------------------------------

### e. Why remove stop words?

Stop words are: - extremely common words (the, a, is, you...) - not
useful for classification\
- add noise to TF‑IDF\
- increase feature size unnecessarily

Removing them improves: - accuracy\
- training speed\
- generalization

------------------------------------------------------------------------

### f. Lemmatization Applied to All Data

Lemmatization = convert words to their dictionary root.

Example:

**Before:**

    "You are a fucking idiot and should be ashamed!!!"

**After:**

    "fuck idiot ashamed"

Effects: - Normalizes phrases\
- Reduces vocabulary\
- Improves model performance\
- Makes TF‑IDF cleaner

------------------------------------------------------------------------

### g. Creating New TF‑IDF With Reduced Features

After cleaning & lemmatization: - We re‑fit a new TF‑IDF vectorizer\
- Vocabulary is now smaller\
- Model becomes more robust & faster

**Why used:**\
TF‑IDF transforms text into numeric features that machine learning can
understand.

**Example** Before cleaning:\
10,000 features\
After cleaning:\
2,000 features → **less noise, higher accuracy**

------------------------------------------------------------------------

------------------------------------------------------------------------

# 2. nlpaug (Text Augmentation)

### a. What is nlpaug?

A library for generating **new synthetic training data** from existing
text.

Useful for: - Class imbalance (hate speech usually \< normal comments) -
Making models more robust

------------------------------------------------------------------------

### b. Augmentation Methods

#### 1. How it works

It modifies existing text using controlled transformations: - Replace
words\
- Add noise\
- Swap characters\
- Insert spelling errors\
- Use synonyms

#### 2. Types

-   **Synonym Replacement**
-   **Keyboard Noise**
-   **Random Word Swap**
-   **Word Deletion**
-   **Contextual augmentation (BERT, RoBERTa)**

#### 3. Can you combine original + augmented?

**YES.**\
This is the recommended approach.\
Dataset = original + augmented.

#### 4. What type of data can be augmented?

-   Text\
-   Audio\
-   Images (but not needed here)

------------------------------------------------------------------------

### 5. Synonym Replacement

#### a. How it works

Replaces a word with its synonyms via WordNet or transformer models.

Example:

    "disgust racist"
    → "revolt racist"

#### b. Why useful?

-   Adds variation\
-   Helps generalization\
-   Prevents overfitting\
-   Helps with small datasets

------------------------------------------------------------------------

### 6. Keyboard Noise

#### a. How it works

Simulates typing errors:

    "racist" → "racost" or "racistt"

#### b. Why useful?

Models become robust to real‑world noisy YouTube comments.

------------------------------------------------------------------------

### 7. Re-vectorizing Augmented Data

After augmentation: 1. Combine all text\
2. Fit TF‑IDF again\
3. Train model on larger data

Why useful? - Vocabulary must update\
- TF‑IDF must reflect new words\
- Prevent mismatch during training

------------------------------------------------------------------------

### 8. Models You Can Use (LogReg, SVM, XGBoost)

Yes → you can train XGBoost, SVM, Logistic Regression, Random Forest,
etc.

How to train:

    model.fit(X_train_vectorized, y_train)

------------------------------------------------------------------------

------------------------------------------------------------------------

# 3. Out‑of‑Fold (OOF) Validation

### a. What is OOF?

A strategy used in cross‑validation to generate **predictions on unseen
data** during training.

### b. How it works

1.  Split data into K folds\
2.  Train on K-1 folds\
3.  Predict on the remaining fold\
4.  Save these predictions\
5.  Repeat for all folds\
6.  Combine all predictions → OOF predictions

### c. Why useful?

-   Prevents overfitting\
-   Provides unbiased predictions\
-   Used for stacking & model evaluation

### d. Cross-validation connection

OOF predictions are the **validation predictions** generated at each
fold.

### e. How to train

Train normally inside each fold.

------------------------------------------------------------------------

### f. SVD Analogy

#### 1. What is SVD?

SVD = matrix factorization → reduces dimensions.

#### 2. How it works

Converts high‑dimensional TF‑IDF (10,000 features) into a reduced space
(300 features).

#### 3. Dimensionality Reduction Strategy

-   Keep only the strongest semantic components\
-   Remove noise\
-   Improve LogisticRegression speed and accuracy

------------------------------------------------------------------------

------------------------------------------------------------------------

# 4. TruncatedSVD

### a. What is TruncatedSVD?

A dimensionality‑reduction technique (similar to PCA but works on sparse
TF‑IDF matrices).

### b. How it Works?

-   Takes TF‑IDF sparse matrix\
-   Computes top N components\
-   Reduces feature space while keeping semantic structure

------------------------------------------------------------------------

### c. Using TruncatedSVD in LogisticRegression

Pipeline:

    TF-IDF → TruncatedSVD → LogisticRegression

Benefits: - Smaller features\
- Faster training\
- Less overfitting

------------------------------------------------------------------------

### d. Grid Search for Optimal Configuration

Grid search tries different combinations: - n_components\
- regularization strength\
- ngram range\
- max_features

It evaluates each configuration using cross‑validation to find the best
setup.

------------------------------------------------------------------------

### e. Training Process (Full)

1.  Clean text\
2.  Lemmatize text\
3.  Vectorize with TF‑IDF\
4.  Reduce dimensions with SVD\
5.  Train Logistic Regression / XGBoost\
6.  Evaluate with OOF or K‑Fold\
7.  Tune with GridSearchCV
