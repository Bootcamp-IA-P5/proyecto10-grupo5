#!/usr/bin/env python3
import os, json, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import spacy

RND = 42
ROOT = '/workspaces/proyecto10-grupo5'
DATA_PATH = os.path.join(ROOT, 'synonym_youtoxic_english_1000.csv')
OUT_DIR = os.path.join(ROOT, 'eda', 'exp_synonym_1000')
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading data:', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# detect targets
possible_targets = ['IsToxic','IsAbusive','IsProvocative','IsObscene','IsHatespeech','IsRacist']
present_targets = [c for c in possible_targets if c in df.columns]
if len(present_targets)==0:
    raise SystemExit('No target columns found')
print('Targets:', present_targets)

# detect text column (case-insensitive, accept common variants)
cols_lower = {c.lower(): c for c in df.columns}
text_col = None
for cand in ['text', 'comment_text', 'comment']:
    if cand in cols_lower:
        text_col = cols_lower[cand]
        break
if text_col is None:
    raise SystemExit('No text column found')
print('Text column:', text_col)

# Lemmatize (re-use saved pickled tfidf if possible, but we need lemmatized text to transform)
print('Loading spaCy model...')
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    import subprocess
    subprocess.check_call(['python','-m','spacy','download','en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

def spacy_lemmatize(series):
    texts = []
    for doc in nlp.pipe(series.astype(str).tolist(), batch_size=50, disable=['ner','parser']):
        lem = ' '.join([t.lemma_.lower() for t in doc if (not t.is_punct) and (not t.is_space)])
        texts.append(lem)
    return pd.Series(texts)

if 'lemmatized' not in df.columns:
    print('Creating lemmatized column (this may take a minute)...')
    df['lemmatized'] = spacy_lemmatize(df[text_col])
else:
    print('lemmatized column already present')

# Split (same random state used in notebook)
stratify_col = df[present_targets[0]] if present_targets[0] in df.columns else None
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RND, stratify=stratify_col if stratify_col is not None else None)
print('Train/Test shapes', train_df.shape, test_df.shape)

# Load or fit TF-IDF
tfidf_path = os.path.join(OUT_DIR, 'tfidf_spacy_synonym_1000.pkl')
from sklearn.feature_extraction.text import TfidfVectorizer
if os.path.exists(tfidf_path):
    print('Loading saved TF-IDF from', tfidf_path)
    tfidf = joblib.load(tfidf_path)
else:
    print('Fitting TF-IDF on training lemmatized text')
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
    tfidf.fit(train_df['lemmatized'])
    joblib.dump(tfidf, tfidf_path)

X_train = tfidf.transform(train_df['lemmatized'])
X_test = tfidf.transform(test_df['lemmatized'])

# Prepare labels
y_train = train_df[present_targets].astype(int).values
y_test = test_df[present_targets].astype(int).values

# Per-label KFold XGBoost to compute OOF and test probs
xgb_params = dict(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=RND)
print('Training per-label XGBoost to recompute OOF and test probabilities...')
probas_oof = np.zeros((len(train_df), len(present_targets)))
test_probas = np.zeros((len(test_df), len(present_targets)))

kf = KFold(n_splits=5, shuffle=True, random_state=RND)
for i, lab in enumerate(present_targets):
    y_col = y_train[:, i]
    oof = np.zeros(len(train_df))
    test_p = np.zeros(len(test_df))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y_col[tr_idx]
        y_val = y_col[val_idx]
        m = XGBClassifier(**{k:v for k,v in xgb_params.items() if k!='use_label_encoder'})
        # suppress warnings; fit without early stopping to avoid xgboost API incompatibilities
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof[val_idx] = m.predict_proba(X_val)[:,1]
        test_p += m.predict_proba(X_test)[:,1] / kf.get_n_splits()
    probas_oof[:, i] = oof
    test_probas[:, i] = test_p
    print('Done label', lab)

# Save recomputed OOF & test prob arrays for reproducibility
np.savez_compressed(os.path.join(OUT_DIR, 'xgb_oof_and_test_probas_synonym_1000.npz'), probas_oof=probas_oof, test_probas=test_probas)

# Find per-label best thresholds on OOF probabilities
thresholds = {}
best_scores = {}
for i, lab in enumerate(present_targets):
    y_col = y_train[:, i]
    best_t = 0.5
    best_f = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probas_oof[:, i] >= t).astype(int)
        f = f1_score(y_col, preds)
        if f > best_f:
            best_f = f
            best_t = float(t)
    thresholds[lab] = best_t
    best_scores[lab] = best_f
    print(f'Label {lab}: best OOF threshold {best_t:.2f} -> OOF F1 {best_f:.4f}')

# Apply thresholds to test_probas and compute test F1
test_preds_thresh = np.zeros_like(test_probas, dtype=int)
for i, lab in enumerate(present_targets):
    test_preds_thresh[:, i] = (test_probas[:, i] >= thresholds[lab]).astype(int)

macro_test_f1 = f1_score(y_test, test_preds_thresh, average='macro')
macro_oof_f1 = f1_score(y_train, (probas_oof >= np.array([thresholds[l] for l in present_targets])).T.astype(int).T, average='macro')

print('OOF macro F1 after per-label thresholding:', macro_oof_f1)
print('Test macro F1 after per-label thresholding:', macro_test_f1)

# Save thresholds and updated summary
th_out = os.path.join(OUT_DIR, 'xgb_oof_thresholds_synonym_1000.json')
with open(th_out, 'w') as f:
    json.dump({'thresholds':thresholds, 'best_oof_f1_per_label':best_scores, 'oof_macro_f1':float(macro_oof_f1), 'test_macro_f1':float(macro_test_f1)}, f, indent=2)
print('Saved thresholds to', th_out)

# Update summary file
summary_path = os.path.join(OUT_DIR, 'summary_synonym_1000_thresholded.json')
summary = {}
if os.path.exists(os.path.join(OUT_DIR, 'summary_synonym_1000.json')):
    with open(os.path.join(OUT_DIR, 'summary_synonym_1000.json')) as f:
        summary = json.load(f)
summary['xgboost']['oof_f1_after_thresholds'] = float(macro_oof_f1)
summary['xgboost']['test_f1_after_thresholds'] = float(macro_test_f1)
summary['xgboost']['thresholds'] = thresholds
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved updated summary to', summary_path)

# Part 2: generate CSV of available artifacts and metrics
print('\nGenerating artifacts CSV...')
art_files = [
    os.path.join(ROOT, 'eda', 'logreg_spacy_oof_gap_search_record.json'),
    os.path.join(ROOT, 'eda', 'xgboost_tuned_overfit_gap.json'),
    os.path.join(OUT_DIR, 'summary_synonym_1000.json'),
    summary_path,
]
rows = []
for p in art_files:
    if not os.path.exists(p):
        continue
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        continue
    entry = {'file': os.path.relpath(p, ROOT)}
    # try to extract f1/test/gap fields heuristically
    if 'conservative' in data:
        entry.update({
            'type': 'exp_summary',
            'conservative_oof_f1': data['conservative'].get('oof_f1'),
            'conservative_test_f1': data['conservative'].get('test_f1'),
            'conservative_overfit_gap': data['conservative'].get('overfit_gap')
        })
    if 'oof_f1' in data and 'test_f1' in data:
        # likely a record like logreg_spacy_oof_gap_search_record.json
        entry.update({'type':'record','oof_f1': data.get('oof_f1'), 'test_f1': data.get('test_f1'), 'overfit_gap': data.get('overfit_gap')})
    if 'train_macro_f1_tuned' in data:
        entry.update({'type':'xgb_tuned','train_macro_f1_tuned': data.get('train_macro_f1_tuned'), 'test_macro_f1_tuned': data.get('test_macro_f1_tuned'), 'overfit_gap': data.get('overfit_gap')})
    rows.append(entry)

df_art = pd.DataFrame(rows)
csv_out = os.path.join(ROOT, 'eda', 'doc', 'artifacts_metrics_synonym_vs_original.csv')
df_art.to_csv(csv_out, index=False)
print('Wrote artifacts CSV to', csv_out)

print('\nAll done.')
