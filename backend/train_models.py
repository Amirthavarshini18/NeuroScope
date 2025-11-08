import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

TARGET_LABELS = ['fatigue', 'anxiety', 'low_mood', 'motivated']


def heuristic_label_map(text, raw_label=None):
    t = (text or '').lower()
    labels = set()
    if raw_label:
        rl = str(raw_label).lower()
        if any(w in rl for w in ['tired', 'fatigue', 'sleep', 'exhaust']):
            labels.add('fatigue')
        if 'anxi' in rl or 'fear' in rl or 'nerv' in rl or 'panic' in rl or 'worry' in rl:
            labels.add('anxiety')
        if 'sad' in rl or 'depress' in rl or 'low' in rl or 'gloom' in rl:
            labels.add('low_mood')
        if 'happy' in rl or 'joy' in rl or 'excite' in rl or 'motiv' in rl or 'pride' in rl:
            labels.add('motivated')

    if any(w in t for w in ['tired', 'exhaust', 'sleepy', 'fatigue']):
        labels.add('fatigue')
    if any(w in t for w in ['anxious', 'anxiety', 'nervous', 'panick', 'panic', 'worry', 'worried']):
        labels.add('anxiety')
    if any(w in t for w in ['sad', 'depressed', 'down', 'low', 'unhappy', 'miserable']):
        labels.add('low_mood')
    if any(w in t for w in ['motiv', 'excited', 'driven', 'energi', 'productive', 'ready']):
        labels.add('motivated')

    return sorted(labels)


def load_and_map():
    rows = []
    try:
        go = load_dataset('debarshichanda/goemotions')
    except Exception:
        try:
            go = load_dataset('go_emotions')
        except Exception:
            go = None
    if go:
        go_label_names = None
        try:
            go_label_names = go['train'].features['labels'].feature.names
        except Exception:
            go_label_names = None
        for split in go.keys():
            for ex in go[split]:
                text = ex.get('text') or ex.get('content') or ex.get('sentence') or ''
                raw_label = ex.get('labels') or ex.get('label')
                mapped = []
                if isinstance(raw_label, list):
                    for i, v in enumerate(raw_label):
                        if v:
                            label_name = go_label_names[i] if go_label_names else str(i)
                            mapped += heuristic_label_map(text, label_name)
                else:
                    mapped = heuristic_label_map(text, raw_label)
                rows.append({'text': text, 'labels': list(set(mapped))})

    try:
        dd = load_dataset('thedevastator/dailydialog-unlock-the-conversation-potential-in')
    except Exception:
        try:
            dd = load_dataset('daily_dialog')
        except Exception:
            dd = None
    if dd:
        for split in dd.keys():
            for ex in dd[split]:
                if 'dialog' in ex and isinstance(ex['dialog'], list):
                    text = ' '.join(ex['dialog'])
                else:
                    text = ex.get('text') or ex.get('sentence') or ''
                raw_label = ex.get('emotion') or ex.get('label')
                mapped = heuristic_label_map(text, raw_label)
                rows.append({'text': text, 'labels': list(set(mapped))})

    df = pd.DataFrame(rows)
    df = df[df['text'].str.len() > 0]
    return df


def prepare_multilabel(df):
    y = np.zeros((len(df), len(TARGET_LABELS)), dtype=int)
    for i, labs in enumerate(df['labels'].tolist()):
        for l in labs:
            if l in TARGET_LABELS:
                y[i, TARGET_LABELS.index(l)] = 1
    return df['text'].tolist(), y


def train_level1(X_train, y_train, X_val, y_val, output_path):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3)
    nb = MultinomialNB(alpha=0.7)
    ov = OneVsRestClassifier(nb)
    pipe = Pipeline([('tfidf', vect), ('clf', ov)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)
    report(y_val, y_prob)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({'pipeline': pipe, 'labels': TARGET_LABELS}, output_path)


def train_level2(X_train, y_train, X_val, y_val, output_path):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
    ov = OneVsRestClassifier(lr)
    pipe = Pipeline([('tfidf', vect), ('clf', ov)])
    pipe.fit(X_train, y_train)

    X_train_feat = pipe.named_steps['tfidf'].transform(X_train)
    X_val_feat = pipe.named_steps['tfidf'].transform(X_val)

    calibrated_clfs = []
    for i in range(y_train.shape[1]):
        base_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
        base_clf.fit(X_train_feat, y_train[:, i])
        try:
            calib = CalibratedClassifierCV(base_clf, cv='prefit', method='sigmoid')
            calib.fit(X_val_feat, y_val[:, i])
        except Exception:
            calib = base_clf
        calibrated_clfs.append(calib)

    model_bundle = {'tfidf': pipe.named_steps['tfidf'], 'calibrated_clfs': calibrated_clfs, 'labels': TARGET_LABELS}

    y_prob = np.column_stack([c.predict_proba(X_val_feat)[:, 1] if hasattr(c, 'predict_proba') else c.decision_function(X_val_feat) for c in calibrated_clfs])
    report(y_val, y_prob)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_bundle, output_path)


def report(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    print('Macro-F1:', f1_score(y_true, y_pred, average='macro', zero_division=0))
    print('Micro-F1:', f1_score(y_true, y_pred, average='micro', zero_division=0))
    aps = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_score[:, i])
        except Exception:
            ap = 0.0
        aps.append(ap)
    for lbl, ap in zip(TARGET_LABELS, aps):
        print(f'AP {lbl}: {ap:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--level', type=int, choices=[1, 2], default=2)
    p.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'models', 'level_pipe.joblib'))
    p.add_argument('--test_size', type=float, default=0.1)
    args = p.parse_args()

    df = load_and_map()
    if df.empty:
        return

    X, y = prepare_multilabel(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y.sum(axis=1))

    if args.level == 1:
        train_level1(X_train, y_train, X_val, y_val, args.output)
    else:
        train_level2(X_train, y_train, X_val, y_val, args.output)


if __name__ == '__main__':
    main()
