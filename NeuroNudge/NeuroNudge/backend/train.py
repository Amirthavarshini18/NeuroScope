import argparse
import os
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def load_and_merge():
    try:
        go = load_dataset('debarshichanda/goemotions')
    except Exception:
        try:
            go = load_dataset('go_emotions')
        except Exception:
            go = None
    try:
        dd = load_dataset('thedevastator/dailydialog-unlock-the-conversation-potential-in')
    except Exception:
        try:
            dd = load_dataset('daily_dialog')
        except Exception:
            dd = None

    rows = []
    if go:
        for split in go.keys():
            for ex in go[split]:
                text = ex.get('text') or ex.get('content') or ex.get('sentence') or ''
                label = ex.get('label') or ex.get('labels')
                if isinstance(label, list):
                    if len(label) == 0:
                        continue
                    label = label[0]
                rows.append({'text': text, 'label': str(label)})
    if dd:
        for split in dd.keys():
            for ex in dd[split]:
                if 'dialog' in ex:
                    text = ' '.join(ex['dialog']) if isinstance(ex['dialog'], list) else (ex.get('text') or '')
                else:
                    text = ex.get('text') or ''
                label = ex.get('emotion') or ex.get('label') or 'neutral'
                rows.append({'text': text, 'label': str(label)})

    df = pd.DataFrame(rows)
    df = df[df['text'].str.len() > 0]
    return df


def train(output_path: str):
    df = load_and_merge()
    if df.empty:
        return
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipe, output_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'models', 'emotion_pipe.pkl'))
    args = p.parse_args()
    train(args.output)
