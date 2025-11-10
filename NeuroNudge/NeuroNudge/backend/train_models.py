import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
import kagglehub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

TARGET_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']


def heuristic_label_map(text, raw_label=None):
    t = (text or '').lower()
    labels = set()
    if raw_label:
        rl = str(raw_label).lower()
        if 'joy' in rl or 'happy' in rl or 'excit' in rl or 'delight' in rl or 'pleased' in rl:
            labels.add('joy')
        if 'sad' in rl or 'depress' in rl or 'gloom' in rl or 'unhappy' in rl or 'sorrow' in rl:
            labels.add('sadness')
        if 'ang' in rl or 'furious' in rl or 'irritat' in rl or 'annoyed' in rl or 'rage' in rl:
            labels.add('anger')
        if 'fear' in rl or 'anxi' in rl or 'scare' in rl or 'afraid' in rl or 'nerv' in rl or 'panic' in rl or 'worry' in rl:
            labels.add('fear')
        if 'surpris' in rl or 'astonish' in rl or 'amaz' in rl or 'shock' in rl:
            labels.add('surprise')
        if 'neutral' in rl or 'calm' in rl:
            labels.add('neutral')

    if any(w in t for w in ['happy', 'joy', 'glad', 'excited', 'delighted', 'pleased', 'great', 'wonderful', 'awesome', 'fantastic']):
        labels.add('joy')
    if any(w in t for w in ['sad', 'depressed', 'down', 'unhappy', 'miserable', 'gloomy', 'sorrow', 'disappoint', 'upset']):
        labels.add('sadness')
    if any(w in t for w in ['angry', 'mad', 'furious', 'irritat', 'annoyed', 'rage', 'frustrat', 'unfair', 'outrage', 'pissed', 'resentful']):
        labels.add('anger')
    if any(w in t for w in ['scared', 'afraid', 'anxious', 'nervous', 'panic', 'worry', 'worried', 'fear', 'terrif', 'dread']):
        labels.add('fear')
    if any(w in t for w in ['surpris', 'astonish', 'amaz', 'shock', 'unexpected', 'sudden']):
        labels.add('surprise')

    return sorted(labels)


def load_and_map():
    rows = []
    
    print("Downloading GoEmotions dataset...")
    try:
        go_path = kagglehub.dataset_download("debarshichanda/goemotions")
        print(f"GoEmotions downloaded to: {go_path}")
        
        go_files = [f for f in os.listdir(go_path) if f.endswith('.csv') or f.endswith('.tsv')]
        print(f"Found files: {go_files}")
        
        if go_files:
            for go_file in go_files:
                try:
                    file_path = os.path.join(go_path, go_file)
                    sep = '\t' if go_file.endswith('.tsv') else ','
                    go_df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
                    print(f"Loaded {go_file}: {len(go_df)} rows, columns: {list(go_df.columns)}")
                    
                    for idx, row in go_df.iterrows():
                        text = str(row.get('text', '') or row.get('comment_text', '') or row.get('sentence', ''))
                        if len(text.strip()) < 3:
                            continue
                        
                        emotion_cols = [c for c in go_df.columns if c not in ['text', 'comment_text', 'id', 'sentence']]
                        raw_labels = []
                        for col in emotion_cols:
                            if row.get(col) == 1 or row.get(col) == True or str(row.get(col)).lower() == 'true':
                                raw_labels.append(col)
                        
                        mapped = []
                        for label in raw_labels:
                            mapped += heuristic_label_map(text, label)
                        
                        if not mapped:
                            mapped = heuristic_label_map(text, None)
                        
                        if mapped:
                            rows.append({'text': text, 'labels': list(set(mapped))})
                    
                    if len(rows) > 5000:
                        break
                except Exception as e:
                    print(f"Error loading {go_file}: {e}")
    except Exception as e:
        print(f"Failed to load GoEmotions: {e}")

    print("Downloading DailyDialog dataset...")
    try:
        dd_path = kagglehub.dataset_download("thedevastator/dailydialog-unlock-the-conversation-potential-in")
        print(f"DailyDialog downloaded to: {dd_path}")
        
        dd_files = [f for f in os.listdir(dd_path) if f.endswith('.csv')]
        print(f"Found files: {dd_files}")
        
        if dd_files:
            for dd_file in dd_files:
                try:
                    dd_df = pd.read_csv(os.path.join(dd_path, dd_file), on_bad_lines='skip')
                    print(f"Loaded {dd_file}: {len(dd_df)} rows, columns: {list(dd_df.columns)}")
                    
                    for idx, row in dd_df.iterrows():
                        text = str(row.get('dialog', '') or row.get('text', '') or row.get('sentence', ''))
                        if len(text.strip()) < 3:
                            continue
                        raw_label = row.get('emotion', '') or row.get('label', '')
                        
                        mapped = heuristic_label_map(text, raw_label)
                        if mapped:
                            rows.append({'text': text, 'labels': list(set(mapped))})
                except Exception as e:
                    print(f"Error loading {dd_file}: {e}")
    except Exception as e:
        print(f"Failed to load DailyDialog: {e}")

    df = pd.DataFrame(rows)
    df = df[df['text'].str.len() > 0]
    print(f"Total dataset size: {len(df)} samples")
    return df


def prepare_multilabel(df):
    y = np.zeros((len(df), len(TARGET_LABELS)), dtype=int)
    for i, labs in enumerate(df['labels'].tolist()):
        for l in labs:
            if l in TARGET_LABELS:
                y[i, TARGET_LABELS.index(l)] = 1
    
    label_counts = y.sum(axis=0)
    print(f"Label distribution: {dict(zip(TARGET_LABELS, label_counts))}")
    
    valid_labels_mask = label_counts >= 10
    if not valid_labels_mask.all():
        print(f"Warning: Some labels have <10 examples. Filtering to labels with sufficient data.")
        valid_indices = np.where(valid_labels_mask)[0]
        y = y[:, valid_indices]
        valid_labels = [TARGET_LABELS[i] for i in valid_indices]
        print(f"Using labels: {valid_labels}")
    else:
        valid_labels = TARGET_LABELS
    
    has_any_label = y.sum(axis=1) > 0
    X_filtered = [df['text'].tolist()[i] for i in range(len(df)) if has_any_label[i]]
    y_filtered = y[has_any_label]
    
    print(f"Filtered dataset: {len(X_filtered)} samples with labels")
    return X_filtered, y_filtered, valid_labels


def train_level1(X_train, y_train, X_val, y_val, output_path, labels):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3)
    nb = MultinomialNB(alpha=0.7)
    ov = OneVsRestClassifier(nb)
    pipe = Pipeline([('tfidf', vect), ('clf', ov)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)
    report(y_val, y_prob, labels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({'pipeline': pipe, 'labels': labels}, output_path)


def train_level2(X_train, y_train, X_val, y_val, output_path, labels):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
    ov = OneVsRestClassifier(lr)
    pipe = Pipeline([('tfidf', vect), ('clf', ov)])
    pipe.fit(X_train, y_train)

    X_train_feat = pipe.named_steps['tfidf'].transform(X_train)
    X_val_feat = pipe.named_steps['tfidf'].transform(X_val)

    calibrated_clfs = []
    for i in range(y_train.shape[1]):
        print(f"Training classifier {i+1}/{y_train.shape[1]} for label '{labels[i]}'")
        
        unique_classes = np.unique(y_train[:, i])
        if len(unique_classes) < 2:
            print(f"  Warning: Label '{labels[i]}' has only {len(unique_classes)} class(es) in training data. Skipping calibration.")
            base_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
            base_clf.fit(X_train_feat, y_train[:, i])
            calibrated_clfs.append(base_clf)
            continue
        
        base_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
        base_clf.fit(X_train_feat, y_train[:, i])
        
        unique_val_classes = np.unique(y_val[:, i])
        if len(unique_val_classes) < 2:
            print(f"  Warning: Label '{labels[i]}' has only {len(unique_val_classes)} class(es) in validation data. Skipping calibration.")
            calibrated_clfs.append(base_clf)
            continue
        
        try:
            calib = CalibratedClassifierCV(base_clf, cv='prefit', method='sigmoid')
            calib.fit(X_val_feat, y_val[:, i])
            calibrated_clfs.append(calib)
        except Exception as e:
            print(f"  Calibration failed for '{labels[i]}': {e}. Using base classifier.")
            calibrated_clfs.append(base_clf)

    model_bundle = {'tfidf': pipe.named_steps['tfidf'], 'calibrated_clfs': calibrated_clfs, 'labels': labels}

    y_prob = np.column_stack([c.predict_proba(X_val_feat)[:, 1] if hasattr(c, 'predict_proba') else c.decision_function(X_val_feat) for c in calibrated_clfs])
    report(y_val, y_prob, labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_bundle, output_path)


def report(y_true, y_score, labels, threshold=0.5):
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
    for lbl, ap in zip(labels, aps):
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

    X, y, labels = prepare_multilabel(df)
    
    y_sums = y.sum(axis=1)
    y_sums_str = [str(int(s)) for s in y_sums]
    
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y_sums_str
        )
    except ValueError:
        print("Stratification failed, using simple random split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

    if args.level == 1:
        train_level1(X_train, y_train, X_val, y_val, args.output, labels)
    else:
        train_level2(X_train, y_train, X_val, y_val, args.output, labels)


if __name__ == '__main__':
    main()
