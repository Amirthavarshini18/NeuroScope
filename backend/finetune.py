import argparse
import os
from collections import Counter
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def collect_examples():
    rows = []
    try:
        go = load_dataset('debarshichanda/goemotions')
    except Exception:
        try:
            go = load_dataset('go_emotions')
        except Exception:
            go = None

    if go:
        try:
            go_label_names = go['train'].features['labels'].feature.names
        except Exception:
            go_label_names = None
        for split in go.keys():
            for ex in go[split]:
                text = ex.get('text') or ex.get('content') or ex.get('sentence') or ''
                label = ex.get('labels') or ex.get('label')
                if isinstance(label, list):
                    idx = None
                    for i, v in enumerate(label):
                        if v:
                            idx = i
                            break
                    if idx is None:
                        continue
                    if go_label_names:
                        label_name = go_label_names[idx]
                    else:
                        label_name = str(idx)
                else:
                    if go_label_names and label is not None:
                        label_name = go_label_names[label]
                    else:
                        label_name = str(label)
                rows.append({'text': text, 'label': label_name})

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
                label = ex.get('emotion') or ex.get('label') or 'neutral'
                rows.append({'text': text, 'label': str(label)})

    df = pd.DataFrame(rows)
    df = df[df['text'].str.len() > 0]
    return df


def prepare_dataset(df, tokenizer, max_length=128):
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    dataset = Dataset.from_dict({
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'label': labels,
    })
    return dataset


def compute_metrics_fn(metric_name, label2id):
    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy.compute(predictions=preds, references=labels)['accuracy'],
            'f1_macro': f1.compute(predictions=preds, references=labels, average='macro')['f1'],
        }

    return compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='distilbert-base-uncased')
    p.add_argument('--output_dir', default=os.path.join(os.path.dirname(__file__), 'models', 'transformer_emotion'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--per_device_train_batch_size', type=int, default=8)
    p.add_argument('--per_device_eval_batch_size', type=int, default=16)
    p.add_argument('--max_length', type=int, default=128)
    args = p.parse_args()

    df = collect_examples()
    if df.empty:
        return

    label_counts = Counter(df['label'].tolist())
    labels = [l for l, _ in label_counts.most_common()]
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    df['label_id'] = df['label'].map(label2id)

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.08, random_state=42, stratify=df['label_id'])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = prepare_dataset(train_df.rename(columns={'label_id': 'label'})[['text', 'label']], tokenizer, max_length=args.max_length)
    val_dataset = prepare_dataset(val_df.rename(columns={'label_id': 'label'})[['text', 'label']], tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        save_total_limit=2,
    )

    compute_metrics = compute_metrics_fn('accuracy', label2id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
