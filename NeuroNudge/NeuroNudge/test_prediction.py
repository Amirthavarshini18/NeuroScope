import sys
sys.path.insert(0, '.')
from backend.model import load_model, predict_emotion
import numpy as np

m = load_model()
print(f'Model type: {m.get("type") if m else None}')
print()

tests = [
    "I miss my old friends. Everyone moved away after college and I feel so isolated. Nothing feels the same anymore.",
    "The project deadline got moved up again without any warning. Management keeps changing requirements and expects us to just deal with it. This is completely unfair.",
    "I am so happy today!",
]

for text in tests:
    if m and m.get('type') == 'level2':
        bundle = m['bundle']
        tfidf = bundle['tfidf']
        clfs = bundle['calibrated_clfs']
        labels = bundle['labels']
        
        X = tfidf.transform([text])
        probs = []
        for c in clfs:
            if hasattr(c, 'predict_proba'):
                p = c.predict_proba(X)
                probs.append(float(p[0][1]))
            else:
                probs.append(0.0)
        
        print(f'\nText: {text[:70]}...')
        print('Probabilities:')
        for label, prob in zip(labels, probs):
            print(f'  {label:<12} {prob:.4f}')
        
        result = predict_emotion(text, m)
        print(f'Final prediction: {result[0]} ({result[1]:.4f})')


