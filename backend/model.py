import os
import re
from typing import Tuple
import numpy as np

BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(BASE_MODELS_DIR, "emotion_pipe.pkl")
LEVEL2_PATH = os.path.join(BASE_MODELS_DIR, "level2_pipe.joblib")
TRANSFORMER_DIR = os.path.join(BASE_MODELS_DIR, "transformer_emotion")


def load_model():
    try:
        import joblib
        if os.path.isdir(TRANSFORMER_DIR) and os.path.exists(os.path.join(TRANSFORMER_DIR, 'config.json')):
            return {'type': 'transformer', 'path': TRANSFORMER_DIR}

        if os.path.exists(LEVEL2_PATH):
            bundle = joblib.load(LEVEL2_PATH)
            return {'type': 'level2', 'bundle': bundle}

        if os.path.exists(MODEL_PATH):
            pipe = joblib.load(MODEL_PATH)
            return {'type': 'sklearn', 'pipe': pipe}
    except Exception:
        pass
    return None


LEXICON = {
    'joy': ['happy', 'joy', 'glad', 'excited', 'delighted', 'pleased', 'good', 'great'],
    'sadness': ['sad', 'down', 'unhappy', 'depressed', 'mourn', 'sorrow', 'gloomy'],
    'anger': ['angry', 'mad', 'furious', 'irritat', 'annoyed', 'rage'],
    'fear': ['scared', 'afraid', 'anxious', 'nervous', 'worried', 'panic'],
    'surprise': ['surprised', 'astonished', 'amazed', 'shocked'],
    'neutral': []
}


def _simple_lexicon_predict(text: str) -> Tuple[str, float]:
    t = text.lower()
    counts = {k: 0 for k in LEXICON}
    for k, words in LEXICON.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", t):
                counts[k] += 1
    best = max(counts.items(), key=lambda x: x[1])
    label = best[0]
    if best[1] == 0:
        label = 'neutral'
        score = 0.5
    else:
        total = sum(counts.values())
        score = best[1] / total if total > 0 else 0.6
    return label, score


def predict_emotion(text: str, model=None) -> Tuple[str, float]:
    text = (text or '').strip()
    if not text:
        return 'neutral', 0.5

    if not model:
        return _simple_lexicon_predict(text)

    try:
        mtype = model.get('type') if isinstance(model, dict) else None
    except Exception:
        mtype = None

    if mtype == 'sklearn' and 'pipe' in model:
        try:
            pipe = model['pipe']
            preds = pipe.predict_proba([text])
            classes = pipe.classes_
            top_idx = preds[0].argmax()
            return classes[top_idx], float(preds[0][top_idx])
        except Exception:
            return _simple_lexicon_predict(text)

    if mtype == 'level2' and 'bundle' in model:
        try:
            bundle = model['bundle']
            tfidf = bundle.get('tfidf')
            clfs = bundle.get('calibrated_clfs')
            labels = bundle.get('labels')
            if tfidf is None or clfs is None:
                return _simple_lexicon_predict(text)
            X = tfidf.transform([text])
            probs = []
            for c in clfs:
                if hasattr(c, 'predict_proba'):
                    p = c.predict_proba(X)
                    probs.append(float(p[0][1]))
                else:
                    try:
                        df = c.decision_function(X)
                        p = 1.0 / (1.0 + pow(2.718281828, -float(df[0])))
                        probs.append(p)
                    except Exception:
                        probs.append(0.0)
            probs = np.array(probs)
            top_idx = int(np.argmax(probs))
            return labels[top_idx], float(probs[top_idx])
        except Exception:
            return _simple_lexicon_predict(text)

    if mtype == 'transformer' and 'path' in model:
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline('text-classification', model=model['path'], return_all_scores=True)
            out = pipe(text)
            scores = out[0]
            best = max(scores, key=lambda x: x['score'])
            return best['label'], float(best['score'])
        except Exception:
            return _simple_lexicon_predict(text)

    return _simple_lexicon_predict(text)


def recommend_from_emotion(emotion: str) -> dict:
    mapping = {
        'fatigue': {'action': 'micro-break', 'message': 'Take a 5–10 minute micro-break: stand, stretch, hydrate.'},
        'anxiety': {'action': 'calm', 'message': 'Use grounding techniques, deep breathing, or calming music.'},
        'low_mood': {'action': 'self-care', 'message': 'Try a brief journaling exercise or a short walk.'},
        'motivated': {'action': 'focus-high', 'message': 'You seem motivated — schedule 60–90 minute focused work blocks.'},
        'joy': {'action': 'focus-high', 'message': 'You seem motivated — schedule 60–90 minute focused work blocks.'},
        'neutral': {'action': 'maintain', 'message': 'You seem steady — continue with planned tasks.'},
        'sadness': {'action': 'self-care', 'message': 'Try a brief journaling exercise or a short walk.'},
        'anger': {'action': 'pause', 'message': 'Take 5 deep breaths or a short calming break.'},
        'fear': {'action': 'calm', 'message': 'Use grounding techniques or calming music.'},
        'surprise': {'action': 're-evaluate', 'message': 'Pause and re-assess priorities given this new information.'}
    }
    return mapping.get(emotion, {'action': 'suggest', 'message': 'Try a micro-break or journaling prompt.'})
