from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from typing import List
from typing import Optional
import os
from .model import load_model, predict_emotion, recommend_from_emotion

app = FastAPI(title="NeuroNudge API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
MODEL = load_model()

DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        text TEXT,
        emotion TEXT,
        score REAL,
        rec_action TEXT,
        rec_msg TEXT,
        productivity REAL,
        valence REAL
    )''')
    conn.commit()
    conn.close()


def map_emotion_to_valence(emotion: str) -> float:
    m = (emotion or '').lower()
    mapping = {
        'motivated': 0.8,
        'joy': 0.9,
        'neutral': 0.0,
        'low_mood': -0.6,
        'sadness': -0.6,
        'anxiety': -0.7,
        'fatigue': -0.4,
        'overwhelmed': -0.8,
        'anger': -0.7,
        'fear': -0.7,
    }
    return float(mapping.get(m, 0.0))

class TextIn(BaseModel):
    text: str

class RecommendIn(BaseModel):
    emotion: Optional[str] = None
    text: Optional[str] = None


class LogIn(BaseModel):
    text: str
    productivity: Optional[float] = None


@app.on_event('startup')
def _startup_db():
    init_db()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict_text")
async def predict_text(payload: TextIn):
    if not payload.text:
        raise HTTPException(status_code=400, detail="text is required")
    label, score = predict_emotion(payload.text, MODEL)
    return {"emotion": label, "score": float(score)}

@app.post("/recommend")
async def recommend(payload: RecommendIn):
    if payload.emotion is None and not payload.text:
        raise HTTPException(status_code=400, detail="provide emotion or text")
    emotion = payload.emotion
    if emotion is None:
        emotion, _ = predict_emotion(payload.text, MODEL)
    rec = recommend_from_emotion(emotion)
    return {"emotion": emotion, "recommendation": rec}


@app.post('/log')
async def log_entry(payload: LogIn):
    if not payload.text:
        raise HTTPException(status_code=400, detail='text is required')
    emotion, score = predict_emotion(payload.text, MODEL)
    rec = recommend_from_emotion(emotion)
    valence = map_emotion_to_valence(emotion)
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO entries (timestamp, text, emotion, score, rec_action, rec_msg, productivity, valence)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (ts, payload.text, emotion, float(score), rec.get('action'), rec.get('message'), payload.productivity, valence))
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return {'id': rowid, 'timestamp': ts, 'emotion': emotion, 'score': float(score), 'recommendation': rec}


@app.get('/logs')
async def get_logs(limit: int = 200) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, text, emotion, score, rec_action, rec_msg, productivity, valence FROM entries ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows[::-1]:
        result.append({'id': r[0], 'timestamp': r[1], 'text': r[2], 'emotion': r[3], 'score': r[4], 'recommendation': {'action': r[5], 'message': r[6]}, 'productivity': r[7], 'valence': r[8]})
    return result


@app.post('/clear_logs')
async def clear_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('DELETE FROM entries')
        conn.commit()
        cleared = c.rowcount
    except Exception:
        conn.rollback()
        cleared = None
    conn.close()
    return {'cleared_rows': cleared}
