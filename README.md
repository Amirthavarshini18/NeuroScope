# NeuroNudge — AI System for Emotionally Adaptive Productivity

NeuroNudge is a demo project that analyzes text to infer emotional state and returns personalized productivity recommendations.

Repository layout
- `backend/` — FastAPI application, model loader, training and fine-tuning scripts
- `frontend/` — single-page UI (HTML/CSS/JS)
- `backend/models/` — saved model artifacts (created by training scripts)

Prediction targets
- Primary labels (multi-label): `fatigue`, `anxiety`, `low_mood`, `motivated`, `overwhelmed`.
- Optional regressions (not implemented by default): valence ∈ [-1,1], arousal ∈ [0,1].

Quick setup
1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Train a TF-IDF baseline or Level-2 model:

```powershell
python backend/train_models.py --level 2 --output backend/models/level2_pipe.joblib
```

3. (Optional) Fine-tune a transformer (longer, GPU recommended):

```powershell
python backend/finetune.py --model_name distilbert-base-uncased --output_dir backend/models/transformer_emotion --epochs 3
```

Run the backend and frontend locally
1. Start the backend (keep this terminal open):

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8002
```

2. Serve the frontend (in a separate terminal):

```powershell
python -m http.server 5500 --directory frontend
```

Open the UI at: `http://127.0.0.1:5500`. If you started the backend on a different port, open the frontend with an API override, for example:
`http://127.0.0.1:5500/index.html?api=http://127.0.0.1:8010`

Files of interest
- `backend/main.py` — FastAPI endpoints: `/health`, `/predict_text`, `/recommend`
- `backend/model.py` — model loader and lexicon fallback
- `backend/train_models.py` — Level 1 and Level 2 TF-IDF trainers (multi-label)
- `backend/finetune.py` — Transformer fine-tuning script
- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` — frontend UI files

Notes
- The repo includes a lexicon fallback so the API and frontend work without a trained model.
- The Level-2 trainer saves a joblib bundle to `backend/models/level2_pipe.joblib` that contains a TF-IDF transformer and per-class calibrated classifiers.
- Improve production performance by training a dedicated policy model and by creating a small, hand-labeled dataset for classes with low coverage.

License: MIT

What we predict (design summary)
---------------------------------
- Primary labels (multi-label): `fatigue`, `anxiety`, `low_mood`, `motivated`, `overwhelmed`.
- Optional regressions: valence ∈ [-1,1], arousal ∈ [0,1] (not implemented by default but planned).

Level-based model guidance included in this repo:
- Level 1 (tiny data ≤2k): TF-IDF + Multinomial Naive Bayes (fast baseline). Script: `backend/train_models.py --level 1`.
- Level 2 (2k–50k): TF-IDF + OneVsRest Logistic Regression with per-class calibration. Script: `backend/train_models.py --level 2`.
- Level 3 (≥10k): Transformer / embeddings based fine-tuning (see `backend/finetune.py` and notes in README).

Hyperparameter tips included in scripts and comments. The `backend/train_models.py` script implements mapping heuristics from HF datasets into the five primary labels, trains the TF-IDF pipelines, calibrates probabilities (Level 2), and reports macro/micro F1 and per-class AP.

Policy (mapping emotions -> nudges)
-----------------------------------
The repo contains `backend/policy.py` which provides a rule-based nudge mapper and simple personalization helpers (user moving-average of embeddings, blending with global logits). The recommended production approach is to train a policy model (LightGBM or Logistic Regression) using logged user responses and labels such as `{micro_break, focus_sprint_25, reprioritize_high, breathing_3min, journaling_prompt, soothing_music}`.

License: MIT
