from typing import Dict, Any
import numpy as np

DEFAULT_POLICY = {
    'fatigue': {'action': 'micro_break', 'message': 'Take a 5–10 minute micro-break: stand, stretch, hydrate.'},
    'anxiety': {'action': 'breathing_3min', 'message': 'Try 3 minutes of paced breathing or calming music.'},
    'low_mood': {'action': 'journaling_prompt', 'message': 'Try a short gratitude journaling prompt: 3 things you did well today.'},
    'motivated': {'action': 'focus_sprint_25', 'message': 'You look motivated — start a 25–50 minute focused sprint.'},
    'overwhelmed': {'action': 'reprioritize_high', 'message': 'Re-evaluate tasks; pick one small priority and start there.'}
}


def rule_map(emotion_probs: Dict[str, float], uncertainty: float = 0.0, context: Dict[str, Any] = None) -> Dict[str, Any]:
    if context is None:
        context = {}

    if uncertainty > 0.6:
        return {'action': 'journaling_prompt', 'message': 'Uncertain — try a short journaling prompt or a micro-break.'}

    if not emotion_probs:
        return DEFAULT_POLICY['low_mood']
    top = max(emotion_probs.items(), key=lambda x: x[1])
    label, prob = top
    rec = DEFAULT_POLICY.get(label, {'action': 'micro_break', 'message': 'Try a short break or journaling.'})

    tod = context.get('time_of_day')
    if rec['action'] == 'focus_sprint_25' and tod is not None and (tod < 8 or tod > 20):
        return {'action': 'journaling_prompt', 'message': 'Late hour — consider journaling or a short wind-down instead of a sprint.'}

    return rec


def blend_user_global(p_global: np.ndarray, p_user: np.ndarray, lam: float = 0.2) -> np.ndarray:
    return (1 - lam) * p_global + lam * p_user


def user_moving_average(existing: np.ndarray, new_embedding: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    if existing is None:
        return new_embedding
    return (1 - alpha) * existing + alpha * new_embedding
