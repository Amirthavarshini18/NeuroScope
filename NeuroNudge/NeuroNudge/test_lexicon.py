import sys
sys.path.insert(0, '.')
from backend.model import _simple_lexicon_predict

texts = [
    "I miss my old friends. Everyone moved away after college and I feel so isolated. Nothing feels the same anymore.",
    "The project deadline got moved up again without any warning. Management keeps changing requirements and expects us to just deal with it. This is completely unfair.",
]

print("Lexicon predictions:")
for text in texts:
    result = _simple_lexicon_predict(text)
    print(f'{result[0]:<12} ({result[1]:.3f}) | {text[:60]}...')
