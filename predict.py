"""
Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys
import pickle
from pathlib import Path

import numpy as np

from features import extract_features

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_pipe = None


def _load_model():
    global _pipe
    if _pipe is None:
        with open(_MODEL_PATH, "rb") as f:
            _pipe = pickle.load(f)
    return _pipe


def predict(image_path: str) -> float:
    pipe = _load_model()
    feats = np.array([extract_features(image_path)])
    score = pipe.predict_proba(feats)[0][1]   # P(screen)
    return float(score)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py some_image.jpg")

    import time
    t0 = time.perf_counter()
    result = predict(sys.argv[1])
    ms = (time.perf_counter() - t0) * 1000

    print(f"{result:.4f}")
    print(f"# latency: {ms:.1f} ms", file=sys.stderr)