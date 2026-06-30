import glob
import pickle
import sys

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features import extract_features


def load_paths_and_labels(real_dir="real", screen_dir="screen"):
    paths, labels = [], []

    real_files = sorted(glob.glob(f"{real_dir}/*"))
    screen_files = sorted(glob.glob(f"{screen_dir}/*"))

    for p in real_files:
        paths.append(p)
        labels.append(0)

    for p in screen_files:
        paths.append(p)
        labels.append(1)

    if len(real_files) == 0 or len(screen_files) == 0:
        sys.exit(
            f"Found {len(real_files)} images in '{real_dir}/' and "
            f"{len(screen_files)} in '{screen_dir}/'. Both folders need images. "
            f"Make sure you're running this from the project root."
        )

    return paths, labels


def main():
    paths, labels = load_paths_and_labels()
    print(f"Loaded {len(paths)} images "
          f"({labels.count(0)} real, {labels.count(1)} screen)")

    feats = []
    for i, p in enumerate(paths, 1):
        try:
            feats.append(extract_features(p))
        except Exception as e:
            sys.exit(f"Failed on {p}: {e}")
        if i % 20 == 0 or i == len(paths):
            print(f"  extracted {i}/{len(paths)}")

    X = np.array(feats)
    y = np.array(labels)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    pipe.fit(X, y)

    coefs = pipe.named_steps["clf"].coef_[0]
    names = ["fft_peak", "laplacian_var", "sat_skew", "sat_p95",
              "lbp_entropy", "dct_artifact", "highlight_ratio"]
    print("\nFeature weights (higher = stronger screen signal):")
    for n, c in sorted(zip(names, coefs), key=lambda x: -abs(x[1])):
        print(f"  {n:18s} {c:+.3f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("\nModel saved to model.pkl")


if __name__ == "__main__":
    main()