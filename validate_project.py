from __future__ import annotations

import pickle
from pathlib import Path


REQUIRED_FILES = [
    "app.py",
    "train.py",
    "preprocess.py",
    "data/dataset.py",
    "requirements.txt",
    "README.md",
]

REQUIRED_MODEL_FILES = [
    "saved_models/model_bundle.pkl",
    "saved_models/word_tfidf.pkl",
    "saved_models/char_tfidf.pkl",
    "saved_models/nb_model.pkl",
    "saved_models/label_encoder.pkl",
    "saved_models/metrics.pkl",
]

REQUIRED_VISUALS = [
    "visuals/confusion_matrix.png",
    "visuals/intent_distribution.png",
    "visuals/language_accuracy.png",
    "visuals/model_comparison.png",
]


def check_paths(paths):
    missing = [path for path in paths if not Path(path).exists()]
    return missing


def main():
    print("Checking project files...")
    missing_files = check_paths(REQUIRED_FILES)
    if missing_files:
        print("Missing required files:")
        for path in missing_files:
            print(f"  - {path}")
        raise SystemExit(1)
    print("Core files: OK")

    missing_models = check_paths(REQUIRED_MODEL_FILES)
    missing_visuals = check_paths(REQUIRED_VISUALS)
    if missing_models or missing_visuals:
        print("Training output is incomplete. Run: python train.py")
        for path in missing_models + missing_visuals:
            print(f"  - missing {path}")
        raise SystemExit(1)

    with open("saved_models/model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    expected_keys = {"model", "word_vectorizer", "char_vectorizer", "label_encoder", "metrics"}
    missing_keys = expected_keys.difference(bundle)
    if missing_keys:
        print(f"Model bundle is missing keys: {sorted(missing_keys)}")
        raise SystemExit(1)

    metrics = bundle["metrics"]
    labels = list(bundle["label_encoder"].classes_)
    print(f"Model: {metrics.get('model_name', 'unknown')}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"CV accuracy: {metrics.get('cv_mean', 0):.2%}")
    print(f"Labels: {', '.join(labels)}")
    print("Project validation: OK")


if __name__ == "__main__":
    main()
