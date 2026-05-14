"""
predict_transformer.py
-----------------------
Inference module for the fine-tuned XLM-RoBERTa intent classifier.

Exposes a single public function:
    predict(text: str) -> tuple[str, float, np.ndarray, list[str]]

Returns the same signature as the old SVM predict() so app.py needs
only minimal changes.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("saved_models/xlm_roberta")
MAX_LEN = 128


@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    """Load model and tokenizer once; cached for the lifetime of the process."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"XLM-RoBERTa model not found at '{MODEL_DIR}'. "
            "Please run 'python train_transformer.py' first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


@lru_cache(maxsize=1)
def _load_label_encoder():
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError(f"label_encoder.pkl not found in '{MODEL_DIR}'.")
    with open(encoder_path, "rb") as f:
        return pickle.load(f)


def load_artifacts() -> dict:
    """
    Return a dict that matches the schema expected by app.py:
        {
            "model": None,          # not used directly
            "label_encoder": ...,
            "metrics": ...,
            "transformer": True,    # flag so app knows it's a transformer model
        }
    """
    label_encoder = _load_label_encoder()

    metrics_path = MODEL_DIR / "metrics.pkl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.pkl not found in '{MODEL_DIR}'.")
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    # Pre-warm the model so first prediction is fast
    _load_model_and_tokenizer()

    return {
        "model": None,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "transformer": True,
    }


def predict(text: str, artifacts: dict | None = None) -> tuple[str, float, np.ndarray, list[str]]:
    """
    Classify a customer-support message.

    Parameters
    ----------
    text : str
        Raw input text in any supported language.
    artifacts : dict, optional
        Loaded artifacts dict (ignored — model loaded via cache).

    Returns
    -------
    intent : str
        Predicted intent label.
    confidence : float
        Probability of the top intent (0–1).
    probabilities : np.ndarray
        Probability for every class (shape: [num_classes]).
    labels : list[str]
        Class names in the same order as `probabilities`.
    """
    model, tokenizer, device = _load_model_and_tokenizer()
    label_encoder = _load_label_encoder()
    labels = list(label_encoder.classes_)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # shape: [1, num_labels]

    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # shape: [num_labels]
    top_idx = int(np.argmax(probs))

    intent = labels[top_idx]
    confidence = float(probs[top_idx])

    return intent, confidence, probs, labels
