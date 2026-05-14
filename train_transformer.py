"""
train_transformer.py
---------------------
Fine-tunes xlm-roberta-base on the multilingual customer-support intent dataset.

Usage:
    python train_transformer.py

Outputs (saved to saved_models/xlm_roberta/):
    - HuggingFace model & tokenizer
    - label_encoder.pkl
    - metrics.pkl
    - visuals/confusion_matrix.png  (and other charts)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data.dataset import generate_dataset

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/customer_support_data.csv")
MODEL_DIR = Path("saved_models/xlm_roberta")
VISUAL_DIR = Path("visuals")
BASE_MODEL = "xlm-roberta-base"

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


# ── Dataset wrapper ────────────────────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Metric function for Trainer ────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


# ── Plotting helpers ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, accuracy, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (Accuracy={accuracy:.2%})")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_intent_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    df["intent"].value_counts().sort_index().plot(kind="bar", color="#2563eb", ax=ax)
    ax.set_title("Intent Distribution")
    ax.set_xlabel("Intent")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "intent_distribution.png", dpi=150)
    plt.close(fig)


def plot_language_accuracy(test_df: pd.DataFrame):
    lang_acc = test_df.groupby("language")["correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(lang_acc.index, lang_acc.values, color="#14b8a6")
    for bar, value in zip(bars, lang_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                f"{value:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Language-wise Accuracy")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "language_accuracy.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(scores: dict[str, float]):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2563eb"]
    bars = ax.bar(list(scores.keys()), list(scores.values()), color=colors, width=0.4)
    for bar, score in zip(bars, scores.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                f"{score:.2%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


# ── Main training routine ──────────────────────────────────────────────────────
def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device.upper()}")
    if device == "cpu":
        print("[WARN] No GPU detected — training on CPU. This may take 45–90 minutes.")

    # ── Load / generate dataset ────────────────────────────────────────────────
    df = generate_dataset(DATA_PATH)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    num_labels = len(label_encoder.classes_)
    label_names = list(label_encoder.classes_)

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text", "language"]],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_texts = X_train["text"].tolist()
    test_texts = X_test["text"].tolist()
    train_labels = y_train.tolist()
    test_labels = y_test.tolist()

    # ── Tokenizer & model ──────────────────────────────────────────────────────
    print(f"[INFO] Loading tokenizer from '{BASE_MODEL}' …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"[INFO] Loading model from '{BASE_MODEL}' …")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label={i: lbl for i, lbl in enumerate(label_names)},
        label2id={lbl: i for i, lbl in enumerate(label_names)},
    )

    # ── Tokenize ───────────────────────────────────────────────────────────────
    print("[INFO] Tokenizing dataset …")
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    print("[INFO] Starting fine-tuning …")
    trainer.train()

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("[INFO] Evaluating …")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    pred_labels = np.argmax(logits, axis=-1)
    accuracy = float(accuracy_score(test_labels, pred_labels))

    report = classification_report(
        test_labels, pred_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # ── Language-wise accuracy ─────────────────────────────────────────────────
    test_df = X_test.copy()
    test_df["actual"] = test_labels
    test_df["predicted"] = pred_labels
    test_df["correct"] = test_df["actual"] == test_df["predicted"]

    # ── Save model & tokenizer ─────────────────────────────────────────────────
    print(f"[INFO] Saving model to '{MODEL_DIR}' …")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # ── Save label encoder ─────────────────────────────────────────────────────
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # ── Build & save metrics (same schema as old train.py) ────────────────────
    metrics = {
        "accuracy": accuracy,
        "cv_mean": accuracy,          # no CV for transformers; use hold-out acc
        "cv_std": 0.0,
        "model_name": "XLM-RoBERTa Base",
        "best_by_holdout": "XLM-RoBERTa Base",
        "selection_strategy": "Fine-tuned xlm-roberta-base with HuggingFace Trainer",
        "classification_report": report,
        "model_scores": {"XLM-RoBERTa Base": accuracy},
        "labels": label_names,
        "sample_count": int(len(df)),
        "language_count": int(df["language"].nunique()),
        "intent_count": int(df["intent"].nunique()),
    }
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # ── Visuals ────────────────────────────────────────────────────────────────
    plot_confusion_matrix(test_labels, pred_labels, label_names, accuracy, "XLM-RoBERTa Base")
    plot_intent_distribution(df)
    plot_language_accuracy(test_df)
    plot_model_comparison({"XLM-RoBERTa Base": accuracy})

    print("\n[OK] Training complete!")
    print(f"   Accuracy : {accuracy:.4f} ({accuracy:.2%})")
    print(f"   Saved to : {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    train()
