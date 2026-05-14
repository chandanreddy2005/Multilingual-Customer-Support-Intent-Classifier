from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from data.dataset import generate_dataset
from preprocess import preprocess


DATA_PATH = Path("data/customer_support_data.csv")
MODEL_DIR = Path("saved_models")
VISUAL_DIR = Path("visuals")


def build_features(train_text, test_text):
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=50000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        max_features=50000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
    )

    train_word = word_vectorizer.fit_transform(train_text)
    test_word = word_vectorizer.transform(test_text)
    train_char = char_vectorizer.fit_transform(train_text)
    test_char = char_vectorizer.transform(test_text)

    return (
        hstack([train_word, train_char]),
        hstack([test_word, test_char]),
        word_vectorizer,
        char_vectorizer,
        train_word,
        test_word,
    )


def plot_confusion_matrix(y_true, y_pred, labels, accuracy, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name} (Accuracy={accuracy:.2%})")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_intent_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df["intent"].value_counts().sort_index().plot(kind="bar", color="#2563eb", ax=ax)
    ax.set_title("Intent Distribution")
    ax.set_xlabel("Intent")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "intent_distribution.png", dpi=150)
    plt.close(fig)


def plot_language_accuracy(test_df):
    lang_acc = test_df.groupby("language")["correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(lang_acc.index, lang_acc.values, color="#14b8a6")
    for bar, value in zip(bars, lang_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Language-wise Accuracy")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "language_accuracy.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(scores):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#94a3b8", "#22c55e", "#2563eb"]
    bars = ax.bar(list(scores.keys()), list(scores.values()), color=colors, width=0.55)
    for bar, score in zip(bars, scores.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.2%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison")
    plt.tight_layout()
    fig.savefig(VISUAL_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    VISUAL_DIR.mkdir(exist_ok=True)

    df = generate_dataset(DATA_PATH)
    df["clean"] = df["text"].apply(preprocess)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])

    X_train, X_test, y_train, y_test = train_test_split(
        df[["clean", "language", "text"]],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    X_train_feat, X_test_feat, word_vectorizer, char_vectorizer, train_word, test_word = build_features(
        X_train["clean"], X_test["clean"]
    )

    svc = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000, dual="auto", random_state=42), cv=5)
    logistic = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", n_jobs=-1, random_state=42, multi_class="multinomial")
    baseline = MultinomialNB(alpha=0.1)

    svc.fit(X_train_feat, y_train)
    logistic.fit(X_train_feat, y_train)
    baseline.fit(train_word, y_train)

    candidates = {
        "Linear SVM": (svc, svc.predict(X_test_feat)),
        "Logistic Regression": (logistic, logistic.predict(X_test_feat)),
        "Naive Bayes": (baseline, baseline.predict(test_word)),
    }
    scores = {name: accuracy_score(y_test, preds) for name, (_, preds) in candidates.items()}

    production_name = "Linear SVM"
    production_model, production_preds = candidates[production_name]
    production_acc = scores[production_name]
    best_name = max(scores, key=scores.get)

    full_features = hstack([word_vectorizer.transform(df["clean"]), char_vectorizer.transform(df["clean"])])
    cv_model = production_model
    cv_scores = cross_val_score(
        cv_model,
        full_features,
        df["label"],
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )

    labels = list(label_encoder.classes_)
    report = classification_report(y_test, production_preds, target_names=labels, output_dict=True, zero_division=0)

    test_df = X_test.copy()
    test_df["actual"] = y_test
    test_df["predicted"] = production_preds
    test_df["correct"] = test_df["actual"] == test_df["predicted"]

    plot_confusion_matrix(y_test, production_preds, labels, production_acc, production_name)
    plot_intent_distribution(df)
    plot_language_accuracy(test_df)
    plot_model_comparison(
        {
            "Naive Bayes": scores["Naive Bayes"],
            "Logistic Regression": scores["Logistic Regression"],
            "Linear SVM": scores["Linear SVM"],
        }
    )

    metrics = {
        "accuracy": float(production_acc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "model_name": production_name,
        "best_by_holdout": best_name,
        "selection_strategy": "Hybrid word+character TF-IDF with calibrated Linear SVM",
        "classification_report": report,
        "model_scores": {name: float(score) for name, score in scores.items()},
        "labels": labels,
        "sample_count": int(len(df)),
        "language_count": int(df["language"].nunique()),
        "intent_count": int(df["intent"].nunique()),
    }

    bundle = {
        "model": production_model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "preprocess_version": "1.0",
    }

    with open(MODEL_DIR / "model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(MODEL_DIR / "word_tfidf.pkl", "wb") as f:
        pickle.dump(word_vectorizer, f)
    with open(MODEL_DIR / "char_tfidf.pkl", "wb") as f:
        pickle.dump(char_vectorizer, f)
    with open(MODEL_DIR / "nb_model.pkl", "wb") as f:
        pickle.dump(production_model, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(word_vectorizer, f)

    print(f"Dataset: {len(df)} samples")
    print(f"Production model: {production_name} | Accuracy: {production_acc:.4f}")
    print(f"Best holdout score: {best_name} | Accuracy: {scores[best_name]:.4f}")
    print(f"5-fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print("Saved model artifacts and visual reports.")


if __name__ == "__main__":
    train()
