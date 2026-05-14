from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MODEL_DIR = Path("saved_models")
DATA_PATH = Path("data/customer_support_data.csv")


st.set_page_config(
    page_title="Multilingual Intent Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            color: #1d4ed8;
            font-size: 2.35rem;
            font-weight: 750;
            letter-spacing: 0;
            text-align: center;
            margin: 0 0 .25rem;
        }
        .sub-title {
            color: #475569;
            font-size: 1.05rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .result-box {
            border-left: 5px solid #2563eb;
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            min-height: 118px;
            color: #000000;
        }
        .result-box-winner {
            border-left: 5px solid #16a34a;
            background: #f0fdf4;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            min-height: 118px;
            color: #000000;
        }
        .result-box h3,
        .result-box b,
        .result-box-winner h3,
        .result-box-winner b {
            color: #000000 !important;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-card h2 {
            margin: 0;
            color: #0f172a;
        }
        .metric-card p {
            margin: .25rem 0 0;
            color: #64748b;
        }
        .intent-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
            font-size: 0.92rem;
        }
        .intent-label {
            min-width: 155px;
            font-weight: 500;
            color: #1e293b;
        }
        .intent-bar-bg {
            flex: 1;
            background: #e2e8f0;
            border-radius: 999px;
            height: 16px;
            position: relative;
            overflow: hidden;
        }
        .intent-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            transition: width 0.4s ease;
        }
        .intent-bar-fill-top {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #16a34a);
        }
        .intent-pct {
            min-width: 46px;
            text-align: right;
            font-weight: 700;
            color: #0f172a;
            font-size: 0.92rem;
        }
        .tip-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            font-size: 0.88rem;
            color: #1e3a8a;
            margin-bottom: 0.75rem;
        }
        .preprocess-box {
            background: #f1f5f9;
            border-left: 4px solid #94a3b8;
            border-radius: 0 6px 6px 0;
            padding: 0.5rem 0.85rem;
            font-family: monospace;
            font-size: 0.82rem;
            color: #475569;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_training():
    result = subprocess.run([sys.executable, "train_transformer.py"], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


@st.cache_resource
def load_artifacts():
    import predict_transformer
    return predict_transformer.load_artifacts()


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=["text", "intent", "language"])


def detect_language(text: str) -> str:
    try:
        from langdetect import detect

        code = detect(text)
        lang_map = {
            "en": "English",
            "hi": "Hindi",
            "kn": "Kannada",
            "te": "Telugu",
            "ta": "Tamil",
        }
        return lang_map.get(code, code.upper())
    except Exception:
        return "Unknown"


def predict(text: str, artifacts):
    import predict_transformer
    return predict_transformer.predict(text, artifacts)


def show_missing_model_state():
    st.warning("Model artifacts were not found. Train the model before using the app.")
    if st.button("Train model now", use_container_width=True):
        with st.spinner("Training the classifier..."):
            code, stdout, stderr = run_training()
        if code == 0:
            st.success("Training complete. Refresh the app to load the new model.")
            with st.expander("Training log"):
                st.code(stdout)
        else:
            st.error("Training failed.")
            st.code(stderr or stdout)


try:
    artifacts = load_artifacts()
except Exception as exc:
    artifacts = None
    load_error = exc
else:
    load_error = None


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Intent", "Analytics"])
st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Model**")
st.sidebar.success("XLM-RoBERTa Base")
st.sidebar.markdown("**Supported Languages & Scripts**")
for language in ["English (Latin)", "Hindi (Devanagari/Roman)", "Kannada (Native/Roman)", "Telugu (Native/Roman)", "Tamil (Native/Roman)"]:
    st.sidebar.markdown(f"- {language}")

if artifacts is None:
    st.markdown('<p class="main-title">Multilingual Customer Support Intent Classifier</p>', unsafe_allow_html=True)
    show_missing_model_state()
    st.caption(str(load_error))
    st.stop()

metrics = artifacts["metrics"]

if page == "Home":
    st.markdown('<p class="main-title">Multilingual Customer Support Intent Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Intent detection for customer support messages across five languages</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2>7</h2><p>Intent Classes</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>5</h2><p>Languages</p></div>', unsafe_allow_html=True)
    with col3:
        cv_score = metrics.get("cv_mean", metrics.get("accuracy", 0.0))
        model_name = metrics.get("model_name", "Classifier")
        st.markdown(
            f'<div class="metric-card"><h2>{cv_score:.0%}</h2><p>CV Accuracy ({model_name})</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Supported Intents")
    intents = [
        "Billing Dispute",
        "Refund Request",
        "Delivery Inquiry",
        "Account Recovery",
        "Product Defect",
        "Technical Support",
        "Payment Failure",
    ]
    cols = st.columns(4)
    for index, intent in enumerate(intents):
        cols[index % 4].success(intent)

    st.markdown("---")
    st.subheader("Workflow")
    step1, step2, step3, step4 = st.columns(4)
    step1.info("Enter a customer support message")
    step2.info("Detect the likely language")
    step3.info("Preprocess and vectorize text")
    step4.info("Predict intent with confidence")

elif page == "Predict Intent":
    st.markdown('<p class="main-title">Predict Customer Intent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Describe your issue in any supported language — the model detects intent automatically</p>',
        unsafe_allow_html=True,
    )

    # ── Keyword hints in sidebar ────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Keyword Hints")
    hint_map = {
        "💳 Billing Dispute": "charged, billed, deducted, extra charge, wrong amount",
        "💰 Refund Request": "refund, money back, return, reimburse, cashback",
        "📦 Delivery Inquiry": "order, delivery, shipped, tracking, arrived, parcel",
        "🔑 Account Recovery": "password, login, forgot, locked, account access",
        "🔧 Product Defect": "broken, defective, damaged, not working, faulty",
        "🖥️ Technical Support": "app crash, error, bug, not loading, screen blank",
        "❌ Payment Failure": "payment failed, transaction declined, not processed",
    }
    for intent_name, keywords in hint_map.items():
        st.sidebar.markdown(f"**{intent_name}**")
        st.sidebar.caption(keywords)

    # ── Example picker ──────────────────────────────────────────────────────
    example_map = {
        "Select an example …": "",
        "🇬🇧 English — Payment failure": "My payment failed during checkout and money was deducted",
        "🇬🇧 English — Refund": "I want a full refund for my order",
        "🇬🇧 English — Delivery": "Where is my package? It hasn't arrived yet",
        "🇬🇧 English — Account": "I forgot my password and cannot log in",
        "🇬🇧 English — Defect": "The product I received is broken",
        "🇮🇳 Hindi — Refund": "mujhe refund chahiye mera paisa wapas karo",
        "🇮🇳 Hindi — Delivery": "mera order kahan hai abhi tak nahi aaya",
        "🇮🇳 Kannada — Order": "nanna order ellide yaavaga baratte",
        "🇮🇳 Telugu — Login": "account login avvatledu password marchipoyanu",
        "🇮🇳 Tamil — Defect": "product broken ah vandhathu replace pannum",
    }
    choice = st.selectbox("Try an example", list(example_map.keys()), key="example_picker")

    st.markdown(
        '<div class="tip-box">'
        "✏️ <b>Tip:</b> Be as specific as possible. Mention keywords like "
        "<i>refund, payment, tracking, password, broken, crash</i> so the model can "
        "detect your intent more accurately. Works in English, Hindi, Kannada, Telugu & Tamil."
        "</div>",
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "Enter your customer support message",
        value=example_map[choice],
        height=130,
        placeholder="e.g. 'I was charged twice for my order' or 'mujhe refund chahiye' …",
        key="user_input_field",
    )

    # Live character count
    if user_input.strip():
        word_count = len(user_input.strip().split())
        st.caption(f"📝 {word_count} word{'s' if word_count != 1 else ''} detected")

    classify_btn = st.button("🔍 Classify Intent", use_container_width=True, type="primary")

    if classify_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a message before classifying.")
        else:
            with st.spinner("Analyzing your message with XLM-RoBERTa …"):
                intent, confidence, probabilities, labels = predict(user_input, artifacts)
                language = detect_language(user_input)

            # ── Result cards ────────────────────────────────────────────────
            st.markdown("### 🎯 Prediction Result")
            result_cols = st.columns(3)
            with result_cols[0]:
                st.markdown(
                    f'<div class="result-box-winner"><b>✅ Predicted Intent</b><br><h3>{intent}</h3></div>',
                    unsafe_allow_html=True,
                )
            with result_cols[1]:
                conf_color = "#000000"
                st.markdown(
                    f'<div class="result-box"><b>📊 Top Confidence</b><br>'
                    f'<h3 style="color:{conf_color}">{confidence:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )
            with result_cols[2]:
                st.markdown(
                    f'<div class="result-box"><b>🌐 Detected Language</b><br><h3>{language}</h3></div>',
                    unsafe_allow_html=True,
                )

            # ── Model transparency note ──────────────────────────────────────
            with st.expander("🤖 About this prediction", expanded=False):
                st.markdown(
                    "**XLM-RoBERTa Base** processes the raw text directly using multilingual subword "
                    "tokenization (SentencePiece). No manual preprocessing is needed — the model "
                    "natively understands Hindi, Kannada, Telugu, Tamil, and English scripts."
                )
                st.caption(f"Input tokens: {len(user_input.split())} words · Model: xlm-roberta-base (125M params)")

            st.markdown("---")
            st.markdown("### 📊 Confidence per Intent")

            # Sort descending for display
            prob_pairs = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)

            chart_col, table_col = st.columns([3, 2])

            # ── Matplotlib bar chart with % annotations ─────────────────────
            with chart_col:
                sorted_labels = [p[0] for p in reversed(prob_pairs)]  # ascending for barh
                sorted_probs  = [p[1] for p in reversed(prob_pairs)]
                colors = [
                    "#16a34a" if lbl == intent else "#3b82f6"
                    for lbl in sorted_labels
                ]
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.barh(sorted_labels, sorted_probs, color=colors, height=0.55, edgecolor="none")

                # Annotate each bar with the percentage
                for bar, prob in zip(bars, sorted_probs):
                    pct_text = f"{prob:.1%}"
                    x_pos = bar.get_width() + 0.01
                    ax.text(
                        x_pos,
                        bar.get_y() + bar.get_height() / 2,
                        pct_text,
                        va="center",
                        ha="left",
                        fontsize=9,
                        fontweight="bold",
                        color="#0f172a",
                    )

                ax.set_xlim(0, 1.18)
                ax.set_xlabel("Confidence", fontsize=9)
                ax.set_title("Intent Probability Distribution", fontsize=11, fontweight="bold")
                ax.tick_params(axis="y", labelsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # ── HTML progress-bar table ──────────────────────────────────────
            with table_col:
                st.markdown("**Ranked Confidence Scores**")
                rows_html = ""
                for rank, (lbl, prob) in enumerate(prob_pairs, 1):
                    pct = prob * 100
                    fill_class = "intent-bar-fill-top" if lbl == intent else "intent-bar-fill"
                    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank}."))
                    rows_html += (
                        f'<div class="intent-row">'
                        f'<span style="min-width:22px;text-align:center">{medal}</span>'
                        f'<span class="intent-label">{lbl}</span>'
                        f'<span class="intent-bar-bg">'
                        f'<div class="{fill_class}" style="width:{pct:.1f}%"></div>'
                        f'</span>'
                        f'<span class="intent-pct">{pct:.1f}%</span>'
                        f'</div>'
                    )
                st.markdown(rows_html, unsafe_allow_html=True)

                # Confidence interpretation
                st.markdown("")
                if confidence >= 0.75:
                    st.success("✅ **High confidence** — prediction is reliable")
                elif confidence >= 0.50:
                    st.warning("⚠️ **Moderate confidence** — try rephrasing with more specific keywords")
                else:
                    st.error("❌ **Low confidence** — message is ambiguous; add more detail")

elif page == "Analytics":
    st.markdown('<p class="main-title">Analytics Dashboard</p>', unsafe_allow_html=True)

    report = metrics.get("classification_report", {})
    macro = report.get("macro avg", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Accuracy ({metrics.get('model_name', 'Classifier')})", f"{metrics.get('accuracy', 0):.2%}")
    m2.metric("Macro Precision", f"{macro.get('precision', 0):.2%}")
    m3.metric("Macro Recall", f"{macro.get('recall', 0):.2%}")
    m4.metric("Macro F1", f"{macro.get('f1-score', 0):.2%}")

    st.markdown("---")

    def show_image(path, caption):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"Run `python train.py` to generate {caption}.")

    col1, col2 = st.columns(2)
    with col1:
        show_image("visuals/confusion_matrix.png", "Confusion Matrix")
        show_image("visuals/model_comparison.png", "Model Comparison")
    with col2:
        show_image("visuals/intent_distribution.png", "Intent Distribution")
        show_image("visuals/language_accuracy.png", "Language-wise Accuracy")

    rows = []
    for intent_name, values in report.items():
        if isinstance(values, dict) and intent_name not in {"accuracy", "macro avg", "weighted avg"}:
            rows.append(
                {
                    "Intent": intent_name,
                    "Precision": f"{values.get('precision', 0):.2f}",
                    "Recall": f"{values.get('recall', 0):.2f}",
                    "F1-Score": f"{values.get('f1-score', 0):.2f}",
                    "Support": int(values.get("support", 0)),
                }
            )

    st.markdown("---")
    st.subheader("Per-Intent Metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Sample")
    df = load_dataset()
    if df.empty:
        st.info("Dataset not found. Run `python train.py` first.")
    else:
        st.dataframe(df.sample(min(20, len(df)), random_state=7), use_container_width=True)
