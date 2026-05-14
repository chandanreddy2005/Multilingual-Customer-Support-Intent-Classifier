#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import pickle
from scipy.sparse import hstack
from preprocess import preprocess

bundle_path = Path("saved_models/model_bundle.pkl")
with open(bundle_path, "rb") as f:
    bundle = pickle.load(f)

test_cases = [
    ("ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ", "Kannada - Delivery Inquiry"),
    ("मुझे रिफंड चाहिए", "Hindi - Refund Request"),
    ("నా ఖాతా లాక్ అయిపోయింది", "Telugu - Account Recovery"),
    ("என் பணம் திரும்பா கொடுங்கள்", "Tamil - Refund Request"),
    ("My payment failed during checkout", "English - Payment Failure"),
]

model = bundle["model"]
label_encoder = bundle["label_encoder"]
labels = label_encoder.classes_

output = []
output.append("Testing Native Script Support:")
output.append("=" * 70)

for text, description in test_cases:
    clean = preprocess(text)
    word_features = bundle["word_vectorizer"].transform([clean])
    char_features = bundle["char_vectorizer"].transform([clean])
    features = hstack([word_features, char_features])

    probabilities = model.predict_proba(features)[0]
    intent_idx = int(probabilities.argmax())
    intent = labels[intent_idx]
    confidence = probabilities[intent_idx]

    output.append(f"\n{description}")
    output.append(f"Text: {repr(text)}")
    output.append(f"✓ Predicted: {intent} ({confidence:.1%})")

result_text = "\n".join(output)

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(result_text)
print("Results saved")
