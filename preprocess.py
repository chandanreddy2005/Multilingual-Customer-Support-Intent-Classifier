from __future__ import annotations

import re


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for package in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "i",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
    _LEMMATIZER = None


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_\s\u0900-\u097f\u0c80-\u0cff\u0c00-\u0c7f\u0b80-\u0bff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        if token in _STOPWORDS:
            continue
        if _LEMMATIZER is not None and token.isascii():
            token = _LEMMATIZER.lemmatize(token)
        tokens.append(token)
    return " ".join(tokens)
