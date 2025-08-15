import re
import string
import numpy as np
from typing import Tuple, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required resources are available at runtime
def _ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

_ensure_nltk()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    lemmas = [LEMMATIZER.lemmatize(tok) for tok in tokens]
    return " ".join(lemmas)

def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    a_set, b_set = set(a_tokens), set(b_tokens)
    if not a_set and not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
