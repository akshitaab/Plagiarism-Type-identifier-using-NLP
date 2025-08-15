import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import textdistance

from utils import clean_text, jaccard_similarity

# ------------------ Config ------------------
DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "plagiarism_dataset.csv"))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ORIG_COL = os.getenv("ORIG_COL", "original_text")
SUSP_COL = os.getenv("SUSP_COL", "suspicious_text")
LABEL_COL = os.getenv("LABEL_COL", "label")
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Clean text
    df["orig_clean"] = df[ORIG_COL].apply(clean_text)
    df["susp_clean"] = df[SUSP_COL].apply(clean_text)

    # TF-IDF per side, then compute cosine between sides
    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1,2))
    # We'll fit on concatenated corpus for a robust vocabulary
    corpus = pd.concat([df["orig_clean"], df["susp_clean"]], axis=0).values
    tfidf.fit(corpus)
    o_vecs = tfidf.transform(df["orig_clean"])
    s_vecs = tfidf.transform(df["susp_clean"])

    # Cosine similarity between pair
    cos_sims = []
    for i in range(df.shape[0]):
        cos = cosine_similarity(o_vecs[i], s_vecs[i])[0][0]
        cos_sims.append(cos)
    df["cosine_similarity"] = cos_sims

    # Jaccard similarity on tokens
    df["jaccard"] = [
        jaccard_similarity(oc.split(), sc.split())
        for oc, sc in zip(df["orig_clean"], df["susp_clean"])
    ]

    # Levenshtein normalized similarity on cleaned strings
    df["levenshtein"] = [
        textdistance.levenshtein.normalized_similarity(oc, sc)
        for oc, sc in zip(df["orig_clean"], df["susp_clean"])
    ]

    return df[["cosine_similarity", "jaccard", "levenshtein"]], tfidf

def train_and_evaluate(X, y, label_encoder) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=None),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        )
    }

    accuracies = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        # Confusion matrix for the best model candidate can be drawn later; for now, draw for each and overwrite
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plt.close()

    # Comparison bar chart
    plt.figure(figsize=(7,4))
    plt.bar(list(accuracies.keys()), list(accuracies.values()))
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    plt.close()

    return accuracies

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your CSV there with columns: "
            f"{ORIG_COL}, {SUSP_COL}, {LABEL_COL}."
        )
    df = pd.read_csv(DATA_PATH)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL].astype(str))

    # Build handcrafted similarity features
    X_feats, _ = build_features(df)

    # Train + evaluate on handcrafted features
    accuracies = train_and_evaluate(X_feats, y, label_encoder)

    print("\nSaved: results/confusion_matrix.png and results/model_comparison.png")

if __name__ == "__main__":
    main()
