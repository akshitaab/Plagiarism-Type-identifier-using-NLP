# 📝 Plagiarism Type Identifier (NLP + ML)

Detect and **classify** plagiarism into four types: **Verbatim, Mosaic, Paraphrased, Translated** using a clean NLP pipeline and multiple ML models (**Naive Bayes, Logistic Regression, SVM, Random Forest**).

## ✨ Highlights
- Clear, modular code (`src/`).
- Uses TF‑IDF + similarity features (cosine, Jaccard, Levenshtein).
- Generates a **classification report, confusion matrix, and a model comparison chart** in `results/`.
- Human-readable docs and comments (great for viva/github).

## 📦 Project Structure
```
plagiarism-type-identifier/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── plagiarism_detection.py
│   └── utils.py
├── data/
│   └── plagiarism_dataset.csv   # ← put your CSV here (see schema below)
└── results/
    ├── confusion_matrix.png
    └── model_comparison.png
```

## 🗂️ Dataset Schema
Place a CSV at `data/plagiarism_dataset.csv` with columns:
```
original_text,suspicious_text,label
"source sentence 1","student sentence 1","Verbatim"
"source sentence 2","student sentence 2","Paraphrased"
...
```
**Allowed labels:** `Verbatim`, `Mosaic`, `Paraphrased`, `Translated`

> Tip: If your file has different column names, update the constants at the top of `plagiarism_detection.py`.

## ▶️ Quickstart
```bash
pip install -r requirements.txt
python src/plagiarism_detection.py
```
Outputs (saved to `results/`):
- `confusion_matrix.png` – per-class performance
- `model_comparison.png` – accuracy bar chart
- Console logs – accuracy & classification reports for each model

## ⚙️ How It Works
1. **Preprocess**: lowercasing, punctuation/stopword removal, lemmatization.
2. **Vectorize**: TF‑IDF on original/suspicious text **separately**.
3. **Similarity features**: cosine similarity (TF‑IDF), Jaccard overlap, Levenshtein similarity.
4. **Train & Evaluate**: MultinomialNB, LogisticRegression, LinearSVC, RandomForestClassifier.
5. **Visualize**: confusion matrix + comparison chart.

## 📈 Why These Models?
- **Naive Bayes** – fast, strong baseline for text.
- **Logistic Regression** – robust linear classifier, interpretable.
- **SVM (Linear)** – excellent in high-dimensional TF‑IDF space.
- **Random Forest** – non-linear, handles interactions and noise.

## 🚧 Limitations
- Paraphrase vs verbatim can still confuse shallow features.
- Cross‑lingual (Translated) cases benefit from multilingual embeddings (future work).

## 🔮 Future Scope
- Swap TF‑IDF for **Sentence-BERT** embeddings.
- Add **cross‑lingual** support (LaBSE, mUSE, mBERT).
- Ship as a **Streamlit** app or REST API.

---
**Author:** Akshita Bharadwaj • **License:** MIT
