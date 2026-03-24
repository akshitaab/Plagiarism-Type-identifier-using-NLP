# Plagiarism Type Identifier — NLP
> Multi-class plagiarism classification using classical NLP pipelines and ensemble ML models.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?style=flat-square&logo=spacy)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

Plagiarism detection is a multi-class classification problem — not just binary matching. This project builds an NLP pipeline that identifies **what type of plagiarism** is present (verbatim, paraphrase, idea/concept, none) rather than simply flagging it.

**Results at a glance:**

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|---|---|---|---|---|
| SVM (TF-IDF) | **91.4%** | **0.89** | 0.90 | 0.88 |
| Logistic Regression | 88.7% | 0.86 | 0.87 | 0.85 |
| Naive Bayes | 83.2% | 0.81 | 0.83 | 0.79 |

> SVM with TF-IDF vectorization outperforms Naive Bayes by **8.2% accuracy** and **+0.08 macro F1**.

---

## Dataset

- **Source:** PAN Plagiarism Corpus / custom labeled text pairs  
- **Format:** `dataset.csv` — paired text documents with plagiarism type labels  
- **Classes:** `verbatim` · `paraphrase` · `idea` · `none`  
- **Note:** Dataset not included due to licensing. Drop in your own `dataset.csv` following the schema in `data/schema.md`.

---

## Methodology

```
Raw Text Pairs
     │
     ▼
1. Preprocessing
   └─ Tokenization → Stopword Removal → Lemmatization (spaCy)
     │
     ▼
2. Feature Engineering
   └─ TF-IDF Vectorization · Cosine Similarity · Jaccard Score · N-gram Overlap
     │
     ▼
3. Model Training
   └─ Naive Bayes · Logistic Regression · SVM (LinearSVC)
     │
     ▼
4. Evaluation
   └─ Accuracy · Precision · Recall · F1 (Macro & Weighted) · Confusion Matrix
```

---

## Project Structure

```
Plagiarism-Type-Identifier-NLP/
│
├── data/
│   ├── schema.md               # Dataset format specification
│   └── sample.csv              # Small sample for quick testing
│
├── src/
│   ├── preprocess.py           # Tokenization, stopword removal, lemmatization
│   ├── features.py             # TF-IDF, similarity metrics, n-gram overlap
│   ├── models.py               # Model definitions and training logic
│   └── evaluate.py             # Metrics, confusion matrix, classification report
│
├── notebooks/
│   └── EDA_and_Experiments.ipynb
│
├── main.py                     # Entry point — runs full pipeline
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone and install
git clone https://github.com/niteshduhan/Plagiarism-Type-Identifier-NLP.git
cd Plagiarism-Type-Identifier-NLP
pip install -r requirements.txt

# Add your dataset
cp /path/to/your/dataset.csv data/dataset.csv

# Run full pipeline
python main.py

# Run on a quick sample
python main.py --data data/sample.csv
```

---

## Key Takeaways

- **Feature engineering matters more than model complexity** — handcrafted similarity features (cosine, Jaccard, n-gram overlap) combined with TF-IDF gave SVM a decisive edge over bag-of-words Naive Bayes.
- **Multi-class NLP classification is a label-imbalance problem** — macro F1 is the metric that actually matters here, not accuracy. Reported both.
- **Classical ML is still competitive** — SVM at 91.4% accuracy demonstrates that transformer-level compute isn't always necessary for structured classification tasks on domain-specific text.

---

## Roadmap

- [ ] BERT / RoBERTa fine-tuning for paraphrase-type detection  
- [ ] Streamlit interface for document upload + real-time classification  
- [ ] Multi-language support  
- [ ] Sentence-level granularity (highlight *which* spans are plagiarised)

---

## Author

**Nitesh Duhan** — Data Scientist · ML Engineer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-niteshduhan--carp112-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/niteshduhan-carp112)
[![Gmail](https://img.shields.io/badge/Gmail-niteshduhan686@gmail.com-red?style=flat-square&logo=gmail)](mailto:niteshduhan686@gmail.com)
[![Instagram](https://img.shields.io/badge/Instagram-@nitesh._duhan-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/nitesh._duhan)

---

*MIT License · Open for collaborations in NLP, ML, and Data Science*
