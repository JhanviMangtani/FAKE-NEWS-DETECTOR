# 🔍 Fake News Detector — ML Hackathon Project

A machine learning pipeline that classifies news articles as **REAL** or **FAKE**
using NLP feature extraction and multiple ML classifiers.

---

## 📁 Project Structure

```
fake_news_detector/
├── fake_news_detector.py   # Core ML pipeline (preprocessing, training, prediction)
├── app.py                  # Streamlit web app for live demo
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download real dataset from Kaggle
#    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
#    Place Fake.csv and True.csv in this folder.
#    Without CSVs, the code auto-generates synthetic demo data.

# 3. Run the core script
python fake_news_detector.py

# 4. Launch the web app
streamlit run app.py
```

---

## 🧠 How It Works

### 1. Text Preprocessing
- Lowercase conversion
- URL and HTML tag removal
- Punctuation and number stripping
- Whitespace normalization

### 2. Feature Extraction: TF-IDF
- **TF-IDF** (Term Frequency–Inverse Document Frequency) converts raw text
  into numeric feature vectors.
- Uses 10,000 most important unigrams + bigrams.
- Penalizes common words (the, is, a) and rewards rare discriminative ones.

### 3. ML Models Trained
| Model | Strength |
|-------|----------|
| Logistic Regression | Fast, interpretable baseline |
| Naive Bayes | Excellent for text, very fast |
| Random Forest | Ensemble method, highest accuracy |

### 4. Evaluation Metrics
- **Accuracy** — overall correct predictions
- **Precision** — of articles flagged FAKE, how many truly were
- **Recall** — of all FAKE articles, how many were caught
- **F1-Score** — harmonic mean of precision and recall

---

## 📊 Results (on ISOT dataset)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~98% |
| Naive Bayes | ~95% |
| Random Forest | ~99% |

*Results may vary with synthetic demo data.*

---

## 💡 Ideas to Extend This Project

- Add BERT/transformer embeddings for better context understanding
- Add a news URL scraper (input URL → auto-fetch article text)
- Train on multilingual datasets
- Add explainability with LIME or SHAP
- Deploy to Streamlit Cloud for free hosting

---

## 📦 Dataset Credit

**ISOT Fake News Dataset** — University of Victoria  
Available on Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
