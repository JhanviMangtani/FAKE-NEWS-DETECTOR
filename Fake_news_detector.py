"""
Fake News Detector - ML Project
Hackathon-ready implementation using NLP + Multiple ML Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.pipeline import Pipeline
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')


# 1. TEXT PREPROCESSING


def preprocess_text(text):
    """Clean and normalize raw news text."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special chars and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# 2. LOAD & PREPARE DATA


def load_data(fake_path='Fake.csv', true_path='True.csv'):
    """
    Load the LIAR / ISOT dataset CSVs.
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    Expected columns: title, text, subject, date
    """
    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df['label'] = 0   # 0 = FAKE
        true_df['label'] = 1   # 1 = REAL

        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✓ Dataset loaded: {len(df)} articles")
        print(f"  FAKE: {(df.label == 0).sum()} | REAL: {(df.label == 1).sum()}")
        return df

    except FileNotFoundError:
        print("⚠ CSV files not found. Generating synthetic demo data...")
        return generate_demo_data()


def generate_demo_data(n=2000):
    """Generate synthetic demo data when the real dataset is unavailable."""
    import random
    random.seed(42)

    fake_phrases = [
        "BREAKING: Scientists SHOCKED by new discovery that they don't want you to know",
        "Government HIDING the truth about vaccines and 5G towers exposed",
        "Celebrity caught in massive scandal you won't believe",
        "Secret cure doctors are SUPPRESSING from the public revealed",
        "Elite conspiracy to control the population exposed by whistleblower",
        "Miracle food that DESTROYS cancer cells overnight discovered",
        "NASA admits ALIENS exist in leaked documents nobody is talking about",
        "Stock market about to CRASH according to insider source from Wall Street",
    ]

    real_phrases = [
        "Researchers publish new study in peer-reviewed journal on climate effects",
        "Central bank announces interest rate decision following economic review",
        "Scientists discover new species of marine life in Pacific Ocean",
        "Government releases annual budget report showing fiscal targets",
        "University study examines social media usage patterns in teenagers",
        "Tech company reports quarterly earnings in line with analyst expectations",
        "Health officials recommend updated vaccination schedule for adults",
        "International summit concludes with agreement on trade tariffs",
    ]

    rows = []
    for _ in range(n // 2):
        rows.append({
            'text': random.choice(fake_phrases) + " " + " ".join(
                random.choices(["shocking", "exposed", "secret", "hidden",
                                "government", "mainstream media", "cover-up",
                                "elite", "they", "truth", "wake up"], k=20)
            ),
            'label': 0
        })
        rows.append({
            'text': random.choice(real_phrases) + " " + " ".join(
                random.choices(["according", "study", "researchers", "officials",
                                "announced", "report", "data", "analysis",
                                "evidence", "published", "confirmed"], k=20)
            ),
            'label': 1
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Demo dataset created: {len(df)} synthetic articles")
    return df



# 3. FEATURE ENGINEERING


def build_features(df):
    """Create the text feature column for modeling."""
    # Combine title + text if both columns exist
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['content'] = df['text'].fillna('')
    else:
        raise ValueError("DataFrame must have a 'text' column")

    df['content'] = df['content'].apply(preprocess_text)
    return df


# 4. MODEL TRAINING & EVALUATION


def train_and_evaluate(df):
    """Train multiple models and compare performance."""

    df = build_features(df)
    X = df['content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'═'*50}")
    print("  MODEL TRAINING & EVALUATION")
    print(f"{'═'*50}")
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

    # Define models with TF-IDF pipelines
    models = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf',   LogisticRegression(max_iter=1000, C=1.0))
        ]),
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf',   MultinomialNB(alpha=0.1))
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
            ('clf',   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
    }

    results = {}
    best_model = None
    best_acc = 0
    best_name = ""

    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'pipeline': pipeline,
            'report': classification_report(y_test, y_pred,
                                            target_names=['FAKE', 'REAL'],
                                            output_dict=True)
        }

        print(f"    ✓ Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_name = name

    # Print detailed report for best model
    print(f"\n{'─'*50}")
    print(f"  BEST MODEL: {best_name} ({best_acc:.4f})")
    print(f"{'─'*50}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best,
                                 target_names=['FAKE', 'REAL']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print("  Confusion Matrix:")
    print(f"             Predicted FAKE  Predicted REAL")
    print(f"  Actual FAKE     {cm[0][0]:5d}          {cm[0][1]:5d}")
    print(f"  Actual REAL     {cm[1][0]:5d}          {cm[1][1]:5d}")

    # Save best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n  ✓ Best model saved to best_model.pkl")

    return best_model, results



# 5. INFERENCE / PREDICTION


def predict(text, model):
    """
    Predict whether a news article is FAKE or REAL.

    Returns:
        dict with label, confidence, and top keywords
    """
    cleaned = preprocess_text(text)
    label_id = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = proba[label_id]

    label = "REAL" if label_id == 1 else "FAKE"
    color = "✅" if label_id == 1 else "🚨"

    # Extract top TF-IDF keywords that influenced the prediction
    tfidf = model.named_steps['tfidf']
    tfidf_vector = tfidf.transform([cleaned])
    feature_names = np.array(tfidf.get_feature_names_out())
    top_indices = tfidf_vector.toarray()[0].argsort()[-5:][::-1]
    top_keywords = feature_names[top_indices].tolist()

    return {
        'label': label,
        'confidence': confidence,
        'fake_probability': proba[0],
        'real_probability': proba[1],
        'top_keywords': top_keywords,
        'emoji': color
    }


def print_prediction(text, result):
    """Pretty-print a prediction result."""
    print(f"\n{'─'*55}")
    print(f"  Article: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"  Verdict : {result['emoji']} {result['label']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  FAKE probability : {result['fake_probability']:.1%}")
    print(f"  REAL probability : {result['real_probability']:.1%}")
    print(f"  Key words : {', '.join(result['top_keywords'])}")


# 6. MAIN


if __name__ == '__main__':
    print("=" * 55)
    print("      FAKE NEWS DETECTOR — ML Hackathon Project")
    print("=" * 55)

    # Load data (uses demo data if CSVs not present)
    df = load_data()

    # Train models
    model, results = train_and_evaluate(df)

    # --- Test with sample articles ---
    print(f"\n{'═'*55}")
    print("  SAMPLE PREDICTIONS")
    print(f"{'═'*55}")

    test_articles = [
        ("BREAKING: Scientists SHOCKED by secret cure that DESTROYS "
         "cancer overnight that government is HIDING from you!"),

        ("Researchers at Stanford University published a peer-reviewed "
         "study in Nature examining the long-term effects of diet on "
         "cardiovascular health in adults over 60."),

        ("EXPOSED: Elite conspiracy to control world population through "
         "water supply revealed by anonymous whistleblower, mainstream "
         "media is covering it up!"),

        ("The Federal Reserve announced a 25-basis-point interest rate "
         "increase on Wednesday, citing persistent inflation data from "
         "the Bureau of Labor Statistics."),
    ]

    for article in test_articles:
        result = predict(article, model)
        print_prediction(article, result)

    print(f"\n{'─'*55}")
    print("  Summary of all models:")
    for name, res in results.items():
        print(f"    {name:<22} Accuracy: {res['accuracy']:.4f}")
    print(f"{'─'*55}\n")