#!/usr/bin/env python3
"""
FactLens - Fake News Detector (Command Line Interface & Model Training)
An end-to-end NLP classifier for detecting real vs. fake news.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def initialize_nltk():
    """Ensure required NLTK resources are available."""
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)


def load_dataset(dataset_dir="dataset"):
    """Load True and Fake CSV datasets."""
    true_path = os.path.join(dataset_dir, "True.csv")
    fake_path = os.path.join(dataset_dir, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(f"Dataset files not found in '{dataset_dir}'. Expected 'True.csv' and 'Fake.csv'.")

    print(f"Loading datasets from '{dataset_dir}'...")
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 1
    fake_df['label'] = 0

    data = pd.concat([true_df, fake_df], ignore_index=True)
    # Combine title and text so the vocabulary represents both headlines and full articles
    data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
    print(f"Loaded {len(true_df):,} real articles and {len(fake_df):,} fake articles (Total: {len(data):,}).")
    return data


def clean_text(text, stop_words):
    """Tokenize, lowercase, and filter stopwords/punctuation."""
    tokens = nltk.word_tokenize(str(text).lower())
    filtered = [w for w in tokens if w not in stop_words and w.isalnum()]
    return " ".join(filtered)


def train_models(data, sample_size=None):
    """Train and evaluate the classification models."""
    initialize_nltk()
    stop_words = set(stopwords.words('english'))

    df = data if sample_size is None else data.sample(sample_size, random_state=42)

    print(f"\nPreprocessing text for {len(df):,} records...")
    df['clean_text'] = df['content'].apply(lambda x: clean_text(x, stop_words))

    print("Extracting TF-IDF features (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = []
    print("\nTraining and evaluating models:")
    print("-" * 65)
    print(f"{'Algorithm':<26} | {'Accuracy':<10} | {'Precision':<10} | {'F1 Score':<10}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "name": name,
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "f1": f1
        })
        print(f"{name:<26} | {acc:.4f}     | {prec:.4f}     | {f1:.4f}")

    print("-" * 65)
    return tfidf, models, stop_words


def predict_news(text, tfidf, model, stop_words):
    """Predict whether a given headline or article is Real or Fake with domain diagnostics."""
    clean = clean_text(text, stop_words)
    tokens = clean.split()
    
    # Check vocabulary coverage
    vocab = tfidf.vocabulary_
    matched_words = [w for w in tokens if w in vocab]
    coverage = len(matched_words) / max(len(tokens), 1)

    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]

    prob_fake = 0.5
    prob_real = 0.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        prob_fake = probs[0]
        prob_real = probs[1]
    
    confidence = max(prob_fake, prob_real)
    
    # Classify result with confidence threshold
    if confidence < 0.60:
        verdict = "UNCERTAIN / LOW CONFIDENCE"
        note = "Notice: The model has low certainty (<60%). The query contains out-of-domain words (e.g. science/space) not covered in the Political/World News training dataset."
    elif pred == 1:
        verdict = "REAL NEWS"
        note = f"Identified as legitimate news with {prob_real*100:.1f}% confidence."
    else:
        verdict = "FAKE NEWS"
        note = f"Identified as fabricated or fake news with {prob_fake*100:.1f}% confidence."

    return {
        "verdict": verdict,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "note": note,
        "matched_words": matched_words,
        "total_words": len(tokens)
    }


def main():
    parser = argparse.ArgumentParser(description="FactLens - News Authenticity Verification & Classifier CLI")
    parser.add_argument("--predict", type=str, help="Single news headline or text to classify using local NLP model")
    parser.add_argument("--live", type=str, help="Verify any real-world news claim live across global wires (100% Free, zero API key)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (train on a 5,000-sample subset for fast verification)")
    args = parser.parse_args()

    # Immediate Live Web Fact-Checking (No model training required)
    if args.live:
        from live_verifier import verify_news_live
        print(f"\n==================================================")
        print(f"[*] FactLens Live Web Fact-Checking Engine")
        print(f"Query: \"{args.live}\"")
        print(f"==================================================")
        res = verify_news_live(args.live)
        print(f"Verdict:         [{res['verdict']}] ({res['confidence']:.1f}% confidence)")
        print(f"Headline:        {res.get('headline', '')}")
        print(f"Analysis:        {res.get('explanation', '')}")
        if res.get('debunked_by'):
            print(f"Debunked By:     {res['debunked_by']}")
        
        articles = res.get('articles', [])
        print(f"\nCorroborating Articles ({len(articles)} found):")
        for i, art in enumerate(articles[:4], 1):
            print(f"  {i}. [{art['source']}] {art['title']}")
            print(f"     Link: {art['link']}")
        print(f"==================================================\n")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")

    data = load_dataset(dataset_dir)
    sample_size = 5000 if args.quick else None

    tfidf, models, stop_words = train_models(data, sample_size=sample_size)
    classifier = models["Multinomial Naive Bayes"]

    if args.predict:
        print(f"\n==================================================")
        print(f"Prediction Query: \"{args.predict}\"")
        print(f"==================================================")
        res = predict_news(args.predict, tfidf, classifier, stop_words)
        print(f"Verdict:         [{res['verdict']}]")
        print(f"Probability:     Real: {res['prob_real']*100:.1f}% | Fake: {res['prob_fake']*100:.1f}%")
        print(f"Vocab Matched:   {len(res['matched_words'])}/{res['total_words']} words ({', '.join(res['matched_words']) if res['matched_words'] else 'None'})")
        print(f"Analysis:        {res['note']}")
        print(f"==================================================\n")
    else:
        # Run test samples
        print("\n--- Testing Sample Headlines ---")
        samples = [
            ("Donald Trump Sends Out Embarrassing New Year Tweet", "Fake News Sample"),
            ("As U.S. budget fight looms, Republicans flip their fiscal script", "Real News Sample"),
            ("Pope Francis shocks world, endorses Donald Trump for presidential election", "Fake News Sample"),
            ("Federal Reserve raises benchmark interest rate by quarter percentage point", "Real News Sample"),
            ("NASA discovers new exoplanet with possible signs of water", "Science / Out-of-Domain Sample")
        ]
        for s, desc in samples:
            res = predict_news(s, tfidf, classifier, stop_words)
            print(f"Headline: '{s}' ({desc})")
            print(f"Verdict:  [{res['verdict']}] (Real: {res['prob_real']*100:.1f}%, Fake: {res['prob_fake']*100:.1f}%)")
            print(f"Note:     {res['note']}\n")


if __name__ == "__main__":
    main()
