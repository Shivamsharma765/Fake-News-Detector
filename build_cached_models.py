import os
import re
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("1. Preparing NLTK...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("2. Loading dataset...")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")

true_df = pd.read_csv(os.path.join(dataset_dir, "True.csv"))
fake_df = pd.read_csv(os.path.join(dataset_dir, "Fake.csv"))
true_df['label'] = 1
fake_df['label'] = 0

data = pd.concat([true_df, fake_df], ignore_index=True)
data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')

print("3. Preprocessing text (optimized)...")
def fast_clean(text):
    words = re.findall(r'\b[a-zA-Z]{2,}\b', str(text).lower())
    return " ".join([w for w in words if w not in stop_words])

data['clean_content'] = data['content'].apply(fast_clean)

print("4. Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_content'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("5. Training all 5 models...")
models = {
    "Multinomial Naive Bayes": MultinomialNB().fit(X_train, y_train),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42).fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train)
}

print("6. Calculating benchmark metrics...")
metrics = []
cms = {}
for name, m in models.items():
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    metrics.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })
    cms[name] = confusion_matrix(y_test, preds)

metrics_df = pd.DataFrame(metrics)

cache_data = {
    "tfidf": tfidf,
    "models": models,
    "metrics_df": metrics_df,
    "cms": cms,
    "num_true": len(true_df),
    "num_fake": len(fake_df)
}

cache_file = os.path.join(current_dir, "models.pkl")
joblib.dump(cache_data, cache_file)
print(f"✅ Cache created successfully at: {cache_file}!")
