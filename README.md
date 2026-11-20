# FactLens: AI Fake News Detector & Live Web Fact-Checker

A comprehensive Natural Language Processing (NLP) and Real-Time Web Intelligence platform that classifies news stories, verifies arbitrary global claims, and debunks misinformation using machine learning and live journalistic wires.

---

## 📌 Dual Verification Engine

FactLens addresses the traditional limitation of static machine learning models (which only know their training data) by providing two complementary verification engines:

### 1. 🌐 Live Global Web Fact-Checking (Any News Topic • 100% Free)
- **Zero API keys required**: Completely free, unlimited queries powered by real-time Google News RSS wires and global journalism registries.
- **Universal coverage**: Works on **any** random news claim, modern topic, or viral rumor: science, technology, medicine/health, space, sports, and current politics.
- **Fact-Checker integration**: Automatically cross-references claims against recognized fact-checking organizations including **FactCheck.org**, **Snopes**, **PolitiFact**, **Reuters Fact Check**, **AP Fact Check**, and **AFP**.
- **Source article citations**: Returns clickable links and publication timestamps from established news wires (Reuters, BBC, AP, NASA, The New York Times, etc.).

### 2. 🧠 Offline NLP Multi-Model Classifier
- Evaluates linguistic and vocabulary reporting patterns trained on **44,898 genuine and fake news articles**.
- Compares 5 distinct machine learning architectures:
  1. **Random Forest Classifier** (99.79% Accuracy)
  2. **Decision Tree Classifier** (99.54% Accuracy)
  3. **Passive Aggressive Classifier** (99.25% Accuracy)
  4. **Logistic Regression** (98.73% Accuracy)
  5. **Multinomial Naive Bayes** (93.06% Accuracy)
- Includes domain-awareness diagnostics: warns when testing out-of-domain words not covered in historical training datasets.

---

## 🎨 User Interfaces

You have two interactive frontend options included out-of-the-box:

### 1. Standalone Glassmorphic Single-Page Web App
Features an ultra-sleek dark glassmorphic design with circular radial confidence gauges, live wire article cards, and mode switching between Live Web Search and Local NLP:

```powershell
python server.py
```
*Opens at `http://localhost:5000`*

### 2. Streamlit Web Dashboard
Features comprehensive model benchmarking charts, interactive confusion matrices, probability breakdown meters, and live web fact-checking:

```powershell
streamlit run app.py
```
*Opens automatically at `http://localhost:8501`*

---

## 🚀 How to Run in VS Code

1. Press `Ctrl + Shift + D` (or click **Run and Debug** on the left sidebar).
2. Choose your preferred run target:
   - **🌐 Launch Streamlit Web UI (Recommended)**
   - **🚀 Launch Standalone Web App (server.py)**
   - **⚡ Run Quick Benchmark CLI**
   - **🏋️ Train on Full Dataset CLI**
3. Press **`F5`**!

---

## 💻 CLI Commands

```powershell
# 1. Verify ANY real-world news claim live across the web (Zero training required)
python train_and_predict.py --live "NASA discovers evidence of water ice on Moon"
python train_and_predict.py --live "Pope Francis endorses Donald Trump for President"

# 2. Test local NLP models on a custom headline
python train_and_predict.py --predict "Federal Reserve raises benchmark interest rate"

# 3. Fast benchmark on a subset
python train_and_predict.py --quick

# 4. Train full models from scratch
python train_and_predict.py
```

---

## 📈 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.79%** | **99.81%** | **99.74%** | **99.78%** |
| **Decision Tree** | **99.54%** | **99.60%** | **99.44%** | **99.52%** |
| **Passive Aggressive** | **99.25%** | **99.23%** | **99.21%** | **99.22%** |
| **Logistic Regression** | **98.73%** | **98.70%** | **98.60%** | **98.65%** |
| **Multinomial Naive Bayes** | **93.06%** | **92.80%** | **92.65%** | **92.72%** |

---

## ☁️ Deployment Guide

FactLens is configured for instant deployment across multiple cloud platforms:

### 1. Deploy on Render (Recommended • Free Web Service)
1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com) and click **New > Web Service**.
3. Select your `Fake-News-Detector` / `FactLens` repository.
4. Render will auto-detect `render.yaml` or you can enter:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
5. Click **Deploy Web Service**! Your live app will be accessible at `https://<your-subdomain>.onrender.com`.

### 2. Deploy on Streamlit Community Cloud (Free)
1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New App**.
3. Select your repository, branch `main`, and main file path: `app.py`.
4. Click **Deploy**!

### 3. Deploy with Docker
```bash
# Build the production image
docker build -t factlens .

# Run the container
docker run -p 5000:5000 factlens
```
Access at `http://localhost:5000`.

---

## 📁 Directory Structure

```
Fake-News-Detector/
├── dataset/
│   ├── Fake.csv           # 23,481 fake news articles
│   └── True.csv           # 21,417 true news articles
├── frontend/
│   ├── index.html         # Modern glassmorphic web UI
│   ├── style.css          # Dark-mode styling & animations
│   └── app.js             # Mode toggle & live news cards rendering
├── live_verifier.py       # 100% Free live web news fact-checker
├── app.py                 # Streamlit web dashboard
├── server.py              # Zero-dependency web server & API (port 5000)
├── models.pkl             # Pre-trained cached model bundle
├── fake_news_detector.ipynb # Interactive Jupyter notebook
├── train_and_predict.py   # CLI tool & inference engine
├── .vscode/               # VS Code debug & launch configs
└── README.md              # Project documentation
```
