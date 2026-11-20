#!/usr/bin/env python3
"""
FactLens - Standalone Web Server & Inference API
Serves modern glassmorphic frontend and handles /api/predict requests.
Supports both Live Global Web Fact-Checking & Local NLP Machine Learning.
"""

import os
import sys
import json
import re
import joblib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nltk
from nltk.corpus import stopwords
from live_verifier import verify_news_live

# Global model store
MODELS = {}
TFIDF = None
STOP_WORDS = None

def init_nlp_and_models():
    global MODELS, TFIDF, STOP_WORDS
    print("\n[1/2] Initializing NLP resources...", flush=True)
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        try:
            nltk.download('stopwords', quiet=True)
            STOP_WORDS = set(stopwords.words('english'))
        except Exception:
            STOP_WORDS = set()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, "models.pkl")

    if os.path.exists(cache_path):
        print(f"[2/2] Loading pre-trained models from cache ({cache_path})...", flush=True)
        cache = joblib.load(cache_path)
        TFIDF = cache["tfidf"]
        MODELS = cache["models"]
        print("[OK] Models loaded instantly from cache!", flush=True)
    else:
        print("[2/2] Training models (first run only)...", flush=True)
        import build_cached_models
        cache = joblib.load(cache_path)
        TFIDF = cache["tfidf"]
        MODELS = cache["models"]

    print("[OK] All 5 classifiers and live verifier ready for web inference!\n", flush=True)


class FactLensHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        super().__init__(*args, directory=frontend_dir, **kwargs)

    def do_GET(self):
        if self.path in ('/health', '/api/health'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"FactLens"}')
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                raw_text = payload.get('text', '').strip()
                mode = payload.get('mode', 'live')
                model_name = payload.get('model', 'Multinomial Naive Bayes')
                
                if not raw_text:
                    self.send_error(400, "Missing 'text' field.")
                    return

                if mode == 'live':
                    # LIVE WEB & FACT CHECK SEARCH
                    live_res = verify_news_live(raw_text)
                    conf_raw = float(live_res.get("confidence", 75.0)) / 100.0
                    verdict_str = live_res.get("verdict", "UNVERIFIED")

                    if "REAL" in verdict_str:
                        p_real = conf_raw
                        p_fake = max(1.0 - conf_raw, 0.02)
                    elif "FAKE" in verdict_str or "HOAX" in verdict_str:
                        p_fake = conf_raw
                        p_real = max(1.0 - conf_raw, 0.02)
                    elif "RUMOR" in verdict_str or "UNVERIFIED" in verdict_str:
                        p_fake = conf_raw
                        p_real = max(1.0 - conf_raw, 0.1)
                    else:
                        p_real = 0.50
                        p_fake = 0.50

                    response = {
                        "mode": "live",
                        "verdict": verdict_str,
                        "badge": live_res.get("badge", "neutral"),
                        "confidence": conf_raw,
                        "headline": live_res.get("headline", ""),
                        "note": live_res.get("explanation", ""),
                        "articles": live_res.get("articles", []),
                        "debunked_by": live_res.get("debunked_by", None),
                        "prob_real": p_real,
                        "prob_fake": p_fake
                    }
                else:
                    # OFFLINE NLP LINGUISTIC MODEL
                    model = MODELS.get(model_name, MODELS["Multinomial Naive Bayes"])
                    words = re.findall(r'\b[a-zA-Z]{2,}\b', raw_text.lower())
                    clean_tokens = [w for w in words if w not in STOP_WORDS]
                    cleaned = " ".join(clean_tokens)

                    vocab = TFIDF.vocabulary_
                    matched_words = [w for w in clean_tokens if w in vocab]
                    unmatched_words = [w for w in clean_tokens if w not in vocab]

                    vec = TFIDF.transform([cleaned])
                    pred = int(model.predict(vec)[0])

                    prob_real = 0.5
                    prob_fake = 0.5
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                        prob_fake = float(probs[0])
                        prob_real = float(probs[1])
                    elif hasattr(model, "decision_function"):
                        score = float(model.decision_function(vec)[0])
                        prob_real = 1.0 / (1.0 + np.exp(-score))
                        prob_fake = 1.0 - prob_real

                    confidence = max(prob_real, prob_fake)

                    if confidence < 0.60:
                        verdict = "UNCERTAIN / LOW CONFIDENCE"
                        note = f"The model has low certainty ({confidence*100:.1f}%). The query contains out-of-domain words not covered in the political/world news dataset."
                    elif pred == 1:
                        verdict = "REAL NEWS"
                        note = f"High alignment with verified journalistic reporting patterns ({prob_real*100:.1f}% confidence)."
                    else:
                        verdict = "FAKE NEWS"
                        note = f"High probability of fabricated content or sensational disinformation ({prob_fake*100:.1f}% confidence)."

                    response = {
                        "mode": "nlp",
                        "verdict": verdict,
                        "pred": pred,
                        "prob_real": prob_real,
                        "prob_fake": prob_fake,
                        "confidence": confidence,
                        "note": note,
                        "matched_words": matched_words,
                        "unmatched_words": unmatched_words,
                        "total_words": len(clean_tokens)
                    }

                response_bytes = json.dumps(response).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_bytes)

            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(404, "Endpoint Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def run_server(port=None):
    init_nlp_and_models()
    if port is None:
        env_port = os.environ.get("PORT")
        if env_port and env_port.isdigit():
            port = int(env_port)
        elif len(sys.argv) > 1 and sys.argv[1].isdigit():
            port = int(sys.argv[1])
        else:
            port = 5000

    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, FactLensHandler)
    url = f"http://0.0.0.0:{port}"
    print("=" * 55, flush=True)
    print(f"[*] FactLens Web App running at: {url} (Port: {port})", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    print("=" * 55 + "\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    run_server()
