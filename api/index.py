# FactLens - Vercel Serverless Function API Entrypoint
from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Ensure api directory and parent directory are on sys.path
api_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(api_dir)
for p in [api_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_verifier import verify_news_live

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if any(self.path.startswith(p) for p in ('/api/health', '/health', '/api/index.py', '/api')):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"FactLens on Vercel"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if '/api/predict' in self.path or '/predict' in self.path:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode('utf-8'))
                raw_text = payload.get('text', '').strip()
                mode = payload.get('mode', 'live')

                if not raw_text:
                    self.send_error(400, "Missing 'text' field.")
                    return

                # Live Global Web Verification (Default & Highly Accurate)
                if mode == 'live':
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
                        "debunked_by": live_res.get("debunked_by", ""),
                        "prob_real": p_real,
                        "prob_fake": p_fake
                    }
                else:
                    # Offline NLP Fallback
                    try:
                        import joblib
                        cache_path = os.path.join(parent_dir, "models.pkl")
                        if os.path.exists(cache_path):
                            cache = joblib.load(cache_path)
                            tfidf = cache["tfidf"]
                            model = cache["models"].get(payload.get('model', 'Multinomial Naive Bayes'))
                            cleaned = raw_text.lower()
                            vec = tfidf.transform([cleaned])
                            pred = int(model.predict(vec)[0])
                            prob_real = 0.5
                            prob_fake = 0.5
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(vec)[0]
                                prob_fake = float(probs[0])
                                prob_real = float(probs[1])
                            confidence = max(prob_real, prob_fake)
                            verdict = "REAL NEWS" if pred == 1 else "FAKE NEWS"
                            response = {
                                "mode": "nlp",
                                "verdict": verdict,
                                "pred": pred,
                                "prob_real": prob_real,
                                "prob_fake": prob_fake,
                                "confidence": confidence,
                                "note": "Evaluated via local NLP classification model.",
                                "matched_words": [],
                                "unmatched_words": [],
                                "total_words": len(raw_text.split())
                            }
                        else:
                            live_res = verify_news_live(raw_text)
                            response = {
                                "mode": "live",
                                "verdict": live_res.get("verdict", "UNVERIFIED"),
                                "confidence": float(live_res.get("confidence", 75.0)) / 100.0,
                                "note": live_res.get("explanation", ""),
                                "articles": live_res.get("articles", [])
                            }
                    except Exception:
                        live_res = verify_news_live(raw_text)
                        response = {
                            "mode": "live",
                            "verdict": live_res.get("verdict", "UNVERIFIED"),
                            "confidence": float(live_res.get("confidence", 75.0)) / 100.0,
                            "note": live_res.get("explanation", ""),
                            "articles": live_res.get("articles", [])
                        }

                resp_bytes = json.dumps(response).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(resp_bytes)

            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(err_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
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
