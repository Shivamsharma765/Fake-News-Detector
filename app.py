import os
import socket
import threading
import streamlit as st
import streamlit.components.v1 as components

# Ensure Backend API on port 5000 is always alive
def ensure_backend_running(port=5000):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return  # already running
    except Exception:
        pass

    try:
        from server import run_server
        t = threading.Thread(target=run_server, args=(port,), daemon=True)
        t.start()
    except Exception as e:
        print(f"Warning: could not auto-start server on port {port}: {e}")

ensure_backend_running(5000)

# Page Configuration
st.set_page_config(
    page_title="FactLens — AI News Credibility & Fact-Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit Default UI Chrome & Fit Full Canvas
st.markdown("""
<style>
    /* Hide all Streamlit default UI chrome */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    
    /* Page background to match reference aesthetic */
    .stApp {
        background-color: #F8F9FD !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow-x: hidden !important;
    }
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        height: 100vh !important;
        min-height: 1000px !important;
        display: block !important;
    }
</style>
""", unsafe_allow_html=True)

# Build Self-Contained Pixel-Perfect Application HTML
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

html_path = os.path.join(FRONTEND_DIR, "index.html")
css_path = os.path.join(FRONTEND_DIR, "style.css")
js_path = os.path.join(FRONTEND_DIR, "app.js")

try:
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    # Inline CSS & JS for zero-latency instant rendering
    bundled_html = html.replace(
        '<link rel="stylesheet" href="style.css" />',
        f"<style>\n{css}\n</style>"
    )
    bundled_html = bundled_html.replace(
        '<script src="app.js"></script>',
        f"<script>\n{js}\n</script>"
    )

    # Render pixel-perfect application
    components.html(bundled_html, height=1050, scrolling=True)

except Exception as e:
    # Graceful fallback to direct port 5000 iframe if files cannot be read
    components.iframe("http://127.0.0.1:5000", height=1050, scrolling=True)
