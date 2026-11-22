"""Quotex AI Signal Agent - Streamlit version."""
import os
import sys
import hmac

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Quotex AI Signal Agent", page_icon="📈", layout="wide")

# --- keys: from Streamlit Secrets (or environment variables) ---
for _k in ("TWELVE_DATA_API_KEY", "GEMINI_API_KEY", "APP_PASSWORD"):
    if not os.environ.get(_k):
        try:
            if _k in st.secrets:
                os.environ[_k] = str(st.secrets[_k])
        except Exception:
            pass

import engine  # noqa: E402  (analysis engine, same code that was tested in Colab)

engine.TD_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
engine.GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# --- optional password (add APP_PASSWORD in Secrets so strangers can't use your free API limits) ---
_pw = os.environ.get("APP_PASSWORD", "").strip()
if _pw and not st.session_state.get("auth_ok"):
    st.title("📈 Quotex AI Signal Agent")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), _pw.encode()):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

if "hist" not in st.session_state:
    st.session_state["hist"] = []
    st.session_state["md"] = None
    st.session_state["chart"] = None

st.title("📈 Quotex AI Signal Agent")
st.caption("Pick the market, market type and candle time, then press Generate Signal. You always get a direction, graded "
           "A (confirmed), B (moderate) or C (weak lean) - the tracker below shows which grades really work for you.")

left, right = st.columns([1, 2])
with left:
    market = st.selectbox("1) Market", engine.MARKETS, index=0)
    mtype = st.radio("2) Market type", ["Live Market", "OTC"], horizontal=True)
    tf = st.selectbox("3) Candle time", list(engine.TF.keys()), index=1)
    mode = st.radio("Signal mode", ["Always give me a direction (recommended)", "Confirmed signals only"],
                    help="Grade A = confirmed, B = moderate, C = weak lean. 'Confirmed only' shows NO TRADE most of the time.")
    strict = st.selectbox("Strictness for Grade A", list(engine.STRICT.keys()), index=0)
    with st.expander("OTC input: screenshot or pasted candles (needed only for OTC)"):
        upload = st.file_uploader("Chart screenshot (60+ candles visible works best)", type=["png", "jpg", "jpeg", "webp"])
        pasted = st.text_area("Or paste candles: open, high, low, close (one per line, oldest first)", height=140)
    with st.expander("Risk & filters"):
        balance = st.number_input("Account balance", min_value=0.0, value=100.0, step=10.0)
        risk = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5)
        payout = st.slider("Quotex payout (%)", 70, 95, 85, 1)
        req_edge = st.checkbox("Confirmed-only mode: also block when the backtest is below break-even", value=False)
    go = st.button("🚀 Generate Signal", type="primary")

if go:
    img = None
    if upload is not None:
        try:
            img = Image.open(upload)
            img.load()
        except Exception:
            img = None
    with st.spinner("Analysing the market..."):
        md, chart, _hdf, _stats, hist = engine.analyze_market(
            market, mtype, tf, strict, img, pasted, balance, risk, payout, req_edge, st.session_state["hist"],
            always_lean=mode.startswith("Always"))
    st.session_state["md"], st.session_state["chart"], st.session_state["hist"] = md, chart, hist

with right:
    if st.session_state["md"]:
        st.markdown(st.session_state["md"])
        if st.session_state["chart"] is not None:
            st.image(st.session_state["chart"])
    else:
        st.info("Choose your options on the left and press **Generate Signal**.")

st.divider()
st.subheader("📒 Track your results (this is how you find out if it really works for you)")
b1, b2, _ = st.columns([1, 1, 3])
if b1.button("✅ Last trade: WIN"):
    _, _, st.session_state["hist"] = engine.record_win(st.session_state["hist"])
if b2.button("❌ Last trade: LOSS"):
    _, _, st.session_state["hist"] = engine.record_loss(st.session_state["hist"])
st.markdown(engine.stats_text(st.session_state["hist"]))
st.dataframe(engine.hist_frame(st.session_state["hist"]), hide_index=True)
st.caption("Educational tool only - not financial advice. Binary options carry a high risk of losing your whole stake. "
           "History is kept only while this browser session is open.")
