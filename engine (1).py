"""Analysis engine for the Quotex AI Signal Agent (used by streamlit_app.py)."""
# ============================================================
#  SETUP, KEYS & SETTINGS
# ============================================================
import os, io, re, json, time, base64, datetime as dt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from matplotlib.figure import Figure

pd.options.mode.chained_assignment = None


def get_secret(name):
    """Keys come from environment variables (the Streamlit app copies its secrets in)."""
    return (os.environ.get(name) or "").strip()


TD_KEY = get_secret("TWELVE_DATA_API_KEY")
GEMINI_KEY = get_secret("GEMINI_API_KEY")

# The agent tries these Gemini models in order until one works (names change over time).
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]

TD_URL = "https://api.twelvedata.com/time_series"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LOG_PATH = "/tmp/signal_log.csv"

MARKETS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
           "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD",
           "XAU/USD", "XAG/USD", "BTC/USD", "ETH/USD"]
CRYPTO = {"BTC/USD", "ETH/USD"}

# interval = main candle, htf = higher timeframe used as trend filter
TF = {
    "1m":  dict(interval="1min",  secs=60,    htf="5min",  htf_secs=300),
    "5m":  dict(interval="5min",  secs=300,   htf="15min", htf_secs=900),
    "15m": dict(interval="15min", secs=900,   htf="1h",    htf_secs=3600),
    "30m": dict(interval="30min", secs=1800,  htf="1h",    htf_secs=3600),
    "1h":  dict(interval="1h",    secs=3600,  htf="4h",    htf_secs=14400),
}
# strictness = only act on the strongest X% of scores seen on this market/timeframe (adapts to every market)
STRICT = {"Balanced (more signals)": 0.88, "Strict (recommended)": 0.94, "Very strict (fewest signals)": 0.97}
OHLC = ["open", "high", "low", "close"]



# ============================================================
#  DATA, INDICATORS, SCORING ENGINE, BACKTEST, AI
# ============================================================

def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# ---------------------------- DATA ----------------------------
_cache = {}
INTERVAL_SECS = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "1h": 3600, "4h": 14400}


def fetch_candles(symbol, interval, size):
    """Download candles (oldest -> newest) from Twelve Data, in UTC."""
    key = (symbol, interval, size)
    secs = INTERVAL_SECS.get(interval, 60)
    hit = _cache.get(key)
    # re-use for max 15 s (saves free API credits) but NEVER across a candle boundary
    if hit and time.time() - hit[0] < 15 and int(hit[0] // secs) == int(time.time() // secs):
        return hit[1].copy()
    params = dict(symbol=symbol, interval=interval, outputsize=size,
                  timezone="UTC", order="ASC", apikey=TD_KEY)
    j = None
    for attempt in range(2):
        r = requests.get(TD_URL, params=params, timeout=30)
        try:
            j = r.json()
        except ValueError:
            raise RuntimeError(f"Twelve Data sent an unreadable reply (HTTP {r.status_code}).")
        if isinstance(j, dict) and j.get("values"):
            break
        msg = str(j.get("message", j)) if isinstance(j, dict) else str(j)
        low = msg.lower()
        if attempt == 0 and "minute" in low and ("credit" in low or "limit" in low):
            time.sleep(12)                              # free plan: 8 requests/minute, wait and retry once
            continue
        raise RuntimeError(f"Twelve Data: {msg}")
    else:
        raise RuntimeError("Twelve Data: rate limit reached, wait one minute and try again.")
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in OHLC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=OHLC).sort_values("datetime").reset_index(drop=True)
    df = df[["datetime"] + OHLC]
    _cache[key] = (time.time(), df.copy())
    return df


def drop_forming_candle(df, secs):
    """Remove the candle that is still being formed, so only CLOSED candles are analysed."""
    if len(df) and df["datetime"].iloc[-1] + pd.Timedelta(seconds=secs) > pd.Timestamp(now_utc()):
        return df.iloc[:-1].reset_index(drop=True)
    return df


def parse_pasted(text):
    """One candle per line: open, high, low, close (oldest first). Extra text/volume is tolerated."""
    rows = []
    for line in (text or "").splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) >= 4:
            o, h, l, c = [float(x) for x in nums[-4:]]
            if h >= max(o, c, l) - 1e-12 and l <= min(o, c, h) + 1e-12:
                rows.append((o, h, l, c))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=OHLC)
    df.insert(0, "datetime", pd.NaT)
    return df


# -------------------------- INDICATORS --------------------------
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df):
    d = df.copy().reset_index(drop=True)
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    for n in (9, 21, 50, 200):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()

    delta = c.diff()
    ag, al = rma(delta.clip(lower=0), 14), rma(-delta.clip(upper=0), 14)
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    d["rsi"] = rsi.fillna(50.0)

    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    ma, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    d["bb_mid"], d["bb_up"], d["bb_lo"] = ma, ma + 2 * sd, ma - 2 * sd
    d["bb_pct"] = (c - d["bb_lo"]) / (d["bb_up"] - d["bb_lo"]).replace(0, np.nan)
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / ma.replace(0, np.nan)

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k_raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["stoch_k"] = k_raw.rolling(3).mean()
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = rma(tr, 14)
    d["atr_avg"] = d["atr"].rolling(200, min_periods=60).mean()   # long baseline for "is the market quiet?"

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=d.index)
    d["plus_di"] = 100 * rma(plus_dm, 14) / d["atr"].replace(0, np.nan)
    d["minus_di"] = 100 * rma(minus_dm, 14) / d["atr"].replace(0, np.nan)
    dx = 100 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"]).replace(0, np.nan)
    d["adx"] = rma(dx, 14)

    rng = (h - l)
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    has = rng > 0
    d["doji"] = has & (body <= 0.1 * rng)
    d["hammer"] = has & (lower >= 2 * body) & (upper <= 0.3 * rng)
    d["shooting"] = has & (upper >= 2 * body) & (lower <= 0.3 * rng)
    d["bull_engulf"] = (c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())
    d["bear_engulf"] = (c < o) & (c.shift() > o.shift()) & (c <= o.shift()) & (o >= c.shift())
    return d


def htf_bias_series(main, htf, htf_secs, main_secs):
    """Higher-timeframe trend bias (-1..+1) for every main candle, using ONLY closed HTF candles (no look-ahead)."""
    h = add_indicators(htf)
    bias = (np.sign(h["ema9"] - h["ema21"]) * 0.35 + np.sign(h["ema21"] - h["ema50"]) * 0.35
            + np.sign(h["macd_hist"]) * 0.15 + np.sign(h["close"] - h["ema50"]) * 0.15)
    hb = pd.DataFrame({"avail": h["datetime"] + pd.Timedelta(seconds=htf_secs), "bias": bias.values})
    hb = hb.dropna().sort_values("avail")
    m = pd.DataFrame({"t": main["datetime"] + pd.Timedelta(seconds=main_secs)})
    out = pd.merge_asof(m, hb, left_on="t", right_on="avail", direction="backward")
    return out["bias"].fillna(0.0).to_numpy(dtype=float)


# --------------------------- SCORING ---------------------------
def compute_score(d, htf_bias=None):
    """Regime-adaptive confluence score from -100 (strong PUT) to +100 (strong CALL)."""
    f = lambda x: np.nan_to_num(np.asarray(x, dtype=float))
    c = d["close"]
    trend = (np.where(d["ema9"] > d["ema21"], 1, -1) * 8 + np.where(d["ema21"] > d["ema50"], 1, -1) * 7
             + np.where(d["macd_hist"] > 0, 1, -1) * 6 + f(np.sign(d["macd_hist"].diff().fillna(0))) * 4
             + np.where(c > d["ema50"], 1, -1) * 5
             + np.where(f(d["adx"]) > 20, f(np.sign(d["plus_di"] - d["minus_di"])), 0) * 5)
    rsi, k, dd, bp = f(d["rsi"]), f(d["stoch_k"]), f(d["stoch_d"]), np.asarray(d["bb_pct"], dtype=float)
    bp = np.where(np.isnan(bp), 0.5, bp)
    mean_rev = (np.select([rsi < 25, rsi < 32, rsi > 75, rsi > 68], [12, 7, -12, -7], 0)
                + np.where((k < 20) & (k > dd), 8, 0) + np.where((k > 80) & (k < dd), -8, 0)
                + np.select([bp < 0.05, bp < 0.15, bp > 0.95, bp > 0.85], [10, 5, -10, -5], 0))
    pat = (np.where(d["bull_engulf"] & (bp < 0.5), 8, 0) + np.where(d["hammer"] & (bp < 0.5), 6, 0)
           - np.where(d["bear_engulf"] & (bp > 0.5), 8, 0) - np.where(d["shooting"] & (bp > 0.5), 6, 0))
    adx = np.where(np.isnan(np.asarray(d["adx"], dtype=float)), 20.0, np.asarray(d["adx"], dtype=float))
    w_t = np.select([adx >= 25, adx < 20], [1.0, 0.5], 0.8)     # trending market: trust trend signals
    w_m = np.select([adx >= 25, adx < 20], [0.3, 1.0], 0.7)     # ranging market: trust reversal signals
    htf = 15 * np.asarray(htf_bias, dtype=float) if htf_bias is not None else 0.0
    score = w_t * trend + w_m * mean_rev + pat + htf
    return pd.Series(np.clip(score, -100, 100), index=d.index).round(1)


def explain_factors(d, htf_bias):
    r, p = d.iloc[-1], d.iloc[-2] if len(d) > 1 else d.iloc[-1]
    out = []
    out.append("EMA9 > EMA21 > EMA50: bullish stack" if r.ema9 > r.ema21 > r.ema50 else
               "EMA9 < EMA21 < EMA50: bearish stack" if r.ema9 < r.ema21 < r.ema50 else "EMAs mixed (no clean trend)")
    out.append(f"MACD histogram {'positive' if r.macd_hist > 0 else 'negative'} and "
               f"{'rising' if r.macd_hist > p.macd_hist else 'falling'}")
    out.append(f"RSI(14) = {r.rsi:.1f}" + (" (oversold)" if r.rsi < 30 else " (overbought)" if r.rsi > 70 else ""))
    if pd.notna(r.stoch_k):
        out.append(f"Stochastic K/D = {r.stoch_k:.0f}/{r.stoch_d:.0f}")
    if pd.notna(r.bb_pct):
        out.append(f"Bollinger position = {r.bb_pct * 100:.0f}% of band" +
                   (" (below lower band)" if r.bb_pct < 0 else " (above upper band)" if r.bb_pct > 1 else ""))
    if pd.notna(r.adx):
        out.append(f"ADX = {r.adx:.0f} ({'trending' if r.adx >= 25 else 'ranging' if r.adx < 20 else 'transition'})")
    if htf_bias is not None:
        out.append(f"Higher-timeframe bias = {htf_bias:+.2f} (-1 bearish ... +1 bullish)")
    for name, label in [("bull_engulf", "bullish engulfing"), ("bear_engulf", "bearish engulfing"),
                        ("hammer", "hammer / long lower wick"), ("shooting", "shooting star / long upper wick"),
                        ("doji", "doji (indecision)")]:
        if bool(r[name]):
            out.append("Pattern: " + label)
    return out


def swing_levels(d, lookback=200, w=4):
    t = d.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(w, len(t) - w):
        if t["high"].iloc[i] == t["high"].iloc[i - w:i + w + 1].max():
            highs.append(float(t["high"].iloc[i]))
        if t["low"].iloc[i] == t["low"].iloc[i - w:i + w + 1].min():
            lows.append(float(t["low"].iloc[i]))
    price = float(t["close"].iloc[-1])
    atr = d["atr"].iloc[-1]
    tol = float(atr) * 0.3 if pd.notna(atr) and atr > 0 else price * 0.0003

    def cluster(levels, reverse):
        out = []
        for v in sorted(levels, reverse=reverse):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out
    res = cluster([x for x in highs if x > price], False)[:3]
    sup = cluster([x for x in lows if x < price], True)[:3]
    return [round(x, 5) for x in sup], [round(x, 5) for x in res]


# --------------------------- BACKTEST ---------------------------
def wilson_lb(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((centre - adj) / den)


def thresholds(score):
    """Score needed per strictness level = the strongest X% of this market's own past scores (floor 25)."""
    fallback = {0.88: 30.0, 0.94: 40.0, 0.97: 48.0, 0.985: 55.0}
    arr = np.abs(score.to_numpy(dtype=float)[100:-1]) if len(score) >= 160 else np.array([])
    arr = arr[~np.isnan(arr)]
    if len(arr) < 60:
        return fallback
    return {q: round(float(max(25.0, np.quantile(arr, q))), 1) for q in fallback}


def backtest(d, score, payout, tmap):
    """Replay the SAME rules on past candles: signal on candle i, result = close of candle i+1 vs close of i."""
    n_all = len(d)
    if n_all < 200:
        return None
    c = d["close"].to_numpy(dtype=float)
    diff = np.append(c[1:] - c[:-1], 0.0)
    s = score.to_numpy(dtype=float)
    idx = np.arange(n_all)
    valid = (idx >= 100) & (idx < n_all - 1) & (diff != 0) & ~np.isnan(s)
    be = 1.0 / (1.0 + payout)
    rows = {}
    for q, thr in tmap.items():
        m = valid & (np.abs(s) >= thr)
        n = int(m.sum())
        w = int(((np.sign(s) == np.sign(diff)) & m).sum())
        wr = w / n if n else 0.0
        m2 = m & (idx >= n_all - 300)
        n2 = int(m2.sum())
        w2 = int(((np.sign(s) == np.sign(diff)) & m2).sum())
        rows[q] = dict(thr=thr, n=n, wins=w, wr=wr, lb=wilson_lb(w, n), ev=wr * payout - (1 - wr),
                       recent_n=n2, recent_wr=(w2 / n2 if n2 else 0.0))
    return dict(rows=rows, breakeven=be, candles=n_all, payout=payout)


def edge_status(bt, q):
    if bt is None:
        return "unknown", "Not enough candles for a backtest."
    r = bt["rows"].get(q)
    if r is None:
        return "unknown", "No backtest row for this strictness."
    if r["n"] < 30:
        return "unknown", f"Only {r['n']} past signals at this strictness - too few to judge."
    if r["wr"] >= bt["breakeven"] + 0.03:
        return "positive", f"Past win rate {r['wr']*100:.1f}% vs break-even {bt['breakeven']*100:.1f}% ({r['n']} trades)."
    if r["wr"] >= bt["breakeven"]:
        return "marginal", f"Past win rate {r['wr']*100:.1f}% is barely above break-even {bt['breakeven']*100:.1f}% ({r['n']} trades)."
    return "negative", f"Past win rate {r['wr']*100:.1f}% is BELOW break-even {bt['breakeven']*100:.1f}% ({r['n']} trades)."


# --------------------------- SESSIONS ---------------------------
def session_info(market, t):
    h = t.hour
    name = ("Asian" if h < 7 else "London" if h < 12 else "London/New York overlap" if h < 16
            else "New York" if h < 21 else "Late/rollover")
    if market in CRYPTO:
        return name, True, "Crypto trades 24/7."
    good = name in ("London", "London/New York overlap", "New York")
    if not good and any(x in market for x in ("JPY", "AUD", "NZD")) and name == "Asian":
        good = True
    note = "Good liquidity session." if good else "Thin/choppy session for this market - signals are less reliable."
    return name, good, note


# ------------------------------ AI ------------------------------
SYSTEM_PROMPT = """You are the head analyst of a professional trading desk with 20+ years in forex, gold and
short-term (scalping / binary options) price action. You receive PRE-COMPUTED numbers (indicators, a rule-based
confluence score, higher-timeframe bias, support/resistance, recent candles, and an honest backtest of the rules).
Use ALL of the following knowledge: market structure (HH/HL/LH/LL, break of structure), support/resistance and
supply/demand, candlestick patterns, trend-following vs mean-reversion by regime (ADX), momentum (RSI, MACD,
Stochastic), volatility (Bollinger squeeze/expansion, ATR), exhaustion and liquidity sweeps, session behaviour.
Decide whether the NEXT candle will close UP (CALL) or DOWN (PUT) versus where it opens.
RULES:
1. Never invent numbers; only use what is given (or visible in a screenshot).
2. Short timeframes are close to a coin flip. Say NO TRADE unless at least 3 independent factors agree.
3. Do not fight a strong higher-timeframe trend (ADX>25) with a counter-trend trade.
4. If the backtest shows the rules are below break-even, be extra strict and mention it.
5. Confidence is YOUR estimate of the setup quality from 0-100 (not a guaranteed probability). Rarely above 75.
6. "signal" may be NO TRADE, but "lean" must ALWAYS be CALL or PUT: your single best directional lean for the
   next candle even when the setup is weak (the trader sees it labelled as a low-quality lean).
7. Reply with ONLY this JSON (no markdown):
{"signal":"CALL|PUT|NO TRADE","lean":"CALL|PUT","confidence":0-100,"market_regime":"trending up|trending down|ranging|breakout|exhaustion",
"trend":"short text","support":[numbers],"resistance":[numbers],"patterns":["..."],
"reasoning":"3-6 sentences using the actual data","invalidation":"what would cancel this idea","risk_notes":"short"}"""


def parse_json(txt):
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("AI reply was not valid JSON")


def call_gemini(prompt, image=None):
    parts = [{"text": prompt}]
    if image is not None:
        img = image.convert("RGB")
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}})
    body = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    errors = []
    for model in GEMINI_MODELS:
        try:
            r = requests.post(GEMINI_URL.format(model=model), json=body, timeout=120,
                              headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"})
            j = r.json()
            if isinstance(j, dict) and j.get("candidates"):
                content = j["candidates"][0].get("content", {})
                txt = "".join(p.get("text", "") for p in content.get("parts", []))
                res = parse_json(txt)
                sig = str(res.get("signal", "NO TRADE")).upper().replace("_", " ").strip()
                res["signal"] = sig if sig in ("CALL", "PUT", "NO TRADE") else "NO TRADE"
                ln = str(res.get("lean", "")).upper().strip()
                res["lean"] = ln if ln in ("CALL", "PUT") else (res["signal"] if res["signal"] != "NO TRADE" else None)
                try:
                    res["confidence"] = int(max(0, min(100, float(res.get("confidence", 0)))))
                except Exception:
                    res["confidence"] = 0
                return res, model
            err = (j.get("error", {}) or {}).get("message", str(j)) if isinstance(j, dict) else str(j)
            errors.append(f"{model}: {err[:160]}")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:160]}")
    raise RuntimeError(" | ".join(errors))


def clean(o):
    if o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if np.isnan(v) else round(v, 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return o.strftime("%Y-%m-%d %H:%M")
    return o



# ============================================================
#  THE AGENT (analysis flow, chart, results tracker)
# ============================================================
HIST_COLS = ["time (UTC)", "market", "type", "tf", "signal", "grade", "confidence", "result"]


def hist_frame(hist):
    if not hist:
        return pd.DataFrame(columns=HIST_COLS)
    return pd.DataFrame([[h["time"], h["market"], h["type"], h["tf"], h["signal"], h.get("grade", ""),
                          ("" if h.get("conf") is None else h["conf"]), h["result"]]
                         for h in hist], columns=HIST_COLS)


def stats_text(hist):
    done = [h for h in hist if h["result"] in ("Win", "Loss")]
    if not done:
        return "No results recorded yet. After each trade press **Win** or **Loss** to track your real win rate."
    w = sum(1 for h in done if h["result"] == "Win")
    n = len(done)
    streak = 0
    for h in reversed(done):
        if h["result"] == "Loss":
            streak += 1
        else:
            break
    txt = f"**Your record:** {w} wins / {n - w} losses = **{w / n * 100:.0f}%** win rate ({n} trades)."
    parts = []
    for g in ("A", "B", "C"):
        gd = [h for h in done if h.get("grade") == g]
        if gd:
            gw = sum(1 for h in gd if h["result"] == "Win")
            parts.append(f"Grade {g}: {gw}/{len(gd)} ({gw / len(gd) * 100:.0f}%)")
    if parts:
        txt += "  \n**By grade:** " + "  ·  ".join(parts)
    if streak >= 3:
        txt += f"\n\n🛑 **{streak} losses in a row - stop trading for this session.** Come back with a clear head."
    elif streak == 2:
        txt += "\n\n⚠️ 2 losses in a row. Be careful - one more loss means stop."
    return txt


def save_log(hist):
    try:
        if hist:
            hist_frame(hist).to_csv(LOG_PATH, index=False)
    except Exception:
        pass


def record_result(result, hist):
    hist = list(hist or [])
    for h in reversed(hist):
        if h["result"] == "Pending":
            h["result"] = result
            save_log(hist)
            break
    return hist_frame(hist), stats_text(hist), hist


def record_win(hist):
    return record_result("Win", hist)


def record_loss(hist):
    return record_result("Loss", hist)


def make_chart(d, levels, title):
    t = d.tail(90).reset_index(drop=True)
    fig = Figure(figsize=(10, 6.4), facecolor="#0e1117")
    gs = fig.add_gridspec(4, 1, hspace=0.08)
    ax = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex=ax)
    for a in (ax, ax2):
        a.set_facecolor("#0e1117")
        a.tick_params(colors="#9aa4b2", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a2f3a")
    x = np.arange(len(t))
    up = (t["close"] >= t["open"]).to_numpy()
    colors = ["#26a69a" if u else "#ef5350" for u in up]
    ax.vlines(x, t["low"], t["high"], colors=colors, linewidth=1)
    height = np.maximum((t["close"] - t["open"]).abs().to_numpy(), (t["high"] - t["low"]).to_numpy() * 0.03)
    ax.bar(x, height, bottom=np.minimum(t["open"], t["close"]).to_numpy(), color=colors, width=0.6)
    ax.plot(x, t["ema9"], color="#ffd54f", linewidth=1, label="EMA9")
    ax.plot(x, t["ema21"], color="#4fc3f7", linewidth=1, label="EMA21")
    ax.plot(x, t["ema50"], color="#ba68c8", linewidth=1, label="EMA50")
    ax.fill_between(x, t["bb_lo"], t["bb_up"], color="#90a4ae", alpha=0.10)
    sup, res = levels
    for v in sup:
        ax.axhline(v, color="#26a69a", linestyle="--", linewidth=0.8, alpha=0.7)
    for v in res:
        ax.axhline(v, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
    lo, hi = float(t["low"].min()), float(t["high"].max())
    pad = (hi - lo) * 0.05 or 1e-6
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="upper left", fontsize=7, facecolor="#0e1117", labelcolor="#9aa4b2", edgecolor="#2a2f3a")
    ax.tick_params(labelbottom=False)
    ax2.plot(x, t["rsi"], color="#ffb74d", linewidth=1)
    ax2.axhline(70, color="#ef5350", linewidth=0.6, linestyle=":")
    ax2.axhline(30, color="#26a69a", linewidth=0.6, linestyle=":")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="#9aa4b2", fontsize=8)
    if t["datetime"].notna().all():
        step = max(1, len(t) // 8)
        ticks = list(range(0, len(t), step))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([t["datetime"].iloc[i].strftime("%H:%M") for i in ticks], fontsize=8)
    fig.suptitle(title, color="white", fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fmt_levels(v):
    try:
        return ", ".join(str(x) for x in v) if v else "-"
    except Exception:
        return str(v)


def run_analysis(market, mtype, tf, strict_label, image, pasted, balance, risk_pct, payout_pct, require_edge,
                 always_lean=True):
    cfg = TF[tf]
    payout = float(payout_pct or 85) / 100.0
    q = STRICT[strict_label]
    thr_base = 35.0
    balance = float(balance or 0)
    risk = float(risk_pct or 2)
    now = now_utc()
    live = (mtype == "Live Market")
    df, bias_arr, notes, gates = None, None, [], []
    stale = False

    # ---------- 1) get data ----------
    if live:
        if not TD_KEY:
            raise RuntimeError("Twelve Data key is missing. Add TWELVE_DATA_API_KEY in the app's Secrets (Streamlit: Manage app -> Settings -> Secrets), then reboot the app.")
        df = drop_forming_candle(fetch_candles(market, cfg["interval"], 1500), cfg["secs"])
        if len(df) < 120:
            raise RuntimeError(f"Only {len(df)} candles received for {market} {tf} - not enough to analyse.")
        try:
            htf = drop_forming_candle(fetch_candles(market, cfg["htf"], 800), cfg["htf_secs"])
            bias_arr = htf_bias_series(df, htf, cfg["htf_secs"], cfg["secs"])
        except Exception as e:
            bias_arr = None
            notes.append(f"Higher-timeframe data unavailable ({str(e)[:90]}); trend filter skipped.")
        age = (now - df["datetime"].iloc[-1]).total_seconds() - cfg["secs"]
        if age > 3 * cfg["secs"] + 300:
            stale = True
    else:
        df = parse_pasted(pasted)
        n_rows = 0 if df is None else len(df)
        if image is None and n_rows < 20:
            raise RuntimeError("OTC needs a chart screenshot (best: 60+ candles visible) or at least 40 pasted candles "
                               "(one per line: open, high, low, close - oldest first). "
                               + ("I could not read any candles from the text you pasted." if (pasted or "").strip() and n_rows == 0
                                  else f"I could read only {n_rows} candles."))
        if df is not None and n_rows < 40:
            notes.append(f"Only {n_rows} pasted candles (40+ needed for indicators) - the AI reads them as raw data.")
            df = None

    # ---------- 2) quant engine ----------
    d = last = bt = None
    sc, thr, quant_sig, htf_now = 0.0, thr_base, "NO TRADE", None
    sup, res, factors = [], [], []
    sname, sgood, snote = "n/a", True, ""
    edge_state, edge_text = "unknown", ""
    if df is not None:
        d = add_indicators(df)
        score = compute_score(d, bias_arr)
        last = d.iloc[-1]
        sc = float(score.iloc[-1])
        htf_now = float(bias_arr[-1]) if bias_arr is not None else None
        tmap = thresholds(score)
        thr_base = tmap[q]
        bt = backtest(d, score, payout, tmap)
        sup, res = swing_levels(d)
        factors = explain_factors(d, htf_now)
        if live:
            sname, sgood, snote = session_info(market, now)
        thr = thr_base + (5 if (live and not sgood) else 0)
        quant_sig = "CALL" if sc >= thr else "PUT" if sc <= -thr else "NO TRADE"
        edge_state, edge_text = edge_status(bt, q)

        # safety filters
        if stale:
            gates.append("Data is old - the market looks closed (weekend/holiday) or the feed is delayed.")
        a, a_avg = last["atr"], last["atr_avg"]
        if pd.notna(a) and pd.notna(a_avg) and a_avg > 0 and a / a_avg < 0.6:
            gates.append("Market is too quiet (ATR far below normal) - price moves are mostly random noise.")
        if pd.notna(a) and a > 0 and (last["high"] - last["low"]) > 2.5 * a:
            gates.append("The last candle was an extreme spike (over 2.5x ATR) - exhaustion risk, wait for it to settle.")
        if (htf_now is not None and pd.notna(last["adx"]) and last["adx"] >= 25 and abs(htf_now) >= 0.7
                and quant_sig != "NO TRADE" and np.sign(htf_now) != (1 if quant_sig == "CALL" else -1)):
            gates.append("Signal is against a strong higher-timeframe trend (counter-trend scalps lose more often).")
        if require_edge and edge_state == "negative":
            gates.append("Backtest check failed: " + edge_text)

    # ---------- 3) AI analyst ----------
    ai, ai_model, ai_err = None, None, None
    if df is not None and gates and (stale or not always_lean):
        ai_err = "skipped (a safety filter already blocked this trade)"
    elif not GEMINI_KEY:
        ai_err = "Gemini key missing"
    else:
        if df is not None:
            ctx = dict(
                market=market, market_type=mtype, timeframe=tf, utc_now=now.strftime("%Y-%m-%d %H:%M:%S"),
                session=sname if live else "unknown (OTC)", last_closed_candle=df["datetime"].iloc[-1] if live else None,
                price=last["close"],
                indicators=dict(ema9=last["ema9"], ema21=last["ema21"], ema50=last["ema50"], ema200=last["ema200"],
                                rsi14=last["rsi"], macd=last["macd"], macd_signal=last["macd_sig"], macd_hist=last["macd_hist"],
                                bb_upper=last["bb_up"], bb_mid=last["bb_mid"], bb_lower=last["bb_lo"], bb_position=last["bb_pct"],
                                stoch_k=last["stoch_k"], stoch_d=last["stoch_d"], atr14=last["atr"], atr_avg_long=last["atr_avg"],
                                adx=last["adx"], plus_di=last["plus_di"], minus_di=last["minus_di"]),
                higher_timeframe=dict(interval=cfg["htf"], bias=htf_now),
                quant_engine=dict(score=sc, entry_threshold=thr, signal=quant_sig, factors=factors),
                support=sup, resistance=res,
                backtest_of_these_rules=(None if bt is None else dict(
                    breakeven_winrate=bt["breakeven"], status=edge_state, detail=edge_text,
                    by_strictness={f"top_{(1 - k) * 100:g}pct_score_{v['thr']}": dict(trades=v["n"], winrate=v["wr"],
                                                                                recent300_winrate=v["recent_wr"])
                                   for k, v in bt["rows"].items()})),
                last_15_candles=d.tail(15)[["datetime"] + OHLC].to_dict("records"),
                warnings_triggered=gates, notes=notes)
            prompt = "Analyze this and decide the NEXT candle:\n" + json.dumps(clean(ctx), indent=1)
        else:
            prompt = ("Market: %s, OTC (broker-generated prices, no external data), timeframe %s. No computed indicators "
                      "are available. Analyze the attached chart screenshot and/or these raw candles (oldest first, "
                      "open/high/low/close):\n%s\nDecide the NEXT candle." % (market, tf, (pasted or "")[:3000]))
        try:
            ai, ai_model = call_gemini(prompt, image)
        except Exception as e:
            ai_err = str(e)[:300]

    # ---------- 4) final decision ----------
    # Step 4a: strict "confirmed" decision (rule engine AND AI agree, all filters pass) = Grade A
    final, conf, reasons, grade = "NO TRADE", 0, [], ""
    if df is not None:
        if gates:
            reasons += gates
        elif ai is None:
            final = quant_sig
            if final != "NO TRADE":
                grade, conf = "B", int(min(70, abs(sc) * 1.2))
            reasons.append(f"AI analyst unavailable ({ai_err}) - decision uses the quant engine only.")
            if final == "NO TRADE":
                reasons.append(f"Quant score {sc:+.0f} is inside the no-trade zone (needs +/-{thr}).")
        else:
            a_sig = ai["signal"]
            if quant_sig == "NO TRADE":
                reasons.append(f"Quant score {sc:+.0f} is below the confirmation level (needs +/-{thr})."
                               + (f" The AI leaned {a_sig}." if a_sig != "NO TRADE" else ""))
            elif a_sig == "NO TRADE":
                reasons.append("The AI analyst did not confirm the rule-engine signal.")
            elif a_sig != quant_sig:
                reasons.append(f"Quant engine says {quant_sig} but AI says {a_sig} - they disagree.")
            elif ai["confidence"] < 60:
                reasons.append(f"AI confidence is only {ai['confidence']}% (needs 60+ for Grade A).")
            else:
                final, grade = a_sig, "A"
                conf = int(min(85, round(0.5 * min(100, abs(sc) * 1.4) + 0.5 * ai["confidence"])))
                if edge_state == "negative":
                    grade = "B"
                    reasons.append("Downgraded to B because the backtest is below break-even: " + edge_text)
    else:
        if ai is None:
            reasons.append(f"AI analyst unavailable ({ai_err}).")
        elif ai["signal"] != "NO TRADE" and ai["confidence"] >= 65:
            final, grade, conf = ai["signal"], "B", int(min(70, ai["confidence"]))
            reasons.append("AI-only analysis (no numeric indicators or backtest) - treat with extra caution.")
        else:
            reasons.append(f"AI saw no clear setup (signal {ai['signal']}, confidence {ai['confidence']}%).")

    # Step 4b: best-guess mode - always give a direction (graded B or C) unless the market is closed
    if final == "NO TRADE" and always_lean and not stale:
        q_dir = 0
        if df is not None:
            q_dir = 1 if sc > 0 else -1 if sc < 0 else 0
            if q_dir == 0:
                q_dir = 1 if float(last["close"]) >= float(last["ema9"]) else -1
        lean = (ai or {}).get("lean") or (ai or {}).get("signal")
        a_dir = {"CALL": 1, "PUT": -1}.get(lean, 0)
        if q_dir and a_dir and q_dir == a_dir:
            direction, grade = q_dir, ("C" if gates else "B")
        elif q_dir and a_dir:
            direction, grade = (q_dir if abs(sc) >= 0.6 * thr else a_dir), "C"
        else:
            direction, grade = (q_dir or a_dir), "C"
        if direction:
            final = "CALL" if direction > 0 else "PUT"
            conf = None
            reasons.append("Rule engine and AI " + ("agree on the direction" if (q_dir and a_dir and q_dir == a_dir)
                                                   else "do not fully agree (or only one source was available)")
                           + ", but not strongly enough for a confirmed Grade A signal.")

    # ---------- 5) entry timing ----------
    when, late = "", False
    if live and df is not None and final != "NO TRADE":
        t_open = df["datetime"].iloc[-1] + pd.Timedelta(seconds=cfg["secs"])
        elapsed = (now - t_open.to_pydatetime()).total_seconds()
        if elapsed <= max(15, 0.25 * cfg["secs"]):
            when = f"⏱️ **Enter NOW** - the candle opened {max(0, int(elapsed))}s ago. Expiry: {tf}."
        else:
            late = True
            n = int(elapsed // cfg["secs"]) + 1
            nxt = t_open + pd.Timedelta(seconds=n * cfg["secs"])
            when = (f"⏱️ This candle started {int(elapsed)}s ago - **too late to enter**. Press Generate again shortly "
                    f"before **{nxt:%H:%M:%S} UTC** and enter at the open. Expiry: {tf}.")

    # ---------- 6) build the report ----------
    title = {"CALL": "🟢 CALL ▲  (UP)", "PUT": "🔴 PUT ▼  (DOWN)", "NO TRADE": "⚪ NO TRADE - skip this one"}[final]
    grade_txt = {"A": "Grade A - confirmed signal", "B": "Grade B - moderate (use a smaller stake)",
                 "C": "Grade C - weak lean, close to a coin flip (small stake or skip)"}
    md = [f"# {title}", f"**{market}** · {mtype} · {tf} candles · {now:%H:%M:%S} UTC"]
    if final != "NO TRADE":
        md.append(f"### {grade_txt[grade]}")
        if grade == "A":
            md.append(f"**Confidence: {conf}%**  \n_Setup-quality estimate - not a guaranteed win chance._")
        if when:
            md.append(when)
        mult = {"A": 1.0, "B": 0.5, "C": 0.25}[grade]
        stake = balance * risk / 100.0 * mult
        md.append(f"💰 **Suggested stake:** {stake:.2f} ({risk:g}% of {balance:.0f}"
                  + ("" if mult == 1 else f", scaled x{mult:g} for grade {grade}") + f")  ·  break-even win rate at "
                  f"{payout * 100:.0f}% payout is {100 / (1 + payout):.1f}%")
    if reasons:
        label = "**Why no trade:** " if final == "NO TRADE" else ("**Notes:** " if grade == "A" else "**Why not Grade A:** ")
        md.append(label + " ".join(reasons))
    if stale:
        md.append("💡 This market looks **closed** right now (forex and gold close at the weekend). Try **BTC/USD or ETH/USD** "
                  "(live, 24/7) or switch to **OTC** and upload a screenshot.")
    for n_ in notes:
        md.append(f"ℹ️ {n_}")

    if df is not None:
        line = f"**Quant score:** {sc:+.0f} (needs ±{thr}) → {quant_sig}"
        if ai:
            line += f"  ·  **AI:** {ai['signal']} ({ai['confidence']}%)"
        if htf_now is not None:
            line += f"  ·  **{cfg['htf']} trend bias:** {htf_now:+.2f}"
        md.append(line)
        if live:
            md.append(f"🕒 **Session:** {sname}. {snote}")
    if ai:
        md.append("---\n### 🧠 AI analyst")
        md.append(f"**Regime:** {ai.get('market_regime', '-')}  \n**Trend:** {ai.get('trend', '-')}")
        if ai.get("patterns"):
            md.append(f"**Patterns:** {fmt_levels(ai.get('patterns'))}")
        md.append(f"**Reasoning:** {ai.get('reasoning', '-')}")
        md.append(f"**Invalidation:** {ai.get('invalidation', '-')}")
        md.append(f"**Risk:** {ai.get('risk_notes', '-')}")
        md.append(f"_Model used: {ai_model}_")
    if df is not None:
        md.append("---\n### 📊 Key levels & factors")
        md.append(f"**Support:** {fmt_levels(sup)}  \n**Resistance:** {fmt_levels(res)}")
        md.append("\n".join(f"- {f_}" for f_ in factors))
    if bt is not None:
        md.append("---\n### 🧪 Backtest of these exact rules on this market (past data)")
        rows = ["| Strictness | Score needed | Past trades | Win rate | Safe low estimate | Last 300 candles |",
                "|---|---|---|---|---|---|"]
        for k, v in bt["rows"].items():
            mark = " ◀ yours" if k == q else ""
            rows.append(f"| top {(1 - k) * 100:g}% of scores{mark} | ±{v['thr']:g} | {v['n']} | {v['wr'] * 100:.1f}% | "
                        f"{v['lb'] * 100:.1f}% | {v['recent_wr'] * 100:.1f}% |")
        md.append("\n".join(rows))
        md.append(f"Break-even win rate: **{bt['breakeven'] * 100:.1f}%** (payout {payout * 100:.0f}%). "
                  f"{edge_text}  \n_Past results do not guarantee future results; this is the honest check on whether the "
                  f"rules have worked on this market and timeframe._")
    md.append("---\n*Educational tool only - not financial advice. Binary options carry a high risk of losing your whole "
              "stake. Test on a demo account first.*")

    chart = None
    if d is not None:
        try:
            chart = make_chart(d, (sup, res), f"{market} {tf} - score {sc:+.0f} - {final}")
        except Exception as e:
            md.append(f"ℹ️ Chart could not be drawn: {str(e)[:100]}")
    entry = None
    if final != "NO TRADE" and not late:
        entry = dict(time=now.strftime("%Y-%m-%d %H:%M:%S"), market=market, type="Live" if live else "OTC",
                     tf=tf, signal=final, grade=grade, conf=conf, result="Pending")
    return "\n\n".join(md), chart, entry


def analyze_market(market, mtype, tf, strict_label, image, pasted, balance, risk_pct, payout_pct, require_edge, hist,
                   always_lean=True):
    hist = list(hist or [])
    try:
        md, chart, entry = run_analysis(market, mtype, tf, strict_label, image, pasted,
                                        balance, risk_pct, payout_pct, require_edge, always_lean)
        if entry:
            for h_ in hist:
                if h_["result"] == "Pending":
                    h_["result"] = "Skipped"
            hist.append(entry)
            save_log(hist)
        return md, chart, hist_frame(hist), stats_text(hist), hist
    except Exception as e:
        msg = (f"# ⚠️ Could not generate a signal\n\n**{e}**\n\n"
               "Common fixes: check that both keys are saved in the app's Secrets, wait a minute if you hit the free "
               "request limit, try crypto (BTC/USD) if it is the weekend, and for OTC upload a clear screenshot.")
        return msg, None, hist_frame(hist), stats_text(hist), hist

