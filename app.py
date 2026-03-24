import streamlit as st
import numpy as np
import joblib
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,179,237,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 80% 80%, rgba(154,104,255,0.08) 0%, transparent 60%);
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 680px; }

/* ── Hero Header ──────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    animation: fadeDown 0.7s ease both;
}
.hero .badge {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.15;
    color: #f0f4ff;
    margin: 0 0 0.75rem;
    letter-spacing: -0.02em;
}
.hero h1 span {
    background: linear-gradient(135deg, #63b3ed 0%, #9a68ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #6b7fa3;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
    line-height: 1.6;
}

/* ── Card ─────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(12px);
    animation: fadeUp 0.6s ease both;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5f80;
    margin-bottom: 1.5rem;
}

/* ── Grade Sliders Label ──────────────────────────── */
.grade-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.3rem;
}
.grade-label .name {
    font-size: 0.9rem;
    font-weight: 500;
    color: #a8bcd8;
}
.grade-label .range {
    font-size: 0.75rem;
    color: #3d5070;
}

/* ── Streamlit widget overrides ───────────────────── */
div[data-testid="stNumberInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
div[data-testid="stNumberInput"] > div:focus-within {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
}
div[data-testid="stNumberInput"] input {
    color: #e8f0ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-align: center;
}
div[data-testid="stNumberInput"] > div {
    background: #000000 !important;   /* pure black */
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}

div[data-testid="stNumberInput"] input {
    color: #000000 !important;   /* white text for contrast */
}

/* ── Slider overrides ─────────────────────────────── */
div[data-testid="stSlider"] > div > div > div[role="slider"] {
    background: #63b3ed !important;
    border: 2px solid #fff !important;
}
div[data-testid="stSlider"] .stSlider > div {
    color: #6b7fa3 !important;
}

/* ── Button ───────────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.25) !important;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.4) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Cards ─────────────────────────────────── */
.result-wrap {
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
.result-score {
    text-align: center;
    padding: 2rem 1rem;
    border-radius: 18px;
    border: 1px solid;
    margin-bottom: 1rem;
}
.result-score.pass {
    background: rgba(52,211,153,0.07);
    border-color: rgba(52,211,153,0.25);
}
.result-score.fail {
    background: rgba(248,113,113,0.07);
    border-color: rgba(248,113,113,0.25);
}
.result-score .score-num {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-score.pass  .score-num { color: #34d399; }
.result-score.fail  .score-num { color: #f87171; }
.result-score .score-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    opacity: 0.5;
}
.result-score.pass .score-label { color: #34d399; }
.result-score.fail .score-label { color: #f87171; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    margin-top: 0.75rem;
}
.status-pill.pass {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
}
.status-pill.fail {
    background: rgba(248,113,113,0.15);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.3);
}

/* ── Grade Bar ────────────────────────────────────── */
.grade-bar-wrap {
    margin-top: 1.25rem;
}
.grade-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}
.grade-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.grade-bar-fill.pass { background: linear-gradient(90deg, #34d399, #059669); }
.grade-bar-fill.fail { background: linear-gradient(90deg, #f87171, #dc2626); }
.grade-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-size: 0.72rem;
    color: #3d5070;
}

/* ── Metric Chips ─────────────────────────────────── */
.metrics-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
}
.metric-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
}
.metric-chip .chip-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8f0ff;
    line-height: 1;
}
.metric-chip .chip-lbl {
    font-size: 0.7rem;
    color: #3d5070;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Divider ──────────────────────────────────────── */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Animations ───────────────────────────────────── */
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes popIn {
    from { opacity:0; transform:scale(0.9); }
    to   { opacity:1; transform:scale(1); }
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl"), True
    except Exception:
        return None, False

model, model_loaded = load_model()


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">AI · Deep Learning · Education</div>
    <h1>Student <span>Grade</span><br>Predictor</h1>
    <p>Enter previous grades to predict the<br>final period score (G3) instantly.</p>
</div>
""", unsafe_allow_html=True)


# ─── Input Card ───────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📋 Student Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=10, max_value=25, value=17, step=1)
with col2:
    g1 = st.number_input("G1 — Period 1", min_value=0, max_value=20, value=12, step=1)
with col3:
    g2 = st.number_input("G2 — Period 2", min_value=0, max_value=20, value=13, step=1)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Grade trend indicator
if g2 > g1:
    trend_text = f"📈 Improving  +{g2-g1} pts from G1 → G2"
    trend_color = "#34d399"
elif g2 < g1:
    trend_text = f"📉 Declining  {g2-g1} pts from G1 → G2"
    trend_color = "#f87171"
else:
    trend_text = "➡️ Stable  No change G1 → G2"
    trend_color = "#63b3ed"

st.markdown(
    f'<p style="font-size:0.85rem;color:{trend_color};margin:0 0 1rem;">{trend_text}</p>',
    unsafe_allow_html=True
)

predict_btn = st.button("⚡  Predict Final Grade (G3)")
st.markdown('</div>', unsafe_allow_html=True)


# ─── Result ───────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(""):
        time.sleep(0.4)   # short dramatic pause

    # ── Predict ──────────────────────────────────────────────────────────────
    if model_loaded:
        raw = model.predict(np.array([[g1, g2]]))[0]
        prediction = int(round(float(raw)))
        prediction = max(0, min(20, prediction))
    else:
        # Demo fallback: simple weighted average when no model.pkl present
        prediction = int(round(0.35 * g1 + 0.65 * g2))
        prediction = max(0, min(20, prediction))
        st.markdown(
            '<p style="font-size:0.78rem;color:#4a5f80;text-align:center;margin-bottom:0.5rem;">'
            '⚠️ model.pkl not found — showing demo prediction</p>',
            unsafe_allow_html=True
        )

    status      = "pass" if prediction >= 10 else "fail"
    status_icon = "✅" if status == "pass" else "❌"
    status_text = "PASS" if status == "pass" else "FAIL"
    pct         = int((prediction / 20) * 100)

    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)

    # Score card
    st.markdown(f"""
    <div class="result-score {status}">
        <div class="score-num">{prediction}<span style="font-size:1.8rem;opacity:0.4"> /20</span></div>
        <div class="score-label">Predicted G3 Score</div>
        <div style="margin-top:0.6rem;">
            <span class="status-pill {status}">{status_icon} &nbsp;{status_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grade bar
    st.markdown(f"""
    <div class="grade-bar-wrap">
        <div class="grade-bar-track">
            <div class="grade-bar-fill {status}" style="width:{pct}%"></div>
        </div>
        <div class="grade-bar-labels">
            <span>0</span>
            <span>Pass threshold → 10</span>
            <span>20</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric chips
    avg_input = round((g1 + g2) / 2, 1)
    delta     = prediction - avg_input
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-chip">
            <div class="chip-val">{g1}</div>
            <div class="chip-lbl">G1 Score</div>
        </div>
        <div class="metric-chip">
            <div class="chip-val">{g2}</div>
            <div class="chip-lbl">G2 Score</div>
        </div>
        <div class="metric-chip">
            <div class="chip-val">{avg_input}</div>
            <div class="chip-lbl">G1–G2 Avg</div>
        </div>
        <div class="metric-chip">
            <div class="chip-val">{delta_str}</div>
            <div class="chip-lbl">vs Avg</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Advice ───────────────────────────────────────────────────────────────
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    if status == "pass":
        if prediction >= 16:
            advice = "🌟 Excellent performance! Keep up the outstanding work."
        elif prediction >= 13:
            advice = "👍 Good result. A little more effort could push you to excellent."
        else:
            advice = "✔️ Just passing. Consistent study habits will improve the score."
    else:
        if prediction >= 7:
            advice = "⚠️ Close to passing. Focus on weak areas and seek extra support."
        else:
            advice = "🚨 Significant improvement needed. Consider tutoring or extra study sessions."

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:14px;padding:1rem 1.25rem;font-size:0.88rem;color:#7a92b8;line-height:1.6;">'
        f'{advice}</div>',
        unsafe_allow_html=True
    )
