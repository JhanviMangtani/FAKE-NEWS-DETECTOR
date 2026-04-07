"""
Fake News Detector - Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import os
from Fake_news_detector import load_data, train_and_evaluate, predict, preprocess_text

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── CSS Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 700; }
    .subtitle   { text-align: center; color: #666; margin-bottom: 2rem; }
    .fake-box   { background: #FEE2E2; border-left: 6px solid #EF4444;
                  border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }
    .real-box   { background: #DCFCE7; border-left: 6px solid #22C55E;
                  border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }
    .metric-box { background: #F1F5F9; border-radius: 8px;
                  padding: 1rem; text-align: center; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by TF-IDF + Machine Learning</div>', unsafe_allow_html=True)

# ── Load / Train Model ───────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models...")
def get_model():
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    df = load_data()
    model, results = train_and_evaluate(df)
    return model, results

model, results = get_model()

# ── Sidebar: Model Info ──────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")

    if results:
        for name, res in results.items():
            acc = res['accuracy']
            color = "🟢" if acc > 0.90 else "🟡" if acc > 0.80 else "🔴"
            st.write(f"{color} **{name}**")
            st.progress(float(acc))
            st.caption(f"Accuracy: {acc:.1%}")
    else:
        st.info("Model loaded from cache")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. Text is cleaned & normalized
    2. TF-IDF converts text to numbers
    3. ML model classifies as REAL/FAKE
    4. Confidence score is returned
    """)
    st.markdown("**Dataset:** ISOT Fake News Dataset (Kaggle)")

# ── Main Input ───────────────────────────────────────────────
st.markdown("### Paste a news article or headline:")

example_fake = ("BREAKING: Government HIDING secret cure for cancer that "
                "elite scientists discovered. Mainstream media is covering "
                "it up. Wake up sheeple! Whistleblower reveals truth.")

example_real = ("The Federal Reserve raised interest rates by 25 basis points "
                "on Wednesday, as policymakers continued efforts to bring "
                "inflation back to the 2% target. The decision was unanimous.")

col1, col2 = st.columns(2)
with col1:
    if st.button("📌 Load fake example"):
        st.session_state['article_text'] = example_fake
with col2:
    if st.button("📌 Load real example"):
        st.session_state['article_text'] = example_real

article = st.text_area(
    "Article text:",
    value=st.session_state.get('article_text', ''),
    height=180,
    placeholder="Paste news article text here..."
)

if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
    if not article.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(article, model)

        # ── Result Display ───────────────────────────────────
        if result['label'] == 'FAKE':
            st.markdown(f"""
            <div class="fake-box">
                <h2>🚨 FAKE NEWS DETECTED</h2>
                <p>Confidence: <strong>{result['confidence']:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="real-box">
                <h2>✅ LIKELY REAL</h2>
                <p>Confidence: <strong>{result['confidence']:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("#### Probability Breakdown")
        col_f, col_r = st.columns(2)
        with col_f:
            st.metric("FAKE probability", f"{result['fake_probability']:.1%}")
            st.progress(float(result['fake_probability']))
        with col_r:
            st.metric("REAL probability", f"{result['real_probability']:.1%}")
            st.progress(float(result['real_probability']))

        # Top keywords
        st.markdown("#### Top Keywords That Influenced This Prediction")
        cols = st.columns(5)
        for i, kw in enumerate(result['top_keywords']):
            cols[i].markdown(f"`{kw}`")

        # Preprocessed text
        with st.expander("🔎 See preprocessed text"):
            st.text(preprocess_text(article))

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Built for hackathon | scikit-learn + Streamlit | ISOT Dataset")