import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}

.header-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #38bdf8;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}

.header-banner p {
    color: #8b949e;
    margin: 0;
    font-size: 0.95rem;
}

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* Result cards */
.result-yes {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}

.result-no {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #f87171;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}

.result-yes h2, .result-no h2 {
    font-size: 1.6rem;
    margin: 0.3rem 0;
}

.result-yes h2 { color: #34d399; }
.result-no  h2 { color: #f87171; }

.result-label {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
}

/* Metric card */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}

.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-card .label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

/* Input styling */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > input {
    background-color: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #075985);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.3);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}

hr {
    border-color: #30363d;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    saved       = load_model()
    model       = saved['model']
    threshold   = saved['threshold']
    preprocessor = saved['preprocessor']
    cap_values  = saved['cap_values']
    model_name  = saved.get('name', 'Gradient Boosting')
except FileNotFoundError:
    st.error("⚠️ `best_model.pkl` not found. Run the notebook save cell first.")
    st.stop()


# ── Preprocessing function ────────────────────────────────────────────────────
TRANSFORM_COLS = ['duration', 'campaign', 'pdays', 'previous']

def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Fix job label
    df['job'] = df['job'].replace('admin.', 'admin')

    # pdays: -1 means never contacted
    df['pdays'] = df['pdays'].replace(-1, 0)

    # Winsorize using trained cap values
    for col, bounds in cap_values.items():
        df[col] = df[col].clip(lower=bounds['lo'], upper=bounds['hi'])

    # log1p transform
    for col in TRANSFORM_COLS:
        df[col] = np.log1p(df[col])

    # Reorder columns to match training
    col_order = ['age', 'job', 'marital', 'education', 'default',
                 'balance', 'housing', 'loan', 'contact', 'day',
                 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    df = df[col_order]

    # Scale & encode using saved preprocessor
    transformed = preprocessor.transform(df)
    return pd.DataFrame(transformed, columns=col_order)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🏦 Bank Term Deposit Predictor</h1>
    <p>Predict whether a client will subscribe to a term deposit based on their profile and contact history.</p>
</div>
""", unsafe_allow_html=True)


# ── Layout: inputs (left) | result (right) ───────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Client Demographics ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Client Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age      = c1.number_input("Age", min_value=18, max_value=95, value=35)
    job      = c2.selectbox("Job", ['admin', 'blue-collar', 'entrepreneur', 'housemaid',
                                     'management', 'retired', 'self-employed', 'services',
                                     'student', 'technician', 'unemployed', 'unknown'])
    marital  = c3.selectbox("Marital Status", ['married', 'single', 'divorced'])

    c4, c5, c6 = st.columns(3)
    education = c4.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default   = c5.selectbox("Credit Default", ['no', 'yes'])
    balance   = c6.number_input("Account Balance (€)", min_value=-8000, max_value=102000, value=1000)

    c7, c8 = st.columns(2)
    housing = c7.selectbox("Housing Loan", ['no', 'yes'])
    loan    = c8.selectbox("Personal Loan", ['no', 'yes'])

    st.markdown('<div class="section-label">Contact Information</div>', unsafe_allow_html=True)
    c9, c10, c11 = st.columns(3)
    contact  = c9.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
    month    = c10.selectbox("Last Contact Month", ['jan','feb','mar','apr','may','jun',
                                                     'jul','aug','sep','oct','nov','dec'])
    day      = c11.number_input("Day of Month", min_value=1, max_value=31, value=15)
    duration = st.slider("Last Call Duration (seconds)", min_value=0, max_value=1300, value=200)

    st.markdown('<div class="section-label">Campaign History</div>', unsafe_allow_html=True)
    c12, c13 = st.columns(2)
    campaign = c12.number_input("Contacts This Campaign", min_value=1, max_value=63, value=2)
    pdays    = c13.number_input("Days Since Last Contact (-1 = never)", min_value=-1, max_value=871, value=-1)

    c14, c15 = st.columns(2)
    previous = c14.number_input("Previous Campaign Contacts", min_value=0, max_value=275, value=0)
    poutcome = c15.selectbox("Previous Outcome", ['unknown', 'success', 'failure', 'other'])

    predict_btn = st.button("⚡ Predict Subscription", use_container_width=True)


# ── Result panel ──────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        raw_input = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': float(balance), 'housing': housing,
            'loan': loan, 'contact': contact, 'day': day, 'month': month,
            'duration': float(duration), 'campaign': float(campaign),
            'pdays': float(pdays), 'previous': float(previous), 'poutcome': poutcome
        }

        try:
            X_input   = preprocess_input(raw_input)
            prob      = model.predict_proba(X_input)[0][1]
            prediction = int(prob >= threshold)

            # Result card
            if prediction == 1:
                st.markdown(f"""
                <div class="result-yes">
                    <div class="result-label">Prediction</div>
                    <h2>✅ Will Subscribe</h2>
                    <p style="color:#6ee7b7; margin:0">This client is likely to subscribe to a term deposit.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no">
                    <div class="result-label">Prediction</div>
                    <h2>❌ Will Not Subscribe</h2>
                    <p style="color:#fca5a5; margin:0">This client is unlikely to subscribe to a term deposit.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar
            st.markdown('<div class="section-label">Subscription Probability</div>', unsafe_allow_html=True)
            st.progress(float(prob))
            st.markdown(f"<div style='text-align:center; font-size:1.8rem; font-family:Syne; font-weight:700; color:#38bdf8'>{prob:.1%}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; color:#8b949e; font-size:0.8rem'>Decision threshold: {threshold:.2f}</div>", unsafe_allow_html=True)

            # Metrics row
            st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="value">0.60</div><div class="label">F1 Score</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="value">0.93</div><div class="label">ROC-AUC</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="value">{threshold:.2f}</div><div class="label">Threshold</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.markdown("""
        <div style="background:#161b22; border:1px dashed #30363d; border-radius:12px;
                    padding:3rem 2rem; text-align:center; color:#8b949e;">
            <div style="font-size:2.5rem; margin-bottom:1rem">🏦</div>
            <div style="font-family:Syne; font-size:1rem; font-weight:600; color:#8b949e">
                Fill in the client details<br>and click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model info box ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem 1.25rem;">
        <div style="font-size:0.7rem; letter-spacing:2px; color:#38bdf8; text-transform:uppercase; font-weight:700; margin-bottom:0.5rem">Active Model</div>
        <div style="font-family:Syne; font-weight:700; color:#e6edf3">{model_name}</div>
        <div style="color:#8b949e; font-size:0.82rem; margin-top:0.25rem">
            SMOTE · RobustScaler · TargetEncoder · Threshold-tuned
        </div>
    </div>
    """, unsafe_allow_html=True)
