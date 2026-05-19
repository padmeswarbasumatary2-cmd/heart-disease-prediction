import streamlit as st
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="HEART AI",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/simple_heart_model.joblib")

# -----------------------------
# ADVANCED CSS
# -----------------------------
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background:
        radial-gradient(circle at top left, #1e3a8a 0%, transparent 25%),
        radial-gradient(circle at bottom right, #7e22ce 0%, transparent 25%),
        linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
    overflow-x: hidden;
}

/* REMOVE STREAMLIT */
header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* GRID BACKGROUND */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: -1;
}

/* MAIN TITLE */
.main-title {
    text-align: center;
    font-size: 78px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: glow 3s infinite alternate;
}

/* GLOW */
@keyframes glow {
    from {
        filter: drop-shadow(0 0 10px rgba(56,189,248,0.4));
    }

    to {
        filter: drop-shadow(0 0 25px rgba(192,132,252,0.8));
    }
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 50px;
    letter-spacing: 1px;
}

/* FLOATING GLASS CARD */
.glass-card {

    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.1);

    backdrop-filter: blur(18px);

    border-radius: 32px;

    padding: 40px;

    box-shadow:
        0 10px 40px rgba(0,0,0,0.5),
        0 0 30px rgba(99,102,241,0.15);

    transition: all 0.4s ease;

    animation: floatCard 6s ease-in-out infinite;
}

/* FLOAT ANIMATION */
@keyframes floatCard {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-8px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* LABELS */
label {
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* INPUTS */
.stNumberInput input {

    background: rgba(255,255,255,0.05) !important;

    border: 1px solid rgba(255,255,255,0.12) !important;

    border-radius: 16px !important;

    color: white !important;

    height: 52px;

    box-shadow: 0 0 15px rgba(99,102,241,0.15);

    transition: 0.3s;
}

.stNumberInput input:focus {
    border: 1px solid #60a5fa !important;
    box-shadow: 0 0 20px rgba(96,165,250,0.6);
}

/* SELECTBOX */
.stSelectbox div[data-baseweb="select"] {

    background: rgba(255,255,255,0.05) !important;

    border-radius: 16px !important;

    border: 1px solid rgba(255,255,255,0.12) !important;

    box-shadow: 0 0 15px rgba(99,102,241,0.15);
}

/* BUTTON */
.stButton > button {

    width: 100%;

    height: 75px;

    border-radius: 20px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #06b6d4,
            #6366f1,
            #8b5cf6
        );

    color: white;

    font-size: 24px;

    font-weight: 800;

    letter-spacing: 1px;

    transition: all 0.4s ease;

    box-shadow:
        0 0 25px rgba(99,102,241,0.4),
        0 0 50px rgba(139,92,246,0.2);
}

/* BUTTON HOVER */
.stButton > button:hover {

    transform: scale(1.03);

    box-shadow:
        0 0 40px rgba(99,102,241,0.8),
        0 0 80px rgba(139,92,246,0.5);
}

/* RESULT BOX */
.result-box {

    margin-top: 40px;

    padding: 40px;

    border-radius: 30px;

    text-align: center;

    font-size: 34px;

    font-weight: 900;

    animation: fadeIn 0.7s ease;
}

/* HIGH RISK */
.high-risk {

    background: rgba(255,0,0,0.08);

    border: 2px solid rgba(255,0,0,0.4);

    color: #ffb4b4;

    box-shadow:
        0 0 40px rgba(255,0,0,0.25);
}

/* LOW RISK */
.low-risk {

    background: rgba(0,255,120,0.08);

    border: 2px solid rgba(0,255,120,0.35);

    color: #bbffd1;

    box-shadow:
        0 0 40px rgba(0,255,120,0.18);
}

/* RESULT FADE */
@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">❤️ HEART AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Futuristic Clinical Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

# -----------------------------
# FLOATING PANEL
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("👤 Age", 1, 120, 45)

    sex = st.selectbox(
        "⚧ Gender",
        ["Male", "Female"]
    )

    sex_value = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "💓 Chest Pain Type",
        [0,1,2,3]
    )

    trestbps = st.number_input(
        "🩸 Resting Blood Pressure",
        80,250,120
    )

    chol = st.number_input(
        "🧪 Cholesterol Level",
        100,600,200
    )

    fbs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120",
        ["No","Yes"]
    )

    fbs_value = 1 if fbs == "Yes" else 0

with col2:

    restecg = st.selectbox(
        "📈 Rest ECG",
        [0,1,2]
    )

    thalach = st.number_input(
        "❤️ Maximum Heart Rate",
        60,250,150
    )

    exang = st.selectbox(
        "🏃 Exercise Induced Angina",
        ["No","Yes"]
    )

    exang_value = 1 if exang == "Yes" else 0

    oldpeak = st.number_input(
        "📉 Oldpeak",
        0.0,10.0,1.0
    )

    slope = st.selectbox(
        "📊 Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "🫀 Major Vessels",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "🧬 Thal",
        [0,1,2,3]
    )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# BUTTON
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🚀 ANALYZE HEART RISK")

# -----------------------------
# PREDICTION
# -----------------------------
if predict:

    input_data = np.array([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs_value,
        restecg,
        thalach,
        exang_value,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    # 0 = Disease
    # 1 = No Disease

    if prediction[0] == 0:

        st.markdown(f"""
        <div class="result-box high-risk">
            ⚠️ HIGH RISK OF HEART DISEASE<br><br>
            Risk Probability: {(1 - probability):.2f}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-box low-risk">
            ✅ LOW RISK OF HEART DISEASE<br><br>
            Safety Probability: {probability:.2f}
        </div>
        """, unsafe_allow_html=True)