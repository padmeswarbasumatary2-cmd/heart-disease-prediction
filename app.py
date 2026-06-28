import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="HEART AI", page_icon="❤️", layout="wide")

# -----------------------------
# Premium CSS
# -----------------------------
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #0f2027, #2c5364 80%);
    min-height: 100vh;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    color: #fff;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}
.premium-card {
    margin: 60px auto 0 auto;
    max-width: 600px;
    background: rgba(255,255,255,0.08);
    border-radius: 32px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 60px #38bdf8;
    padding: 48px 36px;
    backdrop-filter: blur(18px);
    border: 1.5px solid rgba(255,255,255,0.13);
    animation: floatCard 6s ease-in-out infinite;
    transition: box-shadow 0.5s cubic-bezier(.4,2,.6,1), transform 0.5s cubic-bezier(.4,2,.6,1);
}
.premium-card:hover {
    box-shadow: 0 16px 80px #818cf8, 0 0 120px #c084fc;
    transform: scale(1.03) translateY(-8px);
}
@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}
.premium-title {
    font-size: 2.7rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 18px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s infinite alternate;
}
@keyframes glow {
    from { filter: drop-shadow(0 0 10px #38bdf8); }
    to { filter: drop-shadow(0 0 30px #c084fc); }
}
.premium-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #e0e7ef;
    margin-bottom: 36px;
    letter-spacing: 1px;
    font-weight: 500;
}
.premium-btn {
    display: block;
    margin: 0 auto;
    padding: 18px 48px;
    font-size: 1.2rem;
    font-weight: 700;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg, #06b6d4, #6366f1, #8b5cf6);
    color: #fff;
    box-shadow: 0 0 30px #818cf8;
    cursor: pointer;
    transition: background 0.4s, box-shadow 0.4s, transform 0.3s;
}
.premium-btn:hover {
    background: linear-gradient(90deg, #f472b6, #c084fc, #38bdf8);
    box-shadow: 0 0 60px #c084fc;
    transform: scale(1.07);
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    color: #fff !important;
    border: 1px solid #818cf8 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 0 10px #818cf8;
    margin-bottom: 10px;
}
.stNumberInput input:focus {
    border: 1.5px solid #38bdf8 !important;
    box-shadow: 0 0 20px #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model Loader
# -----------------------------
def load_model():
    model_path = "models/simple_heart_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<div class="premium-title">❤️ HEART AI</div>', unsafe_allow_html=True)
st.markdown('<div class="premium-subtitle">Premium Heart Disease Prediction System</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👤 Age", 1, 120, 45)
        sex = st.selectbox("⚧ Gender", ["Male", "Female"])
        cp = st.selectbox("💓 Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("🩸 Resting Blood Pressure", 80,250,120)
        chol = st.number_input("🧪 Cholesterol Level", 100,600,200)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", ["No","Yes"])
    with col2:
        restecg = st.selectbox("📈 Rest ECG", [0,1,2])
        thalach = st.number_input("❤️ Maximum Heart Rate", 60,250,150)
        exang = st.selectbox("🏃 Exercise Induced Angina", ["No","Yes"])
        oldpeak = st.number_input("📉 Oldpeak", 0.0,10.0,1.0)
        slope = st.selectbox("📊 Slope", [0,1,2])
        ca = st.selectbox("🫀 Major Vessels", [0,1,2,3,4])
        thal = st.selectbox("🧬 Thal", [0,1,2,3])
    submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

if submitted:
    if model is None:
        st.error("Model file not found. Please train and upload 'models/simple_heart_model.joblib'.")
    else:
        input_data = np.array([[age, 1 if sex=="Male" else 0, cp, trestbps, chol, 1 if fbs=="Yes" else 0,
                               restecg, thalach, 1 if exang=="Yes" else 0, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        if prediction[0] == 0:
            st.markdown(f"""
            <div class='premium-card' style='background:rgba(255,0,0,0.08);border:2px solid #ff4d4d;'>
                <div class='premium-title' style='background:linear-gradient(90deg,#ff4d4d,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>⚠️ HIGH RISK</div>
                <div class='premium-subtitle'>Probability: {(1-probability):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='premium-card' style='background:rgba(0,255,120,0.08);border:2px solid #38bdf8;'>
                <div class='premium-title' style='background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>✅ LOW RISK</div>
                <div class='premium-subtitle'>Probability: {probability:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
