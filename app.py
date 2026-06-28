import streamlit as st

st.set_page_config(page_title="HEART AI", page_icon="❤️", layout="wide")

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
    margin: 80px auto 0 auto;
    max-width: 520px;
    background: rgba(255,255,255,0.07);
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
    font-size: 3.2rem;
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
    font-size: 1.3rem;
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
</style>

<div class="premium-card">
    <div class="premium-title">❤️ HEART AI</div>
    <div class="premium-subtitle">A Futuristic Clinical Intelligence Dashboard<br>by <b>padmeswarbasumatary2@gmail.com</b></div>
    <button class="premium-btn" onclick="window.location.href='mailto:padmeswarbasumatary2@gmail.com'">Contact Me</button>
</div>
""", unsafe_allow_html=True)
