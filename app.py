import streamlit as st
import utils
from datetime import datetime

# Set config explicitly
st.set_page_config(
    page_title="Visio AI | Enterprise Platform",
    page_icon="assets/images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load dependencies safely
try:
    utils.init_session_state()
    utils.load_css()
except Exception as e:
    st.error(f"Initialization Error: {e}")

# --- Custom Workspace Header ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 24px; border-bottom: 1px solid #E2E8F0; margin-bottom: 32px;">
        <div>
            <h1 style="margin: 0;">Visio AI <span style="font-weight: 300; color: #94A3B8;">Workspace</span></h1>
            <p style="margin: 0; color: #64748B;">Enterprise Data Science Environment</p>
        </div>
        <div style="text-align: right;">
            <div style="font-weight: 600; color: #0F172A;">JAIHO LABS</div>
            <div style="font-size: 13px; color: #64748B;">v2.1.0 • Stable</div>
        </div>
    </div>
""", unsafe_allow_html=True)




# === Hero Section ===
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Welcome to Visio AI</div>
    <div class="hero-subtitle">Enterprise Data Science & Machine Learning Platform</div>
</div>
""", unsafe_allow_html=True)

# === Value Proposition Section (Replaces Metrics) ===
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    <div class="metric-card">
        <h3>🚀 What is Visio AI?</h3>
        <p>A <strong>No-Code AI Platform</strong> that allows you to clean data, visualize trends, and build Machine Learning models (like Random Forest & XGBoost) without writing a single line of code. It runs 100% locally on your machine.</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="metric-card">
        <h3>🌱 Why use it?</h3>
        <p><strong>Sustainability & Privacy.</strong> Unlike cloud AI that consumes massive energy and requires uploading data, Visio AI uses efficient algorithms on your own CPU. Your data never leaves your computer.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# --- Quick Launch Grid (Custom HTML Buttons) ---
st.markdown("<h2 style='margin-top: 40px;'>Launchpad</h2>", unsafe_allow_html=True)

# We use Streamlit columns but render custom HTML buttons inside them. 
# Since we can't easily capture click events from raw HTML to simple streamlit logic without components,
# We will design them to look like cards but stick to Streamlit buttons for functionality, 
# OR use styling on st.button (which we did in styles.css).

# Let's use the styled st.buttons as "Action Cards"

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown("### 📥 Ingestion")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("📂 Data Loader\n\nImport CSV/Excel", use_container_width=True): 
        st.switch_page("pages/1_Data_Loader.py")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("☁️ Word Cloud\n\nText Analysis", use_container_width=True): 
        st.switch_page("pages/9_Word_Cloud.py")

with col_b:
    st.markdown("### 📊 Analytics")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("📈 Explore (EDA)\n\nVisual Analysis", use_container_width=True): 
        st.switch_page("pages/2_EDA.py")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("🎨 Viz Builder\n\nCustom Plots", use_container_width=True): 
        st.switch_page("pages/10_Viz_Builder.py")

with col_c:
    st.markdown("### 🤖 Intelligence")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("🧠 Model Training\n\nSupervised ML", use_container_width=True): 
        st.switch_page("pages/3_Supervised.py")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("🔮 Clustering\n\nUnsupervised ML", use_container_width=True): 
        st.switch_page("pages/4_Unsupervised.py")

with col_d:
    st.markdown("### 🛠️ Utilities")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("📝 Notepad\n\nSession Notes", use_container_width=True): 
        st.switch_page("pages/8_Notes.py")
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    if st.button("📑 Report\n\nGenerate PDF", use_container_width=True): 
        st.switch_page("pages/7_Report.py")


# --- Sidebar Upgrade ---
utils.sidebar_nav()
st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip**: Use 'AutoML' for quick model benchmarking.")
