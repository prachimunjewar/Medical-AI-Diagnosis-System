import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from utils.eda import (
    load_data, get_basic_stats, plot_outcome_distribution,
    plot_feature_distributions, plot_correlation_heatmap, plot_feature_vs_outcome,
)
from utils.ml_model import train_and_save_all, load_model, predict_diabetes
from utils.report_generator import generate_medical_report




# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0F1E;
    color: #E2E8F0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1530 0%, #0A0F1E 100%);
    border-right: 1px solid #1E2D4A;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] .stRadio label {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 6px;
    display: block;
    transition: all 0.2s;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #3B82F6;
    background: #1a2540;
}
.main .block-container {
    background-color: #0A0F1E;
    padding: 2rem 3rem;
    max-width: 1200px;
}
.hero-banner {
    background: linear-gradient(135deg, #0D1B3E 0%, #1a2a5e 50%, #0D1B3E 100%);
    border: 1px solid #1E3A6E;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.hero-sub { color: #94A3B8; font-size: 1rem; margin: 0; }
.hero-tags { display: flex; gap: 10px; margin-top: 1.2rem; flex-wrap: wrap; }
.hero-tag {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93C5FD;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #F1F5F9;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E2D4A;
}
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 1.5rem; }
.stat-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1E2D4A;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #3B82F6; }
.stat-label { font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #60A5FA; line-height: 1; }
.stat-sub { font-size: 0.75rem; color: #475569; margin-top: 4px; }
.info-card { background: #111827; border: 1px solid #1E2D4A; border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem; }
.info-card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.5rem; }
.result-positive { background: linear-gradient(135deg, #1a0a0a, #2d1515); border: 2px solid #EF4444; border-radius: 16px; padding: 1.5rem 2rem; margin: 1rem 0; }
.result-negative { background: linear-gradient(135deg, #0a1a0f, #0f2d1a); border: 2px solid #10B981; border-radius: 16px; padding: 1.5rem 2rem; margin: 1rem 0; }
.result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin: 0 0 4px; }
.result-sub { font-size: 0.9rem; opacity: 0.8; margin: 0; }
.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1rem 0; }
.metric-card { background: #111827; border: 1px solid #1E2D4A; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.metric-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #60A5FA; }
.metric-label { font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
.step-list { counter-reset: step; list-style: none; padding: 0; margin: 0; }
.step-item { counter-increment: step; display: flex; gap: 14px; align-items: flex-start; margin-bottom: 14px; }
.step-item::before {
    content: counter(step);
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: #111827 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}
.report-box {
    background: #0D1530;
    border: 1px solid #1E3A6E;
    border-radius: 14px;
    padding: 2rem;
    line-height: 1.8;
    color: #CBD5E1;
    font-size: 0.95rem;
}
.tumor-card {
    background: linear-gradient(135deg, #1a0f2e, #2d1a4a);
    border: 2px solid #8B5CF6;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.disclaimer {
    background: #111827;
    border-left: 3px solid #F59E0B;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 1rem;
}
.db-stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 1.5rem; }
.sql-badge {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6EE7B7;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: monospace;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/diabetes.csv"

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🧬</div>
        <div style='font-family: Space Grotesk; font-size:1.1rem; font-weight:700; color:#60A5FA;'>MedAI Diagnostics</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>AI-Powered Medical Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊  Data Analysis", "🤖  Train Models", "🔬  Patient Diagnosis", "🧠  Brain Tumor MRI", "🗄️  Patient Records"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1px; background:#1E2D4A; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; padding: 0 0.5rem;'>
        <div style='color:#64748B; font-weight:600; margin-bottom:8px; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;'>Tech Stack</div>
        <div style='margin-bottom:5px;'>⚡ XGBoost · Random Forest</div>
        <div style='margin-bottom:5px;'>🧠 ResNet50 Transfer Learning</div>
        <div style='margin-bottom:5px;'>🤖 Groq LLaMA 3.3 70B</div>
        <div style='margin-bottom:5px;'>📊 SMOTE · GridSearchCV</div>
        <div style='margin-bottom:5px;'>🗄️ SQLite · SQL Queries</div>
        <div>🚀 Streamlit · Plotly</div>
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE 1: EDA ─────────────────────────────────────────────────────────────
if "Data Analysis" in page:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧬 Data Analysis</div>
        <p class="hero-sub">Exploratory analysis of the Pima Indians Diabetes Dataset — 768 patient records, 8 clinical features</p>
        <div class="hero-tags">
            <span class="hero-tag">EDA</span>
            <span class="hero-tag">Feature Engineering</span>
            <span class="hero-tag">Statistical Analysis</span>
            <span class="hero-tag">Plotly Visualizations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">📁 Upload Dataset</div>
            <p style='color:#64748B; font-size:0.9rem;'>Download the Pima Indians Diabetes dataset from Kaggle and upload it here.</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload diabetes.csv", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            os.makedirs("data", exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            st.success("Dataset saved successfully!")
            st.rerun()
        st.stop()

    df = load_data(DATA_PATH)
    stats = get_basic_stats(df)

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-label">Total Records</div><div class="stat-value">{stats['total_records']}</div><div class="stat-sub">Patient entries</div></div>
        <div class="stat-card"><div class="stat-label">Features</div><div class="stat-value">{stats['features']}</div><div class="stat-sub">Clinical variables</div></div>
        <div class="stat-card"><div class="stat-label">Diabetic</div><div class="stat-value" style='color:#EF4444;'>{stats['diabetic']}</div><div class="stat-sub">{round(stats['diabetic']/stats['total_records']*100,1)}% of total</div></div>
        <div class="stat-card"><div class="stat-label">Non-Diabetic</div><div class="stat-value" style='color:#10B981;'>{stats['non_diabetic']}</div><div class="stat-sub">{round(stats['non_diabetic']/stats['total_records']*100,1)}% of total</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 View Raw Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_outcome_distribution(df)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_correlation_heatmap(df)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
        st.plotly_chart(fig, use_container_width=True)

    fig = plot_feature_distributions(df)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_feature_vs_outcome(df)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
    st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 2: TRAIN ───────────────────────────────────────────────────────────
elif "Train Models" in page:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🤖 Model Training</div>
        <p class="hero-sub">Train XGBoost and Random Forest classifiers with SMOTE oversampling and GridSearchCV hyperparameter tuning</p>
        <div class="hero-tags">
            <span class="hero-tag">XGBoost</span><span class="hero-tag">Random Forest</span>
            <span class="hero-tag">SMOTE</span><span class="hero-tag">GridSearchCV</span><span class="hero-tag">Cross Validation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.warning("Please upload the dataset in the Data Analysis page first.")
        st.stop()

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">⚙️ Training Pipeline</div>
        <ol class="step-list">
            <li class="step-item"><div><strong>SMOTE</strong> — Synthetic oversampling to balance classes</div></li>
            <li class="step-item"><div><strong>Sklearn Pipeline</strong> — StandardScaler + Classifier</div></li>
            <li class="step-item"><div><strong>GridSearchCV</strong> — 5-fold cross validation for best hyperparameters</div></li>
            <li class="step-item"><div><strong>Evaluation</strong> — Accuracy, F1, Precision, Recall, ROC-AUC</div></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    df = load_data(DATA_PATH)

    if st.button("🚀 Start Training — XGBoost + Random Forest"):
        with st.spinner("Training with SMOTE + GridSearchCV... (~1-2 minutes)"):
            results = train_and_save_all(df)
            st.session_state["ml_results"] = results
        st.success("✅ Models trained and saved to models/ folder!")

    if "ml_results" in st.session_state:
        results = st.session_state["ml_results"]
        rf = results["rf"]["metrics"]
        xgb = results["xgb"]["metrics"]

        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-card"><div class="info-card-title">🌲 Random Forest</div></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val" style='color:#10B981;'>{rf['accuracy']}%</div><div class="metric-label">Accuracy</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#60A5FA;'>{rf['f1_score']}%</div><div class="metric-label">F1 Score</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#A78BFA;'>{rf['roc_auc']}%</div><div class="metric-label">ROC AUC</div></div>
            </div>
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val" style='color:#F59E0B;'>{rf['precision']}%</div><div class="metric-label">Precision</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#F472B6;'>{rf['recall']}%</div><div class="metric-label">Recall</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#34D399;'>5-fold</div><div class="metric-label">CV Folds</div></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="info-card"><div class="info-card-title">⚡ XGBoost</div></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val" style='color:#10B981;'>{xgb['accuracy']}%</div><div class="metric-label">Accuracy</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#60A5FA;'>{xgb['f1_score']}%</div><div class="metric-label">F1 Score</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#A78BFA;'>{xgb['roc_auc']}%</div><div class="metric-label">ROC AUC</div></div>
            </div>
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val" style='color:#F59E0B;'>{xgb['precision']}%</div><div class="metric-label">Precision</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#F472B6;'>{xgb['recall']}%</div><div class="metric-label">Recall</div></div>
                <div class="metric-card"><div class="metric-val" style='color:#34D399;'>5-fold</div><div class="metric-label">CV Folds</div></div>
            </div>
            """, unsafe_allow_html=True)

        metrics = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
        labels = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Random Forest", x=labels, y=[rf[m] for m in metrics], marker_color="#10B981"))
        fig.add_trace(go.Bar(name="XGBoost", x=labels, y=[xgb[m] for m in metrics], marker_color="#60A5FA"))
        fig.update_layout(
            title="Model Comparison", barmode="group", yaxis_title="Score (%)", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
            legend=dict(bgcolor="rgba(0,0,0,0)"), yaxis=dict(gridcolor="#1E2D4A"),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔧 Best Hyperparameters (GridSearchCV)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Random Forest**")
                st.json(results["rf"]["params"])
            with col2:
                st.markdown("**XGBoost**")
                st.json(results["xgb"]["params"])


# ─── PAGE 3: PATIENT DIAGNOSIS ────────────────────────────────────────────────
elif "Patient Diagnosis" in page:
    from utils.database import init_db, save_patient
    init_db()
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔬 Patient Diagnosis</div>
        <p class="hero-sub">Enter patient clinical data to get diabetes risk prediction and a full AI-generated medical report</p>
        <div class="hero-tags">
            <span class="hero-tag">Risk Prediction</span><span class="hero-tag">Groq LLaMA 3.3</span>
            <span class="hero-tag">Medical Report</span><span class="hero-tag">SQLite Storage</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    models_exist = os.path.exists("models/xgboost.pkl") and os.path.exists("models/random_forest.pkl")
    if not models_exist:
        st.warning("⚠️ Please train the ML models first in the 'Train Models' page.")
        st.stop()

    xgb_model = load_model("xgboost.pkl")
    rf_model = load_model("random_forest.pkl")

    tumor_result = st.session_state.get("tumor_result", None)
    if tumor_result:
        st.markdown(f"""
        <div class="tumor-card">
            <div style='font-size:0.75rem; color:#A78BFA; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>Brain MRI Result Loaded</div>
            <div style='font-family: Space Grotesk; font-size:1.1rem; font-weight:600; color:#E2E8F0;'>🧠 {tumor_result['label']} — {tumor_result['confidence']}% confidence</div>
            <div style='font-size:0.82rem; color:#94A3B8; margin-top:4px;'>This will be included in the medical report</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Patient Clinical Data</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (years)", 1, 120, 30)

    patient_data = {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }

    model_choice = st.selectbox("Select ML Model", ["XGBoost", "Random Forest"])
    model = xgb_model if model_choice == "XGBoost" else rf_model

    if st.button("🔬 Run Diagnosis & Generate AI Report"):
        with st.spinner("Analysing patient data..."):
            ml_result = predict_diabetes(model, patient_data)
            ml_result["model"] = model_choice

        if ml_result["prediction"] == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title" style='color:#EF4444;'>⚠️ {ml_result['label']}</div>
                <p class="result-sub" style='color:#FCA5A5;'>Risk Probability: <strong>{ml_result['probability']}%</strong> — Model: {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title" style='color:#10B981;'>✅ {ml_result['label']}</div>
                <p class="result-sub" style='color:#6EE7B7;'>Risk Probability: <strong>{ml_result['probability']}%</strong> — Model: {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)

        # save to SQLite
        save_patient(patient_data, ml_result, tumor_result)
        st.success("✅ Patient record saved to database (SQLite)")

        st.markdown('<div class="section-header">AI Generated Medical Report</div>', unsafe_allow_html=True)
        with st.spinner("Groq LLaMA 3.3 generating report..."):
            report = generate_medical_report(patient_data, ml_result, tumor_result)

        st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="disclaimer">
            ⚠️ This report is AI-generated for educational purposes only and is not a substitute for professional medical advice.
        </div>
        """, unsafe_allow_html=True)

        st.download_button("📥 Download Report", data=report, file_name="medical_report.txt", mime="text/plain")


# ─── PAGE 4: BRAIN TUMOR MRI ─────────────────────────────────────────────────
elif "Brain Tumor" in page:
    from utils.cnn_model import load_cnn_model, predict_tumor, load_class_names
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧠 Brain Tumor Detection</div>
        <p class="hero-sub">Upload a brain MRI scan for AI-powered 4-class classification using ResNet50 Transfer Learning</p>
        <div class="hero-tags">
            <span class="hero-tag">ResNet50</span><span class="hero-tag">Transfer Learning</span>
            <span class="hero-tag">Glioma</span><span class="hero-tag">Meningioma</span>
            <span class="hero-tag">Pituitary</span><span class="hero-tag">No Tumor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cnn_model = load_cnn_model()

    if not cnn_model:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🔧 CNN Model Not Trained Yet</div>
            <ol class="step-list">
                <li class="step-item"><div>Download <strong>Brain Tumor MRI Dataset</strong> from <a href='https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset' target='_blank' style='color:#60A5FA;'>Kaggle</a></div></li>
                <li class="step-item"><div>Extract to <code style='background:#1E2D4A; padding:2px 6px; border-radius:4px;'>data/brain_tumor/Training/</code> with 4 class folders</div></li>
                <li class="step-item"><div>Run <code style='background:#1E2D4A; padding:2px 6px; border-radius:4px;'>python train_cnn.py</code> in terminal</div></li>
                <li class="step-item"><div>Refresh this page — upload will appear automatically</div></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    uploaded = st.file_uploader("Upload a Brain MRI Image (.jpg / .png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        from PIL import Image
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="section-header">Uploaded MRI Scan</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        with col2:
            st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)
            with st.spinner("ResNet50 analysing MRI scan..."):
                result = predict_tumor(cnn_model, image)

            color = "#EF4444" if result["is_tumor"] else "#10B981"
            border_class = "result-positive" if result["is_tumor"] else "result-negative"
            icon = "⚠️" if result["is_tumor"] else "✅"
            st.markdown(f"""
            <div class="{border_class}">
                <div class="result-title" style='color:{color};'>{icon} {result['label']}</div>
                <p class="result-sub" style='color:{"#FCA5A5" if result["is_tumor"] else "#6EE7B7"};'>
                    Confidence: <strong>{result['confidence']}%</strong>
                </p>
                <p style='font-size:0.85rem; color:#94A3B8; margin-top:8px;'>{result['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            probs = result["all_probabilities"]
            fig = go.Figure(go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=["#EF4444" if k != "No Tumor" else "#10B981" for k in probs.keys()],
                text=[f"{v}%" for v in probs.values()],
                textposition="outside", textfont=dict(color="#CBD5E1"),
            ))
            fig.update_layout(
                title="Class Probabilities", yaxis_title="Probability (%)", height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
                yaxis=dict(gridcolor="#1E2D4A"), margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.session_state["tumor_result"] = result
        st.markdown("""
        <div class="disclaimer">
            🧠 MRI result saved. Go to <strong>Patient Diagnosis</strong> page to include it in the full AI medical report.
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 5: PATIENT RECORDS (SQL) ───────────────────────────────────────────
elif "Patient Records" in page:
     from utils.database import (
        init_db, get_all_patients, get_filtered_patients,
        get_summary_stats, delete_patient, clear_all_patients
    )
    init_db()
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🗄️ Patient Records</div>
        <p class="hero-sub">All diagnosis records stored in SQLite database — filter, query, and analyse historical patient data</p>
        <div class="hero-tags">
            <span class="hero-tag">SQLite</span><span class="hero-tag">SQL Queries</span>
            <span class="hero-tag">INSERT · SELECT · WHERE · DELETE</span><span class="hero-tag">Aggregations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_summary_stats()

    if stats["total"] == 0:
        st.info("No patient records yet. Run a diagnosis in the Patient Diagnosis page first.")
        st.stop()

    # summary stats from SQL
    st.markdown('<div class="section-header">Database Summary <span class="sql-badge">SQL: COUNT · AVG</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-label">Total Patients</div><div class="stat-value">{stats['total']}</div><div class="stat-sub">Records in DB</div></div>
        <div class="stat-card"><div class="stat-label">Diabetic</div><div class="stat-value" style='color:#EF4444;'>{stats['diabetic']}</div><div class="stat-sub">{round(stats['diabetic']/stats['total']*100,1) if stats['total'] else 0}% of total</div></div>
        <div class="stat-card"><div class="stat-label">Non-Diabetic</div><div class="stat-value" style='color:#10B981;'>{stats['non_diabetic']}</div><div class="stat-sub">{round(stats['non_diabetic']/stats['total']*100,1) if stats['total'] else 0}% of total</div></div>
        <div class="stat-card"><div class="stat-label">MRI Scans</div><div class="stat-value" style='color:#A78BFA;'>{stats['mri_scans']}</div><div class="stat-sub">Brain tumor analyses</div></div>
    </div>
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-label">Avg Glucose</div><div class="stat-value">{stats['avg_glucose']}</div><div class="stat-sub">mg/dL</div></div>
        <div class="stat-card"><div class="stat-label">Avg Age</div><div class="stat-value">{stats['avg_age']}</div><div class="stat-sub">Years</div></div>
        <div class="stat-card"><div class="stat-label">Avg BMI</div><div class="stat-value">{stats['avg_bmi']}</div><div class="stat-sub">Body mass index</div></div>
        <div class="stat-card"><div class="stat-label">Database</div><div class="stat-value" style='font-size:1rem; padding-top:0.4rem;'>SQLite</div><div class="stat-sub">Local storage</div></div>
    </div>
    """, unsafe_allow_html=True)

    # filters
    st.markdown('<div class="section-header">Filter Records <span class="sql-badge">SQL: SELECT · WHERE · BETWEEN</span></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        outcome_filter = st.selectbox("Outcome", ["All", "Diabetic", "Non-Diabetic"])
    with col2:
        age_range = st.slider("Age Range", 1, 120, (1, 120))
    with col3:
        glucose_range = st.slider("Glucose Range (mg/dL)", 0, 300, (0, 300))

    df = get_filtered_patients(
        outcome=outcome_filter,
        age_min=age_range[0], age_max=age_range[1],
        glucose_min=glucose_range[0], glucose_max=glucose_range[1],
    )

    st.markdown(f"<p style='color:#64748B; font-size:0.85rem;'>Showing <strong style='color:#60A5FA;'>{len(df)}</strong> records matching filters</p>", unsafe_allow_html=True)

    if not df.empty:
        display_cols = ["id", "timestamp", "age", "glucose", "bmi", "prediction", "risk_probability", "model_used", "brain_tumor_result"]
        st.dataframe(df[display_cols].rename(columns={
            "id": "ID", "timestamp": "Time", "age": "Age",
            "glucose": "Glucose", "bmi": "BMI",
            "prediction": "Outcome", "risk_probability": "Risk %",
            "model_used": "Model", "brain_tumor_result": "MRI Result",
        }), use_container_width=True)

        # charts
        st.markdown('<div class="section-header">Visual Analytics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            outcome_counts = df["prediction"].value_counts()
            fig = go.Figure(go.Pie(
                labels=outcome_counts.index, values=outcome_counts.values,
                marker=dict(colors=["#EF4444", "#10B981"]),
            ))
            fig.update_layout(
                title="Outcome Distribution", height=300,
                paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x="age", y="glucose", color="prediction",
                color_discrete_map={"Diabetic": "#EF4444", "Non-Diabetic": "#10B981"},
                title="Age vs Glucose by Outcome", size="bmi",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1", yaxis=dict(gridcolor="#1E2D4A"),
                xaxis=dict(gridcolor="#1E2D4A"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # download
        csv = df.to_csv(index=False)
        st.download_button("📥 Export Records as CSV", data=csv, file_name="patient_records.csv", mime="text/csv")

    # danger zone
    st.markdown("<div style='height:1px; background:#1E2D4A; margin:2rem 0;'></div>", unsafe_allow_html=True)
    with st.expander("⚠️ Danger Zone — Delete Records"):
        col1, col2 = st.columns(2)
        with col1:
            del_id = st.number_input("Delete record by ID", min_value=1, step=1)
            if st.button("🗑️ Delete This Record"):
                delete_patient(int(del_id))
                st.success(f"Record {del_id} deleted.")
                st.rerun()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚨 Clear ALL Records"):
                clear_all_patients()
                st.success("All records cleared.")
                st.rerun()
