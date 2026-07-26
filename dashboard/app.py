from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
from pdf_report import generate_pdf

# ==============================
# Project Paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from recommendation import generate_recommendations
from explain_ai import explain_prediction
from pdf_report import generate_pdf

# ==============================
# Load Model
# ==============================
model = joblib.load(BASE_DIR / "models" / "grade_change_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

risk_labels = {
    0: "Safe",
    1: "Warning",
    2: "Off-Spec"
}

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(
    page_title="Grade Change Intelligence",
    page_icon="🏭",
    layout="wide"
)



import time

if "startup_complete" not in st.session_state:
    st.session_state.startup_complete = False
    
if not st.session_state.startup_complete:

    st.markdown(
        """
    <h1 style="text-align:center;color:{TEXT};">
    🏭 Grade Change Intelligence
    </h1>
    """,
        unsafe_allow_html=True,
    )

    status = st.empty()
    progress = st.progress(0)

    steps = [
        ("Initializing AI Model...", 45),
        ("Loading XGBoost...", 70),
        ("Loading SHAP Engine...", 90),
        ("Preparing Dashboard...", 100),
    ]

    current = 0

    for message, target in steps:

        status.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:24px;
                color:#94A3B8;
                margin-top:30px;">
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

        while current < target:
            current += 1
            progress.progress(current)
            time.sleep(0.02)

        time.sleep(0.4)

    status.success("✅ System Ready")

    time.sleep(1)

    st.session_state.startup_complete = True

    st.rerun()

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>

#MainMenu,
header,
footer{
    visibility:hidden;
}

.stApp{
    background:#0B1120;
    color:white;
}

.block-container{
    max-width:1400px;
    padding-top:2rem;
}

.hero-title{
    font-size:46px;
    font-weight:800;
    color:white;
}

.hero-subtitle{
    font-size:18px;
    color:#94A3B8;
    margin-bottom:25px;
}

.kpi{
    background:linear-gradient(135deg,#172033,#1F2937);
    border:1px solid #2D3B50;
    border-radius:18px;
    padding:20px;
    text-align:center;
    margin-bottom:10px;
}

.kpi h4{
    color:#94A3B8;
    margin:0;
}

.kpi h2{
    color:white;
    margin-top:10px;
}

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(90deg,#2563EB,#0EA5E9);
    color:white;
}

.analysis-card{
    background:linear-gradient(135deg,#172033,#1E293B);
    border:1px solid #334155;
    border-radius:20px;
    padding:30px;
    margin-top:25px;
    box-shadow:0px 10px 30px rgba(0,0,0,.35);
}

.analysis-title{
    font-size:28px;
    font-weight:700;
    color:white;
    margin-bottom:25px;
}

.status-safe{
    color:#22C55E;
    font-size:36px;
    font-weight:bold;
}

.status-warning{
    color:#FACC15;
    font-size:36px;
    font-weight:bold;
}

.status-danger{
    color:#EF4444;
    font-size:36px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Session State
# ==============================
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "risk" not in st.session_state:
    st.session_state.risk = "Waiting..."

if "status" not in st.session_state:
    st.session_state.status = "🟢 Healthy"

if "result" not in st.session_state:
    st.session_state.result = None

if "top_features" not in st.session_state:
    st.session_state.top_features = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    
# ==============================
# Hero Section
# ==============================

st.markdown("""
<div class="hero-title">🏭 Grade Change Intelligence</div>
<div class="hero-subtitle">
AI Powered Decision Support for Paper Manufacturing
</div>
""", unsafe_allow_html=True)

# ==============================
# KPI Cards
# ==============================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi">
            <h4>Machine Status</h4>
            <h2>{st.session_state.status}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="kpi">
            <h4>AI Model</h4>
            <h2>XGBoost</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    confidence_text = (
    f"{st.session_state.confidence:.2f}%"
    if st.session_state.analysis_done
    else "--"
)

    st.markdown(
    f"""
    <div class="kpi">
        <h4>Confidence</h4>
        <h2>{confidence_text}</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

with c4:
    st.markdown(
        f"""
        <div class="kpi">
            <h4>Risk Level</h4>
            <h2>{st.session_state.risk}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==============================
# User Inputs
# ==============================

left, right = st.columns(2)

with left:
    stock_flow = st.number_input(
        "Stock Flow",
        value=100.0
    )

    filler_flow = st.number_input(
        "Filler Flow",
        value=50.0
    )

    steam_pressure = st.number_input(
        "Steam Pressure",
        value=70.0
    )

    machine_speed = st.number_input(
        "Machine Speed",
        value=1200.0
    )

with right:
    basis_weight = st.number_input(
        "Basis Weight",
        value=80.0
    )

    moisture = st.number_input(
        "Moisture",
        value=6.5
    )

    ash = st.number_input(
        "Ash",
        value=15.0
    )

    caliper = st.number_input(
        "Caliper",
        value=0.30,
        format="%.2f"
    )

# ==============================
# Analyze Button
# ==============================

if st.button("🔍 Analyze Process", use_container_width=True):

    sample_df = pd.DataFrame(
        [[
            stock_flow,
            filler_flow,
            steam_pressure,
            machine_speed,
            basis_weight,
            moisture,
            ash,
            caliper
        ]],
        columns=[
            "Stock_Flow",
            "Filler_Flow",
            "Steam_Pressure",
            "Machine_Speed",
            "Basis_Weight",
            "Moisture",
            "Ash",
            "Caliper"
        ]
    )

    sample_scaled = scaler.transform(sample_df)

    pred = int(model.predict(sample_scaled)[0])

    confidence = float(
        np.max(model.predict_proba(sample_scaled)) * 100
    )

    result = risk_labels[pred]
    
    st.session_state.result = result
    st.session_state.confidence = confidence

    if result == "Safe":
        st.session_state.status = "🟢 Healthy"
        st.session_state.risk = "🟢 LOW"
        status_class = "status-safe"

    elif result == "Warning":
        st.session_state.status = "🟡 Attention"
        st.session_state.risk = "🟡 MEDIUM"
        status_class = "status-warning"

    else:
        st.session_state.status = "🔴 Critical"
        st.session_state.risk = "🔴 HIGH"
        status_class = "status-danger"

    
    st.session_state.confidence = confidence

    explanation = explain_prediction(sample_df)
    top_features = explanation.head(5)
    
    st.session_state.top_features = top_features
    st.session_state.analysis_done = True

    st.markdown("---")
    
# =========================================
# AI Explanation + Confidence
# =========================================

if not st.session_state.analysis_done:
    st.info("👆 Click Analyze Process first.")
    st.stop()

result = st.session_state.result
confidence = st.session_state.confidence
top_features = st.session_state.top_features

left, right = st.columns([2, 1])

# -----------------------------
# AI Explanation
# -----------------------------
with left:

    st.subheader("🧠 AI Explanation")

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        impact = row["Impact"]

        if impact >= 0:
            emoji = "📈"
            sign = "+"
        else:
            emoji = "📉"
            sign = ""

        st.info(
            f"{emoji} **{feature}**\n\nImpact Score: {sign}{impact:.4f}"
        )

# -----------------------------
# Confidence
# -----------------------------
with right:

    st.write("### 🎯 Model Confidence")

    st.progress(confidence / 100)

    st.write(f"### {confidence:.2f}%")

    st.markdown("---")

    st.subheader("🤖 AI Process Analysis")

    if result == "Safe":
        st.success(f"🟢 Prediction: {result}")
    elif result == "Warning":
        st.warning(f"🟡 Prediction: {result}")
    else:
        st.error(f"🔴 Prediction: {result}")

    st.metric("AI Model", "XGBoost")
    st.metric("Risk Level", st.session_state.risk)

    
feedback_file = BASE_DIR / "data" / "operator_feedback.csv"

if not os.path.exists(feedback_file):
    pd.DataFrame(
        columns=[
            "Timestamp",
            "Recommendation",
            "Decision",
            "Prediction"
        ]
    ).to_csv(feedback_file, index=False)

# ------------------------------
# Recommendations
# ------------------------------

left, right = st.columns([1.2, 1])

# ==================================
# AI Recommendations
# ==================================
with left:

    st.subheader("💡 AI Recommendations")

    recommendations = generate_recommendations(
        result,
        stock_flow,
        filler_flow,
        steam_pressure,
        machine_speed,
        basis_weight,
        moisture,
        ash,
        caliper,
    )
    
    

     
    for i, rec in enumerate(recommendations):

        st.info(f"""
✅ {rec}

📌 Source:
Historical Safe Production + XGBoost + SHAP
""")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Accept", key=f"accept_{i}"):

                feedback = pd.read_csv(feedback_file)

                feedback.loc[len(feedback)] = [
                    pd.Timestamp.now(),
                    rec,
                    "Accepted",
                    result
                ]

                feedback.to_csv(feedback_file, index=False)

                st.success("Recommendation Accepted!")

        with col2:
            if st.button("👎 Reject", key=f"reject_{i}"):

                feedback = pd.read_csv(feedback_file)

                feedback.loc[len(feedback)] = [
                    pd.Timestamp.now(),
                    rec,
                    "Rejected",
                    result
                ]

                feedback.to_csv(feedback_file, index=False)

                st.warning("Recommendation Rejected!")
with right:

    st.subheader("📊 Process Correlation Analysis")

    corr_df = pd.read_csv(
        BASE_DIR / "data" / "synthetic_paper_data.csv"
    )

    corr = corr_df.drop(columns=["Risk"]).corr()

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(corr, cmap="coolwarm")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right",
        fontsize=8
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns,
        fontsize=8
    )

    plt.colorbar(im, fraction=0.046)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔗 Top Correlations Found by AI")

corr_matrix = corr_df.drop(columns=["Risk"]).corr()

correlations = []

columns = corr_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        correlations.append({
            "Parameter 1": columns[i],
            "Parameter 2": columns[j],
            "Correlation": corr_matrix.iloc[i, j]
        })

corr_table = pd.DataFrame(correlations)

corr_table["Absolute"] = corr_table["Correlation"].abs()

corr_table = corr_table.sort_values(
    "Absolute",
    ascending=False
)

st.dataframe(
    corr_table[["Parameter 1", "Parameter 2", "Correlation"]].head(5),
    use_container_width=True
)

st.markdown("---")

st.subheader("🎯 AI Suggested Operating Setpoints")

history = pd.read_csv(BASE_DIR / "data" / "synthetic_paper_data.csv")

safe_data = history[history["Risk"] == "Safe"]

setpoints = []



for feature in top_features["Feature"]:

    if feature in safe_data.columns:

        avg = safe_data[feature].mean()

        minimum = safe_data[feature].min()

        maximum = safe_data[feature].max()

        setpoints.append({
            "Parameter": feature,
            "Recommended Setpoint": round(avg, 2),
            "Operating Range": f"{minimum:.2f} - {maximum:.2f}",
            "Source": "Historical Safe Production"
        })

setpoint_df = pd.DataFrame(setpoints)

st.dataframe(setpoint_df, use_container_width=True)

st.markdown("---")
st.subheader("📋 Operator Feedback History")

feedback = pd.read_csv(feedback_file)

st.dataframe(
    feedback,
    use_container_width=True
)

st.markdown("---")

st.subheader("📥 Export Analysis Report")

if st.button("📄 Generate PDF"):

    generate_pdf(
        "Grade_Report.pdf",
        result,
        confidence,
        st.session_state.risk,
        top_features,
        recommendations,
        setpoints,
    )

    st.success("✅ PDF Generated Successfully!")

    with open("Grade_Report.pdf", "rb") as pdf_file:

        st.download_button(
            "⬇ Download Report",
            pdf_file,
            file_name="Honeywell_Grade_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )