from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap

# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "grade_change_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# -----------------------------
# Feature Names
# -----------------------------
feature_names = [
    "Stock_Flow",
    "Filler_Flow",
    "Steam_Pressure",
    "Machine_Speed",
    "Basis_Weight",
    "Moisture",
    "Ash",
    "Caliper",
]

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)


def explain_prediction(sample_df):

    # Scale input
    sample_scaled = scaler.transform(sample_df)

    # SHAP values
    shap_values = explainer.shap_values(sample_scaled)

    # Predicted class
    prediction = int(model.predict(sample_scaled)[0])

    # Handle different SHAP versions
    if isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:
            values = shap_values[0, :, prediction]
        else:
            values = shap_values[0]

    else:
        values = shap_values[prediction][0]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    })

    explanation["Absolute"] = explanation["Impact"].abs()

    explanation = explanation.sort_values(
        by="Absolute",
        ascending=False
    )

    return explanation.drop(columns="Absolute")