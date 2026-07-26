from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "grade_change_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

risk_labels = {
    0: "Safe",
    1: "Warning",
    2: "Off-Spec"
}


def predict_risk(
    stock_flow,
    filler_flow,
    steam_pressure,
    machine_speed,
    basis_weight,
    moisture,
    ash,
    caliper,
):
    data = np.array([[
        stock_flow,
        filler_flow,
        steam_pressure,
        machine_speed,
        basis_weight,
        moisture,
        ash,
        caliper
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    return risk_labels[int(prediction)]


if __name__ == "__main__":
    result = predict_risk(
        100,
        50,
        70,
        1200,
        80,
        6.5,
        15,
        0.30
    )

    print("=" * 40)
    print("Prediction :", result)
    print("=" * 40)