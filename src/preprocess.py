from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "synthetic_paper_data.csv")

risk_mapping = {
    "Safe": 0,
    "Warning": 1,
    "Off-Spec": 2
}

df["Risk"] = df["Risk"].map(risk_mapping)

X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, BASE_DIR / "models" / "scaler.pkl")

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])
print("Features         :", X.shape[1])
print("Scaler Saved Successfully!")