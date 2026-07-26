from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

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

scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Model Accuracy")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, BASE_DIR / "models" / "grade_change_model.pkl")

print("\nModel Saved Successfully!")