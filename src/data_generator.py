import pandas as pd
import numpy as np

np.random.seed(42)

num_samples = 50000

stock_flow = np.random.normal(100, 8, num_samples)
filler_flow = np.random.normal(25, 3, num_samples)
steam_pressure = np.random.normal(70, 5, num_samples)
machine_speed = np.random.normal(1200, 60, num_samples)
basis_weight = np.random.normal(80, 4, num_samples)
moisture = np.random.normal(6.5, 0.5, num_samples)
ash = np.random.normal(15, 2, num_samples)
caliper = np.random.normal(110, 6, num_samples)

risk = []

for i in range(num_samples):

    score = 0

    if basis_weight[i] < 75 or basis_weight[i] > 85:
        score += 2

    if steam_pressure[i] < 65 or steam_pressure[i] > 75:
        score += 2

    if moisture[i] < 5.8 or moisture[i] > 7.2:
        score += 2

    if machine_speed[i] > 1280:
        score += 1

    if stock_flow[i] < 90 or stock_flow[i] > 110:
        score += 1

    if score <= 2:
        risk.append("Safe")
    elif score <= 4:
        risk.append("Warning")
    else:
        risk.append("Off-Spec")

df = pd.DataFrame({
    "Stock_Flow": stock_flow,
    "Filler_Flow": filler_flow,
    "Steam_Pressure": steam_pressure,
    "Machine_Speed": machine_speed,
    "Basis_Weight": basis_weight,
    "Moisture": moisture,
    "Ash": ash,
    "Caliper": caliper,
    "Risk": risk
})

df.to_csv("data/synthetic_paper_data.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())