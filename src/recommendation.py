import streamlit as st


def generate_recommendations(
    prediction,
    stock_flow,
    filler_flow,
    steam_pressure,
    machine_speed,
    basis_weight,
    moisture,
    ash,
    caliper,
):
    recommendations = []
    st.write("Recommendations:", recommendations)

    if prediction == "Safe":
        recommendations.append("Machine is operating within normal limits.")
        recommendations.append("Continue monitoring the process.")

    else:
        if basis_weight < 75:
            recommendations.append("Increase basis weight to at least 75 GSM.")

        elif basis_weight > 85:
            recommendations.append("Reduce basis weight below 85 GSM.")

        if steam_pressure < 65:
            recommendations.append("Increase steam pressure.")

        elif steam_pressure > 75:
            recommendations.append("Reduce steam pressure.")

        if moisture < 5.8:
            recommendations.append("Increase moisture level.")

        elif moisture > 7.2:
            recommendations.append("Reduce moisture level.")

        if machine_speed > 1280:
            recommendations.append("Reduce machine speed by 5–10%.")

        if stock_flow < 90:
            recommendations.append("Increase stock flow.")

        elif stock_flow > 110:
            recommendations.append("Reduce stock flow.")

        if filler_flow < 45:
            recommendations.append("Increase filler flow.")

        elif filler_flow > 55:
            recommendations.append("Reduce filler flow.")

        if ash < 13:
            recommendations.append("Increase ash content.")

        elif ash > 17:
            recommendations.append("Reduce ash content.")

        if caliper < 0.28:
            recommendations.append("Increase caliper.")

        elif caliper > 0.32:
            recommendations.append("Reduce caliper.")

    if len(recommendations) == 0:
        recommendations.append("No corrective action required.")

    return recommendations


if __name__ == "__main__":

    result = generate_recommendations(
        "Off-Spec",
        85,
        60,
        62,
        1300,
        72,
        5.2,
        18,
        0.34
    )

    print("\nRecommendations\n")

    for i, recommendation in enumerate(result, start=1):
        print(f"{i}. {recommendation}")