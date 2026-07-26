from recommendation import generate_recommendations
from explain_ai import explain_prediction
from pdf_report import generate_pdf

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.units import inch


def generate_pdf(
    filename,
    result,
    confidence,
    risk,
    top_features,
    recommendations,
    setpoints
):

    styles = getSampleStyleSheet()

    pdf = SimpleDocTemplate(filename)

    elements = []

    elements.append(
        Paragraph(
            "<b><font size=20>Honeywell</font></b>",
            styles["Title"]
        )
    )

    elements.append(
        Paragraph(
            "<b>Grade Change Intelligence Report</b>",
            styles["Heading1"]
        )
    )

    elements.append(Spacer(1, 0.25 * inch))

    elements.append(
        Paragraph(f"<b>Prediction:</b> {result}", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Risk Level:</b> {risk}", styles["Normal"])
    )

    elements.append(Spacer(1, 0.25 * inch))

    # -----------------------------------
    # AI Explanation
    # -----------------------------------

    elements.append(
        Paragraph("<b>Top Influencing Features</b>", styles["Heading2"])
    )

    feature_data = [["Feature", "Impact"]]

    for _, row in top_features.iterrows():

        feature_data.append([
            row["Feature"],
            f"{row['Impact']:.4f}"
        ])

    table = Table(feature_data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),

        ("GRID",(0,0),(-1,-1),1,colors.black),

        ("BACKGROUND",(0,1),(-1,-1),colors.beige),

        ("ALIGN",(0,0),(-1,-1),"CENTER")
    ]))

    elements.append(table)

    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------------
    # Recommendations
    # -----------------------------------

    elements.append(
        Paragraph("<b>AI Recommendations</b>", styles["Heading2"])
    )

    for rec in recommendations:

        elements.append(
            Paragraph(f"• {rec}", styles["Normal"])
        )

    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------------
    # Setpoints
    # -----------------------------------

    elements.append(
        Paragraph("<b>Suggested Operating Setpoints</b>", styles["Heading2"])
    )

    setpoint_data = [["Parameter","Setpoint"]]

    for row in setpoints:

        setpoint_data.append([
            row["Parameter"],
            str(row["Recommended Setpoint"])
        ])

    table2 = Table(setpoint_data)

    table2.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.green),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("ALIGN",(0,0),(-1,-1),"CENTER")
    ]))

    elements.append(table2)

    pdf.build(elements)