import tempfile

from fastapi.responses import FileResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
from report import REPORT_SECTIONS


def build_pdf_response(result: dict, report: dict) -> FileResponse:
    subject = result["subject_id"]
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_pdf.close()

    doc = SimpleDocTemplate(
        tmp_pdf.name,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=13,
        textColor=colors.HexColor("#1a4b8c"),
        spaceAfter=4,
        spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    )

    story.append(Paragraph("MentalMetrics EEG Analysis Report", title_style))
    story.append(Paragraph(f"Subject ID: {subject}", meta_style))
    prediction = result["prediction"]
    story.append(
        Paragraph(
            f"Model Prediction: <b>{prediction['label']}</b> "
            f"({prediction['confidence']}% confidence)",
            meta_style,
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.3 * cm))

    for key, heading in REPORT_SECTIONS:
        content = report.get(key, "")
        story.append(Paragraph(heading, h1_style))
        for paragraph in content.split("\n\n"):
            if paragraph.strip():
                story.append(
                    Paragraph(paragraph.strip().replace("\n", " "), body_style)
                )
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)

    return FileResponse(
        tmp_pdf.name,
        media_type="application/pdf",
        filename=f"MentalMetrics_Report_{subject}.pdf",
    )
