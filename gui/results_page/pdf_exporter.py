"""
PDF export module for the Climbing Testing Application.

This module provides functionality for generating PDF reports from test data
and analysis results. It creates formatted, professional reports suitable for
sharing or archiving research findings.

Key functionalities:
- Generate comprehensive test reports as PDF documents
- Include data visualizations and charts in reports
- Format tabular data for readability
- Create consistent report layouts with proper branding
- Support for multiple report templates for different purposes

The PDF exporter enables researchers to produce standardized documentation
of test results for academic publication or participant feedback.
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Define a dictionary with explanation text for each parameter:
parameters_explanation_dict = {
    "max_strength": "<b>Maximal Force - MVC (kg):</b> This parameter represents the highest force output achieved during a maximal effort isometric contraction. It is essential for assessing peak strength and overall climbing power.",
    "avg_end_force": "<b>Average End-Force (kg):</b> The mean force measured toward the end of a sustained contraction, reflecting the climber's ability to maintain force as fatigue sets in. This helps evaluate endurance and grip stability.",
    "time_between_max_end_ms": "<b>Average Time btw Max- and End-Force (ms):</b> The average duration from reaching maximum force to the force at the end of a repetition, indicating how quickly force decays under fatigue.",
    "force_drop_pct": "<b>Average Force Drop (%):</b> The percentage decline from peak force to end force, serving as an indicator of fatigue resistance.",
    "avg_rep_force": "<b>Average Rep. Force (kg):</b> The mean force exerted across repetitions, showing overall strength consistency.",
    "critical_force": "<b>Critical Force - CF (kg):</b> Determined by averaging the final repetition peaks, it indicates the sustainable force level a climber can maintain.",
    "reps_to_cf": "<b>Repetitions to CF:</b> The number of repetitions completed before force falls below the critical threshold, directly measuring endurance.",
    "cf_mvc_pct": "<b>CF/MVC (%):</b> The ratio of critical force to maximal force expressed as a percentage. This normalizes endurance relative to peak strength.",
    "work": "<b>Average Work (kg/s):</b> The average work performed per repetition, calculated from the force–time curve.",
    "sum_work": "<b>Sum Work (kg/s):</b> The total work done over the entire test.",
    "avg_work_above_cf": "<b>Average Work above CF (kg/s):</b> The average work performed while the force remains above the critical force, reflecting sustained high-force effort.",
    "sum_work_above_cf": "<b>Sum Work above CF (kg/s):</b> The cumulative work performed above the critical force.",
    "avg_pulling_time_ms": "<b>Average Pulling Time (ms):</b> The average duration of a repetition, reflecting movement speed and control.",
    "rfd_overall": "<b>Rate of Force Development - RFD (ms):</b> The overall time required to develop force, indicative of explosive strength.",
    "rfd_first3": "<b>RFD first three repetitions (ms):</b> The average RFD over the first three repetitions, capturing initial explosive capability.",
    # "rfd_first6": "<b>RFD first six repetitions (ms):</b> The average RFD over the first six repetitions, providing a robust measure of early performance.",
    "rfd_last3": "<b>RFD last three repetitions (ms):</b> The average RFD during the final three repetitions, showing fatigue effects.",
    "rfd_norm_overall": "<b>RFD normalized to force (ms/kg):</b> The overall RFD normalized by the maximal force, allowing relative comparison across athletes.",
    "rfd_norm_first3": "<b>RFD norm. first three rep. (ms/kg):</b> Normalized RFD for the first three repetitions.",
    # "rfd_norm_first6": "<b>RFD norm. first six rep. (ms/kg):</b> Normalized RFD for the first six repetitions.",
    "rfd_norm_last3": "<b>RFD norm. last three rep. (ms/kg):</b> Normalized RFD for the last three repetitions."
}


def filter_parameters_explanation(test_results, explanation_dict):
    """
    Given test_results (a dict containing keys of the test parameters) and an explanation dictionary,
    returns a concatenated string of explanations only for the parameters that appear in test_results.
    """
    explanations = []
    for key in test_results.keys():
        if key in explanation_dict:
            explanations.append(explanation_dict[key])
    return "<br/><br/>".join(explanations)


# Modify the generate_pdf_report function signature to add a new parameter
def generate_pdf_report(
    pdf_path,
    title_text,
    basic_info,
    participant_info,
    test_results,
    nirs_results=None,
    graph_image_path=None,
    rep_results=None,
    rep_graph_image_path=None,
    norm_force_graph_image_path=None,
    ircra_prediction=None,  # New parameter for IRCRA prediction
    parameters_explanation=None
):
    """
    Generate a complete PDF report with:
      - Basic info, Participant info, Test Metrics, Force-Time Graph,
      - Repetition Metrics table, Repetition Graph, IRCRA Prediction, and Parameters Explanation.
      
    Args:
        pdf_path (str): Path to save the PDF file.
        title_text (str): Title of the report.
        basic_info (list): List of (label, value) pairs for basic test info.
        participant_info (list): List of (label, value) pairs for participant info.
        test_results (list): List of (label, value) pairs for test metrics.
        nirs_results (list, optional): List of (label, value) pairs for NIRS metrics.
        graph_image_path (str, optional): Path to the force-time graph image.
        rep_results (list, optional): List of rows for repetition metrics table.
        rep_graph_image_path (str, optional): Path to the repetition graph image.
        norm_force_graph_image_path (str, optional): Path to normalized max force graph image.
        ircra_prediction (dict, optional): Dictionary with IRCRA prediction information.
        parameters_explanation (str, optional): Text explaining test parameters.
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']

    # 1. Title
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 24))

    # Create the basic info table as before
    basic_info_data = [[label, value] for (label, value) in basic_info]
    table_basic = Table(basic_info_data, hAlign='LEFT')
    table_basic.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    # Create the participant info table as before
    participant_info_data = [[label, value] for (label, value) in participant_info]
    table_participant = Table(participant_info_data, hAlign='LEFT')
    table_participant.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    available_width = A4[0] - 33 - 33  # A4 width minus margins
    print('Available width pdf:', available_width)
    col_width = available_width / 2

    left_cell = [Paragraph("Basic Test Information", heading_style), table_basic]
    right_cell = [Paragraph("Participant Information", heading_style), table_participant]

    # Create a combined table with one row and two columns
    combined_info = Table([[left_cell, right_cell]], colWidths=[col_width - 55, col_width + 55])
    combined_info.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(combined_info)
    story.append(Spacer(1, 12))

    # 4. Force-Time Graph Image
    if graph_image_path:
        im = Image(graph_image_path)
        available_width = A4[0] - 5 - 5
        if im.imageWidth > available_width:
            scale = available_width / im.imageWidth
            im.drawWidth = im.imageWidth * scale
            im.drawHeight = im.imageHeight * scale
        story.append(KeepTogether([Paragraph("Force-Time Graph", heading_style), im]))
    story.append(Spacer(1, 12))

    # 5. Test Metrics Table
    if nirs_results:
        nirs_results_data = [[label, value] for (label, value) in nirs_results]
        table_nirs = Table(nirs_results_data, hAlign='LEFT')
        table_nirs.setStyle(TableStyle([
             ('BACKGROUND', (0, 0), (-1, -1), colors.white),
             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
             ('BOX', (0, 0), (-1, -1), 1, colors.black),
             ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(KeepTogether([Paragraph("NIRS Results", heading_style), table_nirs]))

    test_metrics_data = [[label, value] for (label, value) in test_results]
    table_metrics = Table(test_metrics_data, hAlign='LEFT')
    table_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(KeepTogether([Paragraph("Test Results", heading_style), table_metrics]))
    story.append(Spacer(1, 12))

    # 6. Normalized Max Force Analysis Graph Image
    if norm_force_graph_image_path:
        im_norm = Image(norm_force_graph_image_path)
        available_width = A4[0] - 5 - 5
        if im_norm.imageWidth > available_width:
            scale = available_width / im_norm.imageWidth
            im_norm.drawWidth = im_norm.imageWidth * scale
            im_norm.drawHeight = im_norm.imageHeight * scale
        story.append(KeepTogether([Paragraph("Normalized Max Force Analysis", heading_style), im_norm]))
    story.append(Spacer(1, 12))

    # 7. IRCRA Prediction section
    if ircra_prediction and ircra_prediction.get('test_type') == 'ao':
        prediction_text = f"With your predispositions, you could be able to climb IRCRA grade: {ircra_prediction.get('predicted_ircra', '-')}"

        prediction_parts = [Paragraph("Performance Prediction", heading_style),
                            Paragraph(prediction_text, normal_style)]

        # Add analysis if available
        if 'analysis_text' in ircra_prediction:
            prediction_parts.append(Spacer(1, 6))
            prediction_parts.append(Paragraph(ircra_prediction['analysis_text'], normal_style))

        story.append(KeepTogether(prediction_parts))
        story.append(Spacer(1, 12))
    
    # 8. Repetition Graph Image
    if rep_graph_image_path:
        im_rep = Image(rep_graph_image_path)
        available_width = A4[0] - 5 - 5
        if im_rep.imageWidth > available_width:
            scale = available_width / im_rep.imageWidth
            im_rep.drawWidth = im_rep.imageWidth * scale
            im_rep.drawHeight = im_rep.imageHeight * scale
        story.append(KeepTogether([Paragraph("Repetition Graph", heading_style), im_rep]))
    story.append(Spacer(1, 12))

    # 9. Repetition Metrics Table
    if rep_results:
        table_rep = Table(rep_results, hAlign='CENTER')
        table_rep.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(KeepTogether([Paragraph("Repetition Results", heading_style), table_rep]))
        story.append(Spacer(1, 12))

    # 10. Parameters Explanation
    if parameters_explanation:
        story.append(KeepTogether([Paragraph("Measured Parameters Explanation", heading_style),
                                   Paragraph(parameters_explanation, normal_style)]))
        story.append(Spacer(1, 12))

    doc.build(story)


if __name__ == '__main__':
    pdf_file = "exported_report.pdf"
    title = "All-Out Report for Chris Bron"
    basic = [("Test Name", "AO"), ("Data Type", "Force"), ("Arm Tested", "Right"),
             ("Date", "2024-12-12"), ("Time", "20:31:27")]
    participant = [("Name", "Chris"), ("Surname", "Bron"),
                   ("Email", "chrisbronstein@outlook.com"), ("Gender", "m"),
                   ("Dominant Arm", "r"), ("Weight", "77.0"),
                   ("Height", "180.0"), ("Age", "33")]
    metrics = [("Maximal Force - MVC (Kg)", "38.42"),
               ("Average End-Force (Kg)", "19.94"),
               ("Critical Force - CF (Kg)", "15.35")]

    # Path to a saved force-time graph image (or a file-like object)
    graph_img = "logo_uct01.png"

    # Sample repetition metrics table (first row is header)
    rep_metrics_table = [
        ["Rep (#)", "MVC (kg)", "Endforce (kg)", "Force Drop (%)", "Avg. Force (kg)",
         "W (kg.s-1)", "W' (kg.s-1)", "Pull Time (ms)", "RFD (ms)", "RFDnorm (ms.kg-1)"],
        ["1", "38.42", "37.01", "4.0", "32.89", "268.7", "143.3", "8170", "570", "40.4"],
        # Additional rows can be added here...
    ]

    parameters_text = (
        "Maximal Force - MVC (Kg): The highest force output achieved during a maximal effort isometric contraction.\n"
        "Average End-Force (Kg): The mean force measured toward the end of a sustained contraction.\n"
        "Critical Force - CF (Kg): The asymptotic force that can be maintained without fatigue.\n"
        # Add more parameter explanations as needed.
    )

    generate_pdf_report(
        pdf_path=pdf_file,
        title_text=title,
        basic_info=basic,
        participant_info=participant,
        test_results=metrics,
        nirs_results=None,
        graph_image_path=graph_img,
        norm_force_graph_image_path=None,
        rep_results=rep_metrics_table,
        rep_graph_image_path=graph_img,
        parameters_explanation=parameters_text
    )
    print("PDF report generated successfully!")