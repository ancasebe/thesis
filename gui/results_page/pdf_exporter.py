'''
# pdf_exporter.py

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(pdf_path, title_text, basic_info, participant_info, test_metrics,
                        graph_image_path, rep_metrics, parameters_explanation):
    """
    Generates a PDF report with the following sections (in order):
      - Title
      - Basic Test Information (table)
      - Participant Information (table)
      - Force-Time Graph (image)
      - Test Metrics (table)
      - Repetition Metrics (table)
      - Parameters Explanation (text)

    All tables have a white background with black text.
    The first column of every table is bold.
    The repetition metrics header row is split into two lines to help fit the page.
    The Force-Time Graph image is scaled to fit within the available width on an A4 page
    while preserving its aspect ratio.

    Args:
        pdf_path (str): The path where the PDF file will be saved.
        title_text (str): The title for the report.
        basic_info (list of tuple): List of (label, value) pairs for basic test info.
        participant_info (list of tuple): List of (label, value) pairs for participant info.
        test_metrics (list of tuple): List of (label, value) pairs for overall test metrics.
        graph_image_path (str or file-like): The graph image file path or buffer.
        rep_metrics (list of list): A table (list of rows) where the first row is the header.
        parameters_explanation (str): A text block explaining the measured parameters.
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()

    # Use standard styles for title, headings, and normal text
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']

    # 1. Title
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 24))

    # 2. Basic Test Information Table
    story.append(Paragraph("Basic Test Information", heading_style))
    basic_info_data = [[label, value] for (label, value) in basic_info]
    table_basic = Table(basic_info_data, hAlign='LEFT')
    table_basic.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # first column bold
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table_basic)
    story.append(Spacer(1, 12))

    # 3. Participant Information Table
    story.append(Paragraph("Participant Information", heading_style))
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
    story.append(table_participant)
    story.append(Spacer(1, 12))

    # 4. Force-Time Graph Image (placed before Test Metrics)
    story.append(Paragraph("Force-Time Graph", heading_style))
    if graph_image_path:
        im = Image(graph_image_path)
        # Scale the image to fit within the available width while preserving aspect ratio.
        available_width = A4[0] - 5 - 5  # A4 width minus left and right margins (72 each)
        if im.imageWidth > available_width:
            scale = available_width / im.imageWidth
            im.drawWidth = im.imageWidth * scale
            im.drawHeight = im.imageHeight * scale
        story.append(im)
    story.append(Spacer(1, 12))

    # 5. Test Metrics Table
    story.append(Paragraph("Test Metrics", heading_style))
    test_metrics_data = [[label, value] for (label, value) in test_metrics]
    table_metrics = Table(test_metrics_data, hAlign='LEFT')
    table_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table_metrics)
    story.append(Spacer(1, 12))

    # 6. Repetition Metrics Table (if available)
    if rep_metrics:
        story.append(Paragraph("Repetition Metrics", heading_style))
        # Transform the header row to include line breaks for narrow columns.
        header = rep_metrics[0]
        # Manually split header names to mimic your provided pdf formatting.
        transformed_header = [
            "Rep (#)",
            "MVC\n(kg)",
            "Endforce\n(kg)",
            "Force\nDrop (%)",
            "Avg. Force\n(kg)",
            "W\n(kg.s-1)",
            "W'\n(kg.s-1)",
            "Pull\nTime (ms)",
            "RFD\n(ms)",
            "RFDnorm\n(ms.kg-1)"
        ]
        rep_metrics[0] = transformed_header

        table_rep = Table(rep_metrics, hAlign='CENTER')
        table_rep.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # first column bold for data rows
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(table_rep)
        story.append(Spacer(1, 12))

    # 7. Parameters Explanation
    story.append(Paragraph("Measured Parameters Explanation", heading_style))
    story.append(Paragraph(parameters_explanation, normal_style))
    story.append(Spacer(1, 12))

    # Build the PDF document
    doc.build(story)


# Example usage:
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

    generate_pdf_report(pdf_file, title, basic, participant, metrics, graph_img, rep_metrics_table, parameters_text)
    print("PDF report generated successfully!")
'''
# pdf_exporter.py

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_pdf_report(pdf_path, title_text, basic_info, participant_info, test_metrics,
                        graph_image_path, rep_metrics, rep_graph_image_path, parameters_explanation):
    """
    Generates a PDF report with the following sections (in order):
      - Title
      - Basic Test Information (table)
      - Participant Information (table)
      - Force-Time Graph (image)
      - Test Metrics (table)
      - Repetition Metrics (table)
      - Repetition Graph (image)
      - Parameters Explanation (text)

    All tables have a white background with black text.
    Only the first column is rendered in bold.
    The repetition metrics header row is split into two lines.
    Both graph images are scaled (if needed) to fit within the available width on an A4 page
    while preserving aspect ratio.

    Args:
        pdf_path (str): Path to save the PDF.
        title_text (str): Report title.
        basic_info (list of tuple): Basic test info.
        participant_info (list of tuple): Participant info.
        test_metrics (list of tuple): Test metrics.
        graph_image_path (str or file-like): Force-Time Graph image.
        rep_metrics (list of list): Repetition metrics table (first row header).
        rep_graph_image_path (str or file-like): Repetition Graph image.
        parameters_explanation (str): Explanation text.
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

    # # 2. Basic Test Information Table
    # story.append(Paragraph("Basic Test Information", heading_style))
    # basic_info_data = [[label, value] for (label, value) in basic_info]
    # table_basic = Table(basic_info_data, hAlign='LEFT')
    # table_basic.setStyle(TableStyle([
    #     ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    #     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    #     ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    #     ('BOX', (0, 0), (-1, -1), 1, colors.black),
    #     ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    # ]))
    # # story.append(table_basic)
    # # story.append(Spacer(1, 12))
    #
    # # 3. Participant Information Table
    # story.append(Paragraph("Participant Information", heading_style))
    # participant_info_data = [[label, value] for (label, value) in participant_info]
    # table_participant = Table(participant_info_data, hAlign='LEFT')
    # table_participant.setStyle(TableStyle([
    #     ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    #     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    #     ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    #     ('BOX', (0, 0), (-1, -1), 1, colors.black),
    #     ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    # ]))
    # # story.append(table_participant)
    # # story.append(Spacer(1, 12))
    #
    # available_width = A4[0] - 72 - 72  # A4 width minus left/right margins
    # col_width = available_width / 2
    #
    # # Create a new table with one row and two columns:
    # combined_info = Table([[table_basic, table_participant]], colWidths=[col_width, col_width])
    # combined_info.setStyle(TableStyle([
    #     ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    # ]))
    # story.append(combined_info)
    # story.append(Spacer(1, 12))

    from reportlab.platypus import KeepTogether

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

    available_width = A4[0] - 72 - 72  # A4 width minus margins
    col_width = available_width / 2

    # Combine each title with its corresponding table in a cell
    left_cell = [Paragraph("Basic Test Information", heading_style), table_basic]
    right_cell = [Paragraph("Participant Information", heading_style), table_participant]

    # Create a combined table with one row and two columns
    combined_info = Table([[left_cell, right_cell]], colWidths=[col_width, col_width])
    combined_info.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(combined_info)
    story.append(Spacer(1, 12))

    # 4. Force-Time Graph Image
    story.append(Paragraph("Force-Time Graph", heading_style))
    if graph_image_path:
        im = Image(graph_image_path)
        available_width = A4[0] - 5 - 5
        if im.imageWidth > available_width:
            scale = available_width / im.imageWidth
            im.drawWidth = im.imageWidth * scale
            im.drawHeight = im.imageHeight * scale
        story.append(im)
    story.append(Spacer(1, 12))

    # 5. Test Metrics Table
    story.append(Paragraph("Test Metrics", heading_style))
    test_metrics_data = [[label, value] for (label, value) in test_metrics]
    table_metrics = Table(test_metrics_data, hAlign='LEFT')
    table_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table_metrics)
    story.append(Spacer(1, 12))

    # 6. Repetition Graph Image
    story.append(Paragraph("Repetition Graph", heading_style))
    if rep_graph_image_path:
        im_rep = Image(rep_graph_image_path)
        available_width = A4[0] - 5 - 5
        if im_rep.imageWidth > available_width:
            scale = available_width / im_rep.imageWidth
            im_rep.drawWidth = im_rep.imageWidth * scale
            im_rep.drawHeight = im_rep.imageHeight * scale
        story.append(im_rep)
    story.append(Spacer(1, 12))

    # 7. Repetition Metrics Table
    if rep_metrics:
        story.append(Paragraph("Repetition Metrics", heading_style))
        table_rep = Table(rep_metrics, hAlign='CENTER')
        table_rep.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            # ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        # story.append(table_rep)
        story.append(KeepTogether([table_rep]))
        story.append(Spacer(1, 12))

    # 8. Parameters Explanation
    story.append(Paragraph("Measured Parameters Explanation", heading_style))
    story.append(Paragraph(parameters_explanation, normal_style))
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

    generate_pdf_report(pdf_file, title, basic, participant, metrics, graph_img, rep_metrics_table, graph_img, parameters_text)
    print("PDF report generated successfully!")