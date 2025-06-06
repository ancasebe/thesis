"""
New climber registration module for the Climbing Testing Application.

This module defines the NewClimber class which provides the user interface
for adding new climbing participants to the database. It collects personal
information, climbing experience, and other relevant data.

Key functionalities:
- Collect and validate new climber information
- Store climber data in the database
- Link climbers to their administering researcher
- Provide feedback on registration success or errors

The new climber module facilitates the participant onboarding process
for researchers conducting climbing studies.
"""
# pylint: disable=E1101
import re
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QSpinBox, QLabel, QSlider,
                               QComboBox, QPushButton, QGridLayout, QMessageBox, QHBoxLayout)
from PySide6.QtCore import Qt


class NewClimber(QWidget):
    """
    A class representing the registration page of the application.

    Methods:
        setup_ui: Sets up the user interface for the registration page.
        handle_registration: Handles the registration process, validating user input, saving data.
        validate_inputs: Validates the user inputs to ensure completeness and correctness.
    """

    def __init__(self, admin_id, switch_to_test_page, climber_db_manager):
        """Initializes the registration page with references to the switch function and climber_db_manager."""
        super().__init__()
        self.first_column_widgets = {}
        self.second_column_widgets = {}
        self.bouldering_lead_slider = None
        self.indoor_outdoor_slider = None
        self.admin_id = admin_id
        self.switch_to_test_page = switch_to_test_page
        self.climber_db_manager = climber_db_manager
        self.setup_ui()
        # Flag to indicate whether the user is saving
        self.is_saving = False

    def setup_ui(self):
        """Sets up the layout and widgets for the registration page in a dynamic, maintainable way."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Create a grid layout for organizing form in columns
        grid_layout = QGridLayout()

        # Dictionary of inputs and their labels for the first and second columns
        self.first_column_widgets = {
            "Name:": QLineEdit(),
            "Surname:": QLineEdit(),
            "Email:": QLineEdit(),
            "Gender:": QComboBox(),
            "Dominant Arm:": QComboBox(),
            "Weight (kg):": QSpinBox(),
            "Height (cm):": QSpinBox(),
            "Age (years):": QSpinBox(),
        }

        self.second_column_widgets = {
            "IRCRA Scale Level:": QComboBox(),
            "Years of Climbing:": QSpinBox(),
            "Climbing Frequency/week:": QSpinBox(),
            "Climbing Hours/week:": QSpinBox(),
            "Other sports:": QLineEdit(),
            "Sport Frequency/week:": QSpinBox(),
            "Sport Activity (hours/week):": QSpinBox()
        }

        # Configuring certain widgets
        self.first_column_widgets["Gender:"].addItems(["-", "Male", "Female", "Other"])
        self.first_column_widgets["Dominant Arm:"].addItems(["-", "Left", "Right"])
        self.first_column_widgets["Weight (kg):"].setRange(0, 200)
        self.first_column_widgets["Height (cm):"].setRange(0, 240)
        self.first_column_widgets["Age (years):"].setRange(0, 100)

        self.second_column_widgets["IRCRA Scale Level:"].addItems(
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']
        )
        self.second_column_widgets["Years of Climbing:"].setRange(0, 50)
        self.second_column_widgets["Climbing Frequency/week:"].setRange(0, 7)
        self.second_column_widgets["Climbing Hours/week:"].setRange(0, 50)
        self.second_column_widgets["Sport Frequency/week:"].setRange(0, 7)
        self.second_column_widgets["Sport Activity (hours/week):"].setRange(0, 50)

        # Adding widgets to the grid layout in the first column (left)
        for i, (label, widget) in enumerate(self.first_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 0)
            grid_layout.addWidget(widget, i, 1)

        # Adding standard widgets to the grid layout in the second column (right)
        for i, (label, widget) in enumerate(self.second_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 2)
            grid_layout.addWidget(widget, i, 3)

        # Add the custom slider layouts for Bouldering/Lead and Indoor/Outdoor climbing
        bouldering_lead_layout, self.bouldering_lead_slider = self.create_slider_layout("Bouldering", "Lead Climbing")
        indoor_outdoor_layout, self.indoor_outdoor_slider = self.create_slider_layout("Indoor", "Outdoor Climbing")

        # Add these custom layouts directly into the grid layout without dictionary keys
        grid_layout.addLayout(bouldering_lead_layout, len(self.second_column_widgets), 2, 1, 2)
        grid_layout.addLayout(indoor_outdoor_layout, len(self.second_column_widgets) + 1, 2, 1, 2)

        # Submit and switch button
        submit_button = QPushButton("Save questioner")
        submit_button.clicked.connect(self.handle_registration)

        # switch_button = QPushButton("Back to Test Selection")
        # switch_button.clicked.connect(self.switch_to_test_page)

        # Add the grid layout and buttons to the main layout
        layout.addLayout(grid_layout)
        layout.addWidget(submit_button)
        # layout.addWidget(switch_button)

        self.setLayout(layout)

    def create_slider_layout(self, left_label_text, right_label_text):
        """Creates a layout for a slider with labels on either side to represent switched percentages."""
        layout = QHBoxLayout()
        left_label = QLabel(f"{left_label_text}: 50%")
        right_label = QLabel(f"{right_label_text}: 50%")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setSingleStep(5)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                background: gray; /* Sets the entire slider track to blue */
            }
            QSlider::handle:horizontal {
                background: white; /* Handle color */
                border: 1px solid black;
                width: 10px;
                margin: -5px 0; /* Centers the handle on the groove */
                border-radius: 7px;
            }
            """)
        slider.setValue(50)
        slider.valueChanged.connect(lambda value: self.update_slider_labels(value, left_label, right_label))
        layout.addWidget(left_label)
        layout.addWidget(slider)
        layout.addWidget(right_label)
        return layout, slider

    @staticmethod
    def update_slider_labels(value, left_label, right_label):
        """Updates the labels for a slider to show switched percentage values."""
        left_label.setText(f"{left_label.text().split(':')[0]}: {100 - value}%")
        right_label.setText(f"{right_label.text().split(':')[0]}: {value}%")

    def validate_inputs(self):
        """
        Validates that all input fields are filled and the passwords match.

        Returns:
            bool: True if all inputs are valid, False otherwise.
        """
        errors = []

        # Access input values dynamically from the dictionaries
        text_inputs = {
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            "email": self.first_column_widgets["Email:"].text(),
        }

        numerous_inputs = {
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            # "climbing_freq": self.second_column_widgets["Climbing Frequency/week:"].value(),
            # "climbing_hours": self.second_column_widgets["Climbing Hours/week:"].value(),
        }

        select_inputs = {
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "ircra": self.second_column_widgets["IRCRA Scale Level:"].currentText(),
        }

        # Mapping field names to user-friendly labels
        field_labels = {
            "name": "Name",
            "surname": "Surname",
            "email": "Email",
            "gender": "Gender",
            "dominant_arm": "Dominant Arm",
            "ircra": "IRCRA Scale Level",
            "weight": "Weight",
            "height": "Height",
            "age": "Age",
            "climbing_freq": "Climbing Frequency/week",
            "climbing_hours": "Climbing Hours/week",
        }

        # Collect missing/invalid fields
        missing_fields = []

        for field_name, value in text_inputs.items():
            if isinstance(value, str) and not value.strip():
                missing_fields.append(field_labels[field_name])

        for field_name, value in select_inputs.items():
            if isinstance(value, str) and value == "-":
                missing_fields.append(field_labels[field_name])

        for field_name, value in numerous_inputs.items():
            if isinstance(value, int) and value <= 0:
                missing_fields.append(field_labels[field_name])

        # Special validation: Email format check
        email = self.first_column_widgets["Email:"].text().strip()
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            errors.append("Invalid email format.")

        # Display all errors in a single message
        if missing_fields:
            error_message = "The following fields must be filled:\n" + "\n".join(missing_fields)
            QMessageBox.warning(self, "Form Validation Error", error_message)
            return False  # Validation failed

        return True  # Validation passed

    def handle_registration(self):
        """
        Handles the registration process by collecting user data, validating inputs,
        and registering the user in the database.
        """

        self.is_saving = True  # Set the flag to indicate saving is in progress

        # Collect the percentages from the sliders with switched values
        bouldering_percentage = 100 - self.bouldering_lead_slider.value()
        lead_climbing_percentage = self.bouldering_lead_slider.value()

        climbing_indoor_percentage = 100 - self.indoor_outdoor_slider.value()
        climbing_outdoor_percentage = self.indoor_outdoor_slider.value()

        climber_data = {
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            "email": self.first_column_widgets["Email:"].text(),
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            "ircra": self.second_column_widgets["IRCRA Scale Level:"].currentText(),
            "years_climbing": self.second_column_widgets["Years of Climbing:"].value(),
            "climbing_freq": self.second_column_widgets["Climbing Frequency/week:"].value(),
            "climbing_hours": self.second_column_widgets["Climbing Hours/week:"].value(),
            "climbing_indoor": climbing_indoor_percentage,
            "climbing_outdoor": climbing_outdoor_percentage,
            "bouldering": bouldering_percentage,
            "lead_climbing": lead_climbing_percentage,
            "sport_other": self.second_column_widgets["Other sports:"].text(),
            "sport_freq": self.second_column_widgets["Sport Frequency/week:"].value(),
            "sport_activity_hours": self.second_column_widgets["Sport Activity (hours/week):"].value()
        }

        if not self.validate_inputs():
            return

        # Save the climber data if validation passes
        success = self.climber_db_manager.register_climber(self.admin_id, **climber_data)

        if success:
            QMessageBox.information(self, "Success", "Questioner successfully saved.")
            if hasattr(self, 'dialog'):
                self.dialog.close()
            editing = False
            self.switch_to_test_page(editing)  # Redirect to the main page or next step

        else:
            self.is_saving = False  # Reset the flag if saving fails
            QMessageBox.warning(self, "Error", "Saving information failed.")

    def closeEvent(self, event):
        """Intercepts the close event to confirm unsaved changes."""
        if not self.is_saving:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to leave without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
