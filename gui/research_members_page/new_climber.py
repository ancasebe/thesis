"""
This module defines the `RegistrationPage` class, which handles user registration.

Key functionalities:
- Displaying a form to collect user data for registration.
- Validating user inputs, such as ensuring password confirmation.
- Registering new users by saving their information to the database.
- Switching to the main application page upon successful registration.
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

    def __init__(self, admin_id, switch_to_research_members, db_manager):
        """Initializes the registration page with references to the switch function and db_manager."""
        super().__init__()
        self.first_column_widgets = {}
        self.second_column_widgets = {}
        self.bouldering_lead_slider = None
        self.indoor_outdoor_slider = None
        self.admin_id = admin_id
        self.switch_to_research_members = switch_to_research_members
        self.db_manager = db_manager
        self.setup_ui()

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
            "French Scale Level:": QComboBox(),
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

        self.second_column_widgets["French Scale Level:"].addItems(
            ["-", "5", "6a", "6a+", "6b", "6b+", "6c", "7a", "7a+", "7b", "7b+",
             "7c", "7c+", "8a", "8a+", "8b", "8b+", "8c", "8c+", "9a", "9a+", "9b", "9b+", "9c"]
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

        switch_button = QPushButton("Back to Research Members")
        switch_button.clicked.connect(self.switch_to_research_members)

        # Add the grid layout and buttons to the main layout
        layout.addLayout(grid_layout)
        layout.addWidget(submit_button)
        layout.addWidget(switch_button)

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
        }

        # Checking if required fields are filled
        for field_name, value in text_inputs.items():
            if isinstance(value, str) and not value.strip():
                errors.append(f"{field_name} cannot be empty.")

        # Special validation: Email format check
        email = self.first_column_widgets["Email:"].text().strip()
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            errors.append("Invalid email format.")

        # Check if there are any errors and display them
        if errors:
            QMessageBox.warning(self, "Form Validation Error", "\n".join(errors))
            return False  # Validation failed
        return True  # Validation passed

    def handle_registration(self):
        """
        Handles the registration process by collecting user data, validating inputs,
        and registering the user in the database.
        """

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
            "french_scale": self.second_column_widgets["French Scale Level:"].currentText(),
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
        success = self.db_manager.register_climber(self.admin_id, **climber_data)

        if success:
            QMessageBox.information(self, "Success", "Questioner successfully saved.")
            self.switch_to_research_members()  # Redirect to the main page or next step
        else:
            QMessageBox.warning(self, "Error", "Saving information failed.")