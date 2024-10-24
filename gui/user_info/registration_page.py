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
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QSpinBox, QLabel, QSlider,
    QComboBox, QPushButton, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt


class RegistrationPage(QWidget):
    """
    A class representing the registration page of the application.

    Methods:
        setup_ui: Sets up the user interface for the registration page.
        handle_registration: Handles the registration process, validating user input, saving data.
        validate_inputs: Validates the user inputs to ensure completeness and correctness.
    """

    def __init__(self, switch_to_main, switch_to_login, db_manager):
        """Initializes the registration page with references to the switch function and db_manager."""
        super().__init__()
        self.first_column_widgets = {}
        self.second_column_widgets = {}
        self.switch_to_main = switch_to_main
        self.switch_to_login = switch_to_login
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
            "Username:": QLineEdit(),
            "Password:": QLineEdit(),
            "Confirm Password:": QLineEdit(),
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
            "Bouldering and Lead Climbing:": QSlider(Qt.Horizontal),
            "Indoor and Outdoor Climbing:": QSlider(Qt.Horizontal),
            "Other sports:": QLineEdit(),
            "Sport Frequency/week:": QSpinBox(),
            "Sport Activity (hours/week):": QSpinBox()
        }

        # Configuring certain widgets
        self.first_column_widgets["Password:"].setEchoMode(QLineEdit.Password)
        self.first_column_widgets["Confirm Password:"].setEchoMode(QLineEdit.Password)
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
        self.second_column_widgets["Bouldering and Lead Climbing:"].setRange(0, 100)
        self.second_column_widgets["Bouldering and Lead Climbing:"].setSingleStep(5)
        self.second_column_widgets["Bouldering and Lead Climbing:"].setValue(50)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setRange(0, 100)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setSingleStep(5)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setValue(50)

        # Adding widgets to the grid layout in the first column (left)
        for i, (label, widget) in enumerate(self.first_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 0)
            grid_layout.addWidget(widget, i, 1)

        # Adding widgets to the grid layout in the second column (right)
        for i, (label, widget) in enumerate(self.second_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 2)
            grid_layout.addWidget(widget, i, 3)

        # Submit and switch button
        submit_button = QPushButton("Register")
        submit_button.clicked.connect(self.handle_registration)

        switch_button = QPushButton("Already have an account? Log In")
        switch_button.clicked.connect(self.switch_to_login)

        # Add the grid layout and buttons to the main layout
        layout.addLayout(grid_layout)
        layout.addWidget(submit_button)
        layout.addWidget(switch_button)

        self.setLayout(layout)

    def validate_inputs(self):
        """
        Validates that all input fields are filled and the passwords match.

        Returns:
            bool: True if all inputs are valid, False otherwise.
        """
        errors = []

        # Access input values dynamically from the dictionaries
        text_inputs = {
            "username": self.first_column_widgets["Username:"].text(),
            "password": self.first_column_widgets["Password:"].text(),
            "confirm_password": self.first_column_widgets["Confirm Password:"].text(),
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            "email": self.first_column_widgets["Email:"].text(),
        }

        numerous_inputs = {
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            "years_climbing": self.second_column_widgets["Years of Climbing:"].value(),
            "climbing_freq": self.second_column_widgets["Climbing Frequency/week:"].value(),
            "climbing_hours": self.second_column_widgets["Climbing Hours/week:"].value(),
        }

        select_inputs = {
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "french_scale": self.second_column_widgets["French Scale Level:"].currentText(),
        }

        # Checking if required fields are filled
        for field_name, value in text_inputs.items():
            if isinstance(value, str) and not value.strip():
                errors.append(f"{field_name} cannot be empty.")

        for field_name, value in select_inputs.items():
            if isinstance(value, str) and value == "-":
                errors.append(f"{field_name} must be filled.")

        for field_name, value in numerous_inputs.items():
            if isinstance(value, int) and value <= 0:
                errors.append(f"{field_name} must be greater than zero.")

        password = self.first_column_widgets["Password:"].text()

        # Special validation: check if passwords match
        if password != self.first_column_widgets["Confirm Password:"].text():
            errors.append("Passwords do not match.")
        else:
            if len(password) < 6 and len(password) != 0:
                errors.append("Password must be at least 6 characters long.")

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
        username = self.first_column_widgets["Username:"].text().strip()
        password = self.first_column_widgets["Password:"].text()

        # Collect the percentages from the sliders
        bouldering_percentage = self.second_column_widgets["Bouldering and Lead Climbing:"].value()
        lead_climbing_percentage = 100 - bouldering_percentage

        climbing_indoor_percentage = self.second_column_widgets["Indoor and Outdoor Climbing:"].value()
        climbing_outdoor_percentage = 100 - climbing_indoor_percentage

        user_data = {
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
            return  # Stop registration if validation fails

        # Proceed with saving the user data if validation passes and username is unique
        success = self.db_manager.register_user(username, password, **user_data)

        if success:
            QMessageBox.information(self, "Success", "Registration successful!")
            self.switch_to_main(username)  # Redirect to the main page or next step
        else:
            QMessageBox.warning(self, "Error", "Username already exists or registration failed.")
