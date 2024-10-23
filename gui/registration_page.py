"""
This module defines the `RegistrationPage` class, which handles user registration.

Key functionalities:
- Displaying a form to collect user data for registration.
- Validating user inputs, such as ensuring password confirmation.
- Registering new users by saving their information to the database.
- Switching to the main application page upon successful registration.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QLabel, QSlider,
    QComboBox, QPushButton, QGridLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt


class RegistrationPage(QWidget):
    """
    A class representing the registration page of the application.

    Methods:
        setup_ui: Sets up the user interface for the registration page.
        handle_registration: Handles the registration process, validating user input and saving data.
        validate_inputs: Validates the user inputs to ensure completeness and correctness.
    """

    def __init__(self, switch_to_main, switch_to_login, db_manager):
        """Initializes the registration page with references to the switch function and db_manager."""
        super().__init__()
        self.switch_to_main = switch_to_main
        self.switch_to_login = switch_to_login
        self.db_manager = db_manager
        self.setup_ui()

    def setup_ui(self):
        """Sets up the layout and widgets for the registration page."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Create a grid layout for organizing form in columns
        grid_layout = QGridLayout()

        # Username and Password fields
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)

        # Name, Surname, Email
        self.name_input = QLineEdit()
        self.surname_input = QLineEdit()
        self.email_input = QLineEdit()

        # Gender and Dominant Arm selection
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["-", "Male", "Female", "Other"])

        self.dominant_arm_combo = QComboBox()
        self.dominant_arm_combo.addItems(["-", "Left", "Right"])

        # Weight, Height, Age
        self.weight_input = QSpinBox()
        self.weight_input.setMinimum(0)
        self.weight_input.setMaximum(150)

        self.height_input = QSpinBox()
        self.height_input.setMinimum(0)
        self.height_input.setMaximum(220)

        self.age_input = QSpinBox()
        self.age_input.setMinimum(0)
        self.age_input.setMaximum(70)

        # French Scale Level
        self.french_scale_combo = QComboBox()
        self.french_scale_combo.addItems([
            "-", "5", "6a", "6a+", "6b", "6b+", "6c", "7a", "7a+", "7b", "7b+", "7c", "7c+",
            "8a", "8a+", "8b", "8b+", "8c", "8c+", "9a", "9a+", "9b", "9b+", "9c"
        ])

        # Years of Climbing
        self.years_climbing_input = QSpinBox()
        self.years_climbing_input.setMinimum(0)
        self.years_climbing_input.setMaximum(30)

        # Climbing Frequency, Hours per week, Sport Frequency, Sport Activity Hours
        self.climbing_freq_input = QSpinBox()
        self.climbing_freq_input.setMinimum(0)
        self.climbing_freq_input.setMaximum(7)

        self.climbing_hours_input = QSpinBox()
        self.climbing_hours_input.setMinimum(0)
        self.climbing_hours_input.setMaximum(50)

        self.sport_other = QLineEdit()

        self.sport_freq_input = QSpinBox()
        self.sport_freq_input.setMinimum(0)
        self.sport_freq_input.setMaximum(7)

        self.sport_activity_input = QSpinBox()
        self.sport_activity_input.setMinimum(0)
        self.sport_activity_input.setMaximum(50)

        # Sliders for Bouldering/Lead Climbing and Indoor/Outdoor Climbing percentages
        self.bouldering_lead_slider = QSlider(Qt.Horizontal)
        self.bouldering_lead_slider.setRange(0, 100)
        self.bouldering_lead_slider.setSingleStep(5)  # Step of 5%
        self.bouldering_lead_slider.setValue(50)  # Default: 50% Bouldering, 50% Lead Climbing
        self.bouldering_lead_slider.valueChanged.connect(self.update_bouldering_lead)

        self.climbing_indoor_outdoor_slider = QSlider(Qt.Horizontal)
        self.climbing_indoor_outdoor_slider.setRange(0, 100)
        self.climbing_indoor_outdoor_slider.setSingleStep(5)  # Step of 5%
        self.climbing_indoor_outdoor_slider.setValue(50)  # Default: 50% Indoor, 50% Outdoor
        self.climbing_indoor_outdoor_slider.valueChanged.connect(self.update_indoor_outdoor)

        # Labels for showing the percentages
        self.bouldering_label = QLabel("Bouldering: 50%")
        self.lead_climbing_label = QLabel("Lead Climbing: 50%")

        self.indoor_label = QLabel("Climbing Indoor: 50%")
        self.outdoor_label = QLabel("Climbing Outdoor: 50%")

        # First column (left side)
        grid_layout.addWidget(QLabel("Username:"), 0, 0)
        grid_layout.addWidget(self.username_input, 0, 1)

        grid_layout.addWidget(QLabel("Password:"), 1, 0)
        grid_layout.addWidget(self.password_input, 1, 1)

        grid_layout.addWidget(QLabel("Confirm Password:"), 2, 0)
        grid_layout.addWidget(self.confirm_password_input, 2, 1)

        grid_layout.addWidget(QLabel("Name:"), 3, 0)
        grid_layout.addWidget(self.name_input, 3, 1)

        grid_layout.addWidget(QLabel("Surname:"), 4, 0)
        grid_layout.addWidget(self.surname_input, 4, 1)

        grid_layout.addWidget(QLabel("Email:"), 5, 0)
        grid_layout.addWidget(self.email_input, 5, 1)

        grid_layout.addWidget(QLabel("Gender:"), 6, 0)
        grid_layout.addWidget(self.gender_combo, 6, 1)

        grid_layout.addWidget(QLabel("Dominant Arm:"), 7, 0)
        grid_layout.addWidget(self.dominant_arm_combo, 7, 1)

        # Weight, Height, Age on the left side
        grid_layout.addWidget(QLabel("Weight (kg):"), 8, 0)
        grid_layout.addWidget(self.weight_input, 8, 1)

        grid_layout.addWidget(QLabel("Height (cm):"), 9, 0)
        grid_layout.addWidget(self.height_input, 9, 1)

        grid_layout.addWidget(QLabel("Age (years):"), 10, 0)
        grid_layout.addWidget(self.age_input, 10, 1)

        # Second column (right side)
        grid_layout.addWidget(QLabel("French Scale Level:"), 0, 2)
        grid_layout.addWidget(self.french_scale_combo, 0, 3)

        grid_layout.addWidget(QLabel("Years of Climbing:"), 1, 2)
        grid_layout.addWidget(self.years_climbing_input, 1, 3)

        grid_layout.addWidget(QLabel("Climbing Frequency/week:"), 2, 2)
        grid_layout.addWidget(self.climbing_freq_input, 2, 3)

        grid_layout.addWidget(QLabel("Climbing Hours/week:"), 3, 2)
        grid_layout.addWidget(self.climbing_hours_input, 3, 3)

        # Add sliders for Bouldering/Lead Climbing and Indoor/Outdoor Climbing
        grid_layout.addWidget(QLabel("Bouldering and Lead Climbing:"), 4, 2)
        grid_layout.addWidget(self.bouldering_lead_slider, 4, 3)
        grid_layout.addWidget(self.bouldering_label, 5, 2)
        grid_layout.addWidget(self.lead_climbing_label, 5, 3)

        grid_layout.addWidget(QLabel("Indoor and Outdoor Climbing:"), 6, 2)
        grid_layout.addWidget(self.climbing_indoor_outdoor_slider, 6, 3)
        grid_layout.addWidget(self.indoor_label, 7, 2)
        grid_layout.addWidget(self.outdoor_label, 7, 3)

        grid_layout.addWidget(QLabel("Other sports (excluding climbing):"), 8, 2)
        grid_layout.addWidget(self.sport_other, 8, 3)

        grid_layout.addWidget(QLabel("Sport Frequency/week (excluding climbing):"), 9, 2)
        grid_layout.addWidget(self.sport_freq_input, 9, 3)

        grid_layout.addWidget(QLabel("Sport Activity (hours/week):"), 10, 2)
        grid_layout.addWidget(self.sport_activity_input, 10, 3)

        # Submit button
        submit_button = QPushButton("Register")
        submit_button.clicked.connect(self.handle_registration)

        switch_button = QPushButton("Already have an account? Log In")
        switch_button.setFlat(True)
        switch_button.clicked.connect(self.switch_to_login)
        layout.addWidget(switch_button)

        # Add the grid layout and the button to the main layout
        layout.addLayout(grid_layout)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def update_bouldering_lead(self):
        """Update the bouldering and lead climbing percentages based on the slider value."""
        bouldering_percentage = self.bouldering_lead_slider.value()
        lead_climbing_percentage = 100 - bouldering_percentage
        self.bouldering_label.setText(f"Bouldering: {bouldering_percentage}%")
        self.lead_climbing_label.setText(f"Lead Climbing: {lead_climbing_percentage}%")

    def update_indoor_outdoor(self):
        """Update the climbing indoor and outdoor percentages based on the slider value."""
        indoor_percentage = self.climbing_indoor_outdoor_slider.value()
        outdoor_percentage = 100 - indoor_percentage
        self.indoor_label.setText(f"Climbing Indoor: {indoor_percentage}%")
        self.outdoor_label.setText(f"Climbing Outdoor: {outdoor_percentage}%")

    def validate_inputs(self):
        """
        Validates that all input fields are filled and the passwords match.

        Returns:
            bool: True if all inputs are valid, False otherwise.
        """
        errors = []

        # Dictionary holding the input widgets and their corresponding validation checks
        text_inputs = {
            "username": self.username_input.text(),
            "password": self.password_input.text(),
            "confirm_password": self.confirm_password_input.text(),
            "name": self.name_input.text(),
            "surname": self.surname_input.text(),
            "email": self.email_input.text(),
        }

        numerous_inputs = {
            "weight": self.weight_input.value(),  # Using QSpinBox value
            "height": self.height_input.value(),
            "age": self.age_input.value(),
            "years_climbing": self.years_climbing_input.value(),
            "climbing_freq": self.climbing_freq_input.value(),
            "climbing_hours": self.climbing_hours_input.value(),
        }

        select_inputs = {
            "gender": self.gender_combo.currentText(),
            "dominant_arm": self.dominant_arm_combo.currentText(),
            "french_scale": self.french_scale_combo.currentText(),
        }

        # Checking if required fields are filled
        for field_name, value in text_inputs.items():
            # Example validation for text fields (shouldn't be empty)
            if isinstance(value, str) and not value.strip():
                errors.append(f"{field_name} cannot be empty.")

        for field_name, value in select_inputs.items():
            # Example validation for text fields (shouldn't be empty)
            if isinstance(value, str) and value == "-":
                errors.append(f"{field_name} must be filled.")

        for field_name, value in numerous_inputs.items():
            # Example validation for numeric fields (shouldn't be zero or negative)
            if isinstance(value, int) and value <= 0:
                errors.append(f"{field_name} must be greater than zero.")

        password = self.password_input.text()

        # Special validation: check if passwords match
        if password != self.confirm_password_input.text():
            errors.append("Passwords do not match.")
        else:
            if len(password) < 6 and len(password) != 0:
                errors.append("Password must be at least 6 characters long.")

        # Check if there are any errors and display them
        if errors:
            QMessageBox.warning(self, "Form Validation Error", "\n".join(errors))
            return False  # Validation failed
        # else:
        return True  # Validation passed

    def handle_registration(self):
        """
        Handles the registration process by collecting user data, validating inputs,
        and registering the user in the database.
        """
        username = self.username_input.text().strip()
        password = self.password_input.text()

        # Collect the percentages from the sliders
        climbing_indoor_percentage = self.bouldering_lead_slider.value()
        climbing_outdoor_percentage = 100 - climbing_indoor_percentage

        bouldering_percentage = self.climbing_indoor_outdoor_slider.value()
        lead_climbing_percentage = 100 - bouldering_percentage

        user_data = {
            "confirm_password": self.confirm_password_input.text(),
            "name": self.name_input.text(),
            "surname": self.surname_input.text(),
            "email": self.email_input.text(),
            "gender": self.gender_combo.currentText(),
            "dominant_arm": self.dominant_arm_combo.currentText(),
            "weight": self.weight_input.value(),  # Using QSpinBox value
            "height": self.height_input.value(),
            "age": self.age_input.value(),
            "french_scale": self.french_scale_combo.currentText(),
            "years_climbing": self.years_climbing_input.value(),
            "climbing_freq": self.climbing_freq_input.value(),
            "climbing_hours": self.climbing_hours_input.value(),
            "climbing_indoor": climbing_indoor_percentage,
            "climbing_outdoor": climbing_outdoor_percentage,
            "bouldering": bouldering_percentage,
            "lead_climbing": lead_climbing_percentage,
            "sport_other": self.sport_other.text(),
            "sport_freq": self.sport_freq_input.value(),
            "sport_activity_hours": self.sport_activity_input.value()
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
