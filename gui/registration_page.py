from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QLabel,
    QComboBox, QPushButton, QGridLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt


class RegistrationPage(QWidget):
    def __init__(self, switch_to_main, switch_to_login, db_manager):
        super().__init__()
        self.switch_to_main = switch_to_main
        self.switch_to_login = switch_to_login
        self.db_manager = db_manager
        self.setup_ui()

    def setup_ui(self):
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

        # self.arm_tested_combo = QComboBox()
        # self.arm_tested_combo.addItems(["Dominant", "Non-Dominant"])

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
        self.french_scale_combo.addItems(["-", "5", "6a", "6a+", "6b", "6b+", "6c", "7a", "7a+", "7b", "7b+", "7c",
                                          "7c+", "8a", "8a+", "8b", "8b+", "8c", "8c+", "9a", "9a+", "9b", "9b+", "9c"])

        # Years of Climbing
        self.years_climbing_input = QSpinBox()
        self.years_climbing_input.setMinimum(0)
        self.years_climbing_input.setMaximum(30)

        # Time spent climbing indoors (%)
        self.climbing_indoor_input = QSpinBox()
        self.climbing_indoor_input.setMinimum(0)
        self.climbing_indoor_input.setMaximum(100)

        # Time spent lead climbing (%)
        self.lead_climbing_slider = QSpinBox()
        self.lead_climbing_slider.setMinimum(0)
        self.lead_climbing_slider.setMaximum(100)

        # Time spent bouldering (%)
        self.bouldering_input = QSpinBox()
        self.bouldering_input.setMinimum(0)
        self.bouldering_input.setMaximum(100)

        self.sport_other = QLineEdit()
        # self.sport_other.addItems(["Running", "Cycling", "Swimming", "Ski mountaineering", "Skiing", "Snowboarding",
        #                            "Surfing", "Yoga", "Volleyball", "Basketball", "Other"])

        # Climbing Frequency, Hours per week, Sport Frequency, Sport Activity Hours
        self.climbing_freq_input = QSpinBox()
        self.climbing_freq_input.setMinimum(0)
        self.climbing_freq_input.setMaximum(7)

        self.climbing_hours_input = QSpinBox()
        self.climbing_hours_input.setMinimum(0)
        self.climbing_hours_input.setMaximum(50)

        self.sport_freq_input = QSpinBox()
        self.sport_freq_input.setMinimum(0)
        self.sport_freq_input.setMaximum(7)

        self.sport_activity_input = QSpinBox()
        self.sport_activity_input.setMinimum(0)
        self.sport_activity_input.setMaximum(50)

        # Organize the inputs into a 2-column layout using grid layout

        # First column
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

        grid_layout.addWidget(QLabel("Weight (kg):"), 0, 2)
        grid_layout.addWidget(self.weight_input, 0, 3)

        grid_layout.addWidget(QLabel("Height (cm):"), 1, 2)
        grid_layout.addWidget(self.height_input, 1, 3)

        grid_layout.addWidget(QLabel("Age (years):"), 2, 2)
        grid_layout.addWidget(self.age_input, 2, 3)

        # Second column

        grid_layout.addWidget(QLabel("French Scale Level:"), 3, 2)
        grid_layout.addWidget(self.french_scale_combo, 3, 3)

        grid_layout.addWidget(QLabel("Years of Climbing:"), 4, 2)
        grid_layout.addWidget(self.years_climbing_input, 4, 3)

        grid_layout.addWidget(QLabel("Time spent climbing indoor (%):"), 5, 2)
        grid_layout.addWidget(self.climbing_indoor_input, 5, 3)

        grid_layout.addWidget(QLabel("Time spent lead climbing (%):"), 6, 2)
        grid_layout.addWidget(self.lead_climbing_slider, 6, 3)

        grid_layout.addWidget(QLabel("Time spent bouldering (%):"), 7, 2)
        grid_layout.addWidget(self.bouldering_input, 7, 3)

        grid_layout.addWidget(QLabel("Other sports:"), 8, 2)
        grid_layout.addWidget(self.sport_other, 8, 3)

        grid_layout.addWidget(QLabel("Climbing Frequency/week:"), 9, 2)
        grid_layout.addWidget(self.climbing_freq_input, 9, 3)

        grid_layout.addWidget(QLabel("Climbing Hours/week:"), 10, 2)
        grid_layout.addWidget(self.climbing_hours_input, 10, 3)

        grid_layout.addWidget(QLabel("Sport Frequency/week (excluding climbing):"), 11, 2)
        grid_layout.addWidget(self.sport_freq_input, 11, 3)

        grid_layout.addWidget(QLabel("Sport Activity (hours/week):"), 12, 2)
        grid_layout.addWidget(self.sport_activity_input, 12, 3)

        # Submit button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.handle_registration)

        # Add the grid layout and the button to the main layout
        layout.addLayout(grid_layout)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def validate_inputs(self):
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
        username = self.username_input.text().strip()
        password = self.password_input.text()

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
            "climbing_indoor": self.climbing_indoor_input.value(),
            "lead_climbing": self.lead_climbing_slider.value(),
            "bouldering": self.bouldering_input.value(),
            "sport_other": self.sport_other.text(),
            "climbing_freq": self.climbing_freq_input.value(),
            "climbing_hours": self.climbing_hours_input.value(),
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

