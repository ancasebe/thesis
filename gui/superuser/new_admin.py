from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFormLayout, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class RegistrationPage(QWidget):
    """
    A class representing the registration page of the application.

    Attributes:
        switch_to_main (function): Callback to switch to the main page after successful registration.
        db_manager (object): Manager for handling database interactions related to superuser.

    Methods:
        setup_ui: Sets up the user interface for the registration page.
        handle_registration: Processes registration, validates user input, and saves data.
        validate_inputs: Validates user inputs for completeness and correctness.
    """
    def __init__(self, switch_to_main, db_manager):
        super().__init__()
        self.widgets_column = {}
        self.switch_to_main = switch_to_main
        self.db_manager = db_manager
        self.setup_ui()

    def setup_ui(self):
        """Sets up the layout and widgets for the registration page of new admin."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Welcome Text
        welcome_label = QLabel("Create a New Admin Account")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(welcome_label)

        # Registration Form
        form_layout = QFormLayout()

        self.widgets_column = {
            "Username:": QLineEdit(),
            "Password:": QLineEdit(),
            "Confirm Password:": QLineEdit(),
            "Research Name:": QLineEdit()
        }

        # Adding widgets to the form layout
        for label, widget in self.widgets_column.items():
            form_layout.addRow(QLabel(label), widget)

        self.widgets_column["Password:"].setEchoMode(QLineEdit.Password)
        self.widgets_column["Confirm Password:"].setEchoMode(QLineEdit.Password)

        layout.addLayout(form_layout)

        # Register Button
        register_button = QPushButton("Register")
        register_button.clicked.connect(self.handle_registration)
        layout.addWidget(register_button)

        # Switch to Login
        switch_button = QPushButton("Back to all users")
        switch_button.clicked.connect(self.switch_to_main)
        layout.addWidget(switch_button)

        # University Logos at the Bottom
        logos_layout = QHBoxLayout()
        logos_layout.setAlignment(Qt.AlignCenter)

        # Logo 1
        logo1 = QLabel()
        pixmap1 = QPixmap('resources/university_logo1.png')
        if pixmap1.isNull():
            logo1.setText("Logo 1")
            logo1.setAlignment(Qt.AlignCenter)
        else:
            pixmap1 = pixmap1.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo1.setPixmap(pixmap1)
            logo1.setAlignment(Qt.AlignCenter)
        logos_layout.addWidget(logo1)

        # Logo 2
        logo2 = QLabel()
        pixmap2 = QPixmap('resources/university_logo2.png')
        if pixmap2.isNull():
            logo2.setText("Logo 2")
            logo2.setAlignment(Qt.AlignCenter)
        else:
            pixmap2 = pixmap2.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo2.setPixmap(pixmap2)
            logo2.setAlignment(Qt.AlignCenter)
        logos_layout.addWidget(logo2)

        layout.addLayout(logos_layout)

        self.setLayout(layout)

    def validate_inputs(self):
        """
        Validates that all input fields are filled and passwords match.

        Returns:
            bool: True if all inputs are valid, False otherwise.
        """
        inputs = {
            "username": self.widgets_column["Username:"].text().strip(),
            "password": self.widgets_column["Password:"].text(),
            "confirm_password": self.widgets_column["Confirm Password:"].text(),
            "research_name": self.widgets_column["Research Name:"].text(),
        }

        errors = []
        for field_name, value in inputs.items():
            if isinstance(value, str) and not value.strip():
                errors.append(f"{field_name.capitalize()} cannot be empty.")

        if inputs["password"] != inputs["confirm_password"]:
            errors.append("Passwords do not match.")

        if len(inputs["password"]) < 6:
            errors.append("Password must be at least 6 characters long.")

        if errors:
            QMessageBox.warning(self, "Form Validation Error", "\n".join(errors))
            return False
        return True

    def handle_registration(self):
        """
        Handles the registration process by collecting user data, validating inputs,
        and registering the user in the database.
        """
        username = self.widgets_column["Username:"].text().strip()
        password = self.widgets_column["Password:"].text()

        admin_data = {
            "research_name": self.widgets_column["Research Name:"].text().strip(),
        }

        if not self.validate_inputs():
            return

        success = self.db_manager.register_admin(username, password, **admin_data)
        if success:
            QMessageBox.information(self, "Registration Successful", "You can now log in.")
            self.switch_to_main(username)
        else:
            QMessageBox.warning(self, "Registration Failed", "Username already exists.")