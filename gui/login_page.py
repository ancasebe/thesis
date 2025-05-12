"""
Login interface module for the Climbing Testing Application.

This module defines the LoginPage class which provides the user interface for
authentication. It handles user input, credential verification, and navigation
to the main application after successful login.

Key functionalities:
- Display login form for username and password entry
- Authenticate user credentials against the database
- Provide feedback for failed login attempts
- Navigate to appropriate application view after login
- Support for administrator registration

The login page serves as the entry point to the application, ensuring that
only authorized users can access research data and testing capabilities.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFormLayout, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class LoginPage(QWidget):
    """
    A class representing the login page of the application.

    Methods:
        setup_ui: Sets up the user interface for the login page.
        handle_login: Handles user login, verifying credentials.
    """
    def __init__(self, switch_to_main, db_manager):
        """Initializes the login page with references to the switch functions and db_manager."""
        super().__init__()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.switch_to_main = switch_to_main
        self.db_manager = db_manager
        self.setup_ui()

    def setup_ui(self):
        """Sets up the layout and widgets for the login page."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Welcome Text
        welcome_label = QLabel("Welcome to the testing climbers app!")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(welcome_label)

        # Login Form
        form_layout = QFormLayout()

        self.username_input.setPlaceholderText("Enter your username")
        form_layout.addRow("Username:", self.username_input)

        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password:", self.password_input)

        layout.addLayout(form_layout)

        # Login Button
        login_button = QPushButton("Log In")
        login_button.clicked.connect(self.handle_login)
        layout.addWidget(login_button)

        # University Logos at the Bottom
        logos_layout = QHBoxLayout()
        logos_layout.setAlignment(Qt.AlignCenter)

        # Fixed height for both logos - increased for better quality
        fixed_height = 50  # Increased from 50 to 100 for better resolution

        # Logo 1
        logo1 = QLabel()
        pixmap1 = QPixmap('resources/logo_uibk.jpg')  # Replace with your logo path
        if pixmap1.isNull():
            logo1.setText("UIBK")
            logo1.setAlignment(Qt.AlignCenter)
        else:
            # Keep aspect ratio and use high-quality scaling
            pixmap1 = pixmap1.scaled(
                pixmap1.width() * fixed_height // pixmap1.height(),
                fixed_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            logo1.setPixmap(pixmap1)
            logo1.setAlignment(Qt.AlignCenter)
        logos_layout.addWidget(logo1)

        logo2 = QLabel()
        pixmap2 = QPixmap('resources/logo_uct.jpeg')  # Replace with your logo path
        if pixmap2.isNull():
            logo2.setText("UCT Prague")
            logo2.setAlignment(Qt.AlignCenter)
        else:
            # Keep aspect ratio and use high-quality scaling
            pixmap2 = pixmap2.scaled(
                pixmap2.width() * fixed_height // pixmap2.height(),
                fixed_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            logo2.setPixmap(pixmap2)
            logo2.setAlignment(Qt.AlignCenter)
        logos_layout.addWidget(logo2)

        layout.addLayout(logos_layout)

        self.setLayout(layout)

    def handle_login(self):
        """
        Handles user login by verifying the entered username and password.
        If the credentials are valid, it switches to the main page.
        """
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Please enter both username and password.")
            return

        if self.db_manager.verify_user(username, password):
            QMessageBox.information(self, "Login Successful", f"Welcome, {username}!")
            self.switch_to_main(username)
            # Clear inputs
            self.username_input.clear()
            self.password_input.clear()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")
