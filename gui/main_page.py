"""
This module defines the main application page that users see after logging in.

Key functionalities:
- Displays a welcome message and an image.
- Provides a placeholder for sections like Calibration, Testing, and Results.
- Handles user logout functionality through a callback.
"""

# pylint: disable=E1101

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QSizePolicy, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from test_page.test_page import TestPage
from user_info.setting_page import SettingsPage
from db_manager import DatabaseManager


def create_placeholder_page(title):
    """
    Creates a placeholder page for sections that are not yet implemented.

    Args:
        title (str): The title of the placeholder page.

    Returns:
        QWidget: A widget with a centered label displaying the title.
    """
    page = QWidget()
    layout = QVBoxLayout()
    layout.setAlignment(Qt.AlignCenter)
    page.setLayout(layout)

    label = QLabel(f"{title} Page")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("font-size: 20px;")
    layout.addWidget(label)

    return page


class MainAppPage(QWidget):
    """
    The main application page where users navigate to different pages like Calibration or Testing.

    Args:
        username (str): The username of the currently logged-in user.
        logout_callback (function): A callback function to log the user out.
    """
    def __init__(self, username, logout_callback):
        super().__init__()
        self.username = username
        self.logout_callback = logout_callback
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface for the main application page.
        Includes a welcome message and an image placeholder.
        """
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Welcome Text
        welcome_label = QLabel(f"User {self.username} is logged in.")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(welcome_label)

        # Image in the Middle
        image_label = QLabel()
        pixmap = QPixmap('resources/main_image.png')  # Replace with your main image path
        if pixmap.isNull():
            image_label.setText("Main Image Not Found.")
            image_label.setAlignment(Qt.AlignCenter)
        else:
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        self.setLayout(layout)


class MainPage(QWidget):
    """
    MainPage is the primary interface after the user logs in. It provides a navigation menu
    on the left, allowing the user to switch between different sections of the application
    such as Calibration, Testing, Training, Results, Settings, and About.

    Args:
        username (str): The username of the logged-in user.
        logout_callback (function): The function to call when the user clicks the "Log Out" button.
    """

    def __init__(self, username, logout_callback):
        """
        Initializes the MainPage with a username and a logout callback function.

        Args:
            username (str): The username of the logged-in user.
            logout_callback (function): The function to call when the user logs out.
        """
        super().__init__()
        self.username = username
        self.logout_callback = logout_callback
        # Stacked widget to hold different application pages
        self.app_stacked_widget = QStackedWidget()
        self.setup_ui()
        # Add various sections (pages) to the stacked widget
        self.main_app_page = MainAppPage(self.username, self.logout_callback)
        self.db_manager = DatabaseManager
        self.test_page = TestPage(self.username, self.logout_callback)
        self.setting_page = SettingsPage(self.username, self.db_manager)

        self.app_stacked_widget.addWidget(self.main_app_page)  # Index 0
        self.app_stacked_widget.addWidget(create_placeholder_page("Calibration"))  # Index 1
        self.app_stacked_widget.addWidget(self.test_page)      # Index 2
        self.app_stacked_widget.addWidget(create_placeholder_page("Training"))     # Index 3
        self.app_stacked_widget.addWidget(create_placeholder_page("Results"))      # Index 4
        self.app_stacked_widget.addWidget(self.setting_page)     # Index 5
        self.app_stacked_widget.addWidget(create_placeholder_page("About"))        # Index 6

    def setup_ui(self):
        """
        Sets up the user interface for the main page, including:
        - A navigation menu on the left for different sections.
        - A stacked widget on the right that switches between pages based on the selected section.
        """
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Navigation menu on the left
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setAlignment(Qt.AlignTop)
        nav_widget.setLayout(nav_layout)
        nav_widget.setFixedWidth(150)  # Fixed width for navigation

        # Navigation buttons for various sections
        buttons = {
            "Calibration": self.show_calibration,
            "Testing": self.show_testing,
            "Training": self.show_training,
            "Results": self.show_results,
            "Settings": self.show_settings,
            "About": self.show_about,
            "Log Out": self.logout_callback  # Log Out button at the end
        }

        # Add buttons to the navigation layout
        for btn_text, btn_slot in buttons.items():
            button = QPushButton(btn_text)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(btn_slot)
            nav_layout.addWidget(button)

        # Add stretch to push buttons to the top (except Log Out)
        nav_layout.addStretch()

        # self.app_stacked_widget.addWidget(self.main_app_page)  # Index 0
        # self.app_stacked_widget.addWidget(create_placeholder_page("Calibration"))  # Index 1
        # self.app_stacked_widget.addWidget(self.test_page)      # Index 2
        # self.app_stacked_widget.addWidget(create_placeholder_page("Training"))     # Index 3
        # self.app_stacked_widget.addWidget(create_placeholder_page("Results"))      # Index 4
        # self.app_stacked_widget.addWidget(create_placeholder_page("Settings"))     # Index 5
        # self.app_stacked_widget.addWidget(create_placeholder_page("About"))        # Index 6

        # Add navigation and stacked widget to the main layout
        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.app_stacked_widget)

    # Navigation Methods for Main Application

    def show_calibration(self):
        """
        Switches the current view to the Calibration section (index 1).
        """
        self.app_stacked_widget.setCurrentIndex(1)

    def show_testing(self):
        """
        Switches the current view to the Testing section (index 2).
        """
        self.app_stacked_widget.setCurrentIndex(2)

    def show_training(self):
        """
        Switches the current view to the Training section (index 3).
        """
        self.app_stacked_widget.setCurrentIndex(3)

    def show_results(self):
        """
        Switches the current view to the Results section (index 4).
        """
        self.app_stacked_widget.setCurrentIndex(4)

    def show_settings(self):
        """
        Switches the current view to the Settings section (index 5).
        """
        self.app_stacked_widget.setCurrentIndex(5)

    def show_about(self):
        """
        Switches the current view to the About section (index 6).
        """
        self.app_stacked_widget.setCurrentIndex(6)
