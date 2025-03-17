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
    QStackedWidget, QSizePolicy
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from test_page.test_page import TestPage
from gui.superuser.login_db_manager import LoginDatabaseManager
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.superuser.new_admin import RegistrationPage


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
    def __init__(self, username, admin_id, logout_callback):
        super().__init__()
        self.username = username
        self.admin_id = admin_id
        self.logout_callback = logout_callback
        # self.is_superuser = is_superuser
        self.app_stacked_widget = QStackedWidget()
        self.widget_mapping = {}  # To track added widgets
        self.setup_ui()

        # Add main app page initially
        self.main_app_page = MainAppPage(self.username, self.logout_callback)
        self.add_widget_to_stack(self.main_app_page, "main")
        # print(self.admin_id)

    def add_widget_to_stack(self, widget, key):
        """
        Adds a widget to the stacked widget if not already added.
        """
        if key not in self.widget_mapping:
            self.app_stacked_widget.addWidget(widget)
            self.widget_mapping[key] = widget

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
            "Testing": self.show_testing,
            "Training": self.show_training,
            "Results": self.show_results,
            "Settings": self.show_settings,
            "About": self.show_about,
            "Log Out": self.logout_callback  # Log Out button at the end
        }

        # Add the "Handle Admins" button only if the user is a superuser
        if self.admin_id == 1:
            buttons["Handle Admins"] = self.show_handle_admins

        # Add buttons to the navigation layout
        for btn_text, btn_slot in buttons.items():
            button = QPushButton(btn_text)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setFocusPolicy(Qt.NoFocus)  # Disable focus for this button
            button.clicked.connect(btn_slot)
            nav_layout.addWidget(button)

        # Add stretch to push buttons to the top (except Log Out)
        nav_layout.addStretch()

        # Add navigation and stacked widget to the main layout
        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.app_stacked_widget)

    def show_testing(self):
        """
        Switch to the Testing page.
        """
        if "testing" not in self.widget_mapping:
            # Instantiate the database manager
            climber_db_manager = ClimberDatabaseManager()

            test_page = TestPage(db_manager=climber_db_manager,
                                 admin_id=self.admin_id,
                                 # is_superuser=self.is_superuser,
                                 main_stacked_widget=self.app_stacked_widget)
            self.add_widget_to_stack(test_page, "testing")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["testing"])

    def show_training(self):
        """
        Switch to the Training page.
        """
        if "training" not in self.widget_mapping:
            training_page = create_placeholder_page("Training")
            self.add_widget_to_stack(training_page, "training")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["training"])

    def show_results(self):
        """
        Switch to the Results page.
        """
        if "results" not in self.widget_mapping:
            results_page = create_placeholder_page("Results")
            self.add_widget_to_stack(results_page, "results")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["results"])

    def show_handle_admins(self):
        """
        Switch to the Handle Admins page. Only for superusers.
        """
        if "handle_admins" not in self.widget_mapping:
            login_db_manager = LoginDatabaseManager()
            handle_admins_page = RegistrationPage(
                switch_to_main=self.show_about,
                db_manager=login_db_manager
            )
            self.add_widget_to_stack(handle_admins_page, "handle_admins")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["handle_admins"])

    def show_settings(self):
        """
        Switch to the Settings page.
        """
        if "settings" not in self.widget_mapping:
            settings_page = create_placeholder_page("Settings")
            self.add_widget_to_stack(settings_page, "settings")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["settings"])

    def show_about(self):
        """
        Switch to the About page.
        """
        if "about" not in self.widget_mapping:
            about_page = create_placeholder_page("About")
            self.add_widget_to_stack(about_page, "about")
        self.app_stacked_widget.setCurrentWidget(self.widget_mapping["about"])

