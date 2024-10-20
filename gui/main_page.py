from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QSizePolicy, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


def create_placeholder_page(title):
    """Creates a placeholder page for other sections."""
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
    def __init__(self, username, logout_callback):
        super().__init__()
        self.username = username
        self.logout_callback = logout_callback
        self.setup_ui()

    def setup_ui(self):
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
    def __init__(self, username, logout_callback):
        super().__init__()
        self.username = username
        self.logout_callback = logout_callback
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Navigation menu on the left
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setAlignment(Qt.AlignTop)
        nav_widget.setLayout(nav_layout)
        nav_widget.setFixedWidth(150)  # Fixed width for navigation

        # Navigation buttons
        buttons = {
            "Calibration": self.show_calibration,
            "Testing": self.show_testing,
            "Training": self.show_training,
            "Results": self.show_results,
            "Settings": self.show_settings,
            "About": self.show_about,
            "Log Out": self.logout_callback  # Added Log Out button at the end
        }

        for btn_text, btn_slot in buttons.items():
            button = QPushButton(btn_text)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(btn_slot)
            nav_layout.addWidget(button)

        # Add stretch to push buttons to the top (excluding Log Out)
        nav_layout.addStretch()

        # Stacked widget to hold different application pages
        self.app_stacked_widget = QStackedWidget()
        self.main_app_page = MainAppPage(self.username, self.logout_callback)
        self.app_stacked_widget.addWidget(self.main_app_page)  # Index 0
        self.app_stacked_widget.addWidget(create_placeholder_page("Calibration"))  # Index 1
        self.app_stacked_widget.addWidget(create_placeholder_page("Testing"))      # Index 2
        self.app_stacked_widget.addWidget(create_placeholder_page("Training"))     # Index 3
        self.app_stacked_widget.addWidget(create_placeholder_page("Results"))      # Index 4
        self.app_stacked_widget.addWidget(create_placeholder_page("Settings"))     # Index 5
        self.app_stacked_widget.addWidget(create_placeholder_page("About"))        # Index 6

        # Add navigation and app stacked widget to main layout
        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.app_stacked_widget)

    # Navigation Methods for Main Application
    def show_calibration(self):
        self.app_stacked_widget.setCurrentIndex(1)

    def show_testing(self):
        self.app_stacked_widget.setCurrentIndex(2)

    def show_training(self):
        self.app_stacked_widget.setCurrentIndex(3)

    def show_results(self):
        self.app_stacked_widget.setCurrentIndex(4)

    def show_settings(self):
        self.app_stacked_widget.setCurrentIndex(5)

    def show_about(self):
        self.app_stacked_widget.setCurrentIndex(6)
