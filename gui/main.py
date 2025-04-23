"""
This module is the entry point of the application, providing the main window that handles page
transitions between login, registration, and the main application interface.

Key functionalities:
- Manages the application's main window and page transitions.
- Switches between login, registration, and the main page.
- Handles user logout and application shutdown.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from gui.superuser.login_db_manager import LoginDatabaseManager
from gui.login_page import LoginPage
from gui.superuser.new_admin import RegistrationPage
from gui.main_page import MainPage


class MainWindow(QMainWindow):
    """
    The main window class that handles page transitions between login, registration,
    and the main application interface.

    Attributes:
        login_db_manager (DatabaseManager): The database manager for handling user data.
        stacked_widget (QStackedWidget): A stacked widget to switch between pages.
    """

    def __init__(self):
        """Initializes the main window, setting up page transitions and database connection."""

        super().__init__()
        self.login_db_manager = LoginDatabaseManager()
        self.login_db_manager.register_superuser()  # Ensure default superuser exists

        self.setWindowTitle("Testing climbers app")
        self.setMinimumSize(800, 600)  # Suitable starting size

        # Stacked Widget to hold different pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialize Pages
        self.login_page = LoginPage(
            switch_to_main=self.show_main_app,
            switch_to_register=self.show_registration,
            db_manager=self.login_db_manager
        )
        self.registration_page = RegistrationPage(
            switch_to_main=self.show_main_app,
            db_manager=self.login_db_manager
        )

        # Add Pages to Stacked Widget
        self.stacked_widget.addWidget(self.login_page)          # Index 0
        self.stacked_widget.addWidget(self.registration_page)   # Index 1

        # Start with Login Page
        self.stacked_widget.setCurrentIndex(0)

    def create_application_widget(self, username, admin_id):
        """
        Creates the main application interface after successful login or registration.

        Args
            username (str): The username of the logged-in user.
            admin_id (int): The ID of the logged-in admin, passed to MainPage and other components.
        """
        app_widget = MainPage(username, admin_id, self.logout)
        self.stacked_widget.addWidget(app_widget)  # Index 2
        self.stacked_widget.setCurrentWidget(app_widget)

    def show_main_app(self, username):
        """
        Switches to the main application after login or registration by retrieving the admin_id.

        Args:
            username (str): The username of the logged-in admin.
        """
        admin_id = self.login_db_manager.get_admin_id(username)
        if admin_id is not None:
            self.create_application_widget(username, admin_id)
        else:
            QMessageBox.warning(self, "Error", "Failed to retrieve admin ID.")

    def show_login(self):
        """Switches to the login page."""
        self.stacked_widget.setCurrentIndex(0)

    def show_registration(self):
        """Switches to the registration page."""
        self.stacked_widget.setCurrentIndex(1)

    def logout(self):
        """
        Logs the user out, asking for confirmation and returning to the login page if confirmed.
        """
        reply = QMessageBox.question(
            self,
            'Log Out',
            "Are you sure you want to log out?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Remove the main application widget and return to login
            if self.stacked_widget.count() > 2:
                main_app_widget = self.stacked_widget.widget(2)
                self.stacked_widget.removeWidget(main_app_widget)
                main_app_widget.deleteLater()
            self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        """
        Ensures the database connection is closed before closing the application.
        """
        self.login_db_manager.close()
        event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
