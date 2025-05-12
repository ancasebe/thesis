"""
Main application module for the Climbing Testing Application.

This module initializes the application, sets up the main window, and coordinates
navigation between different application views. It serves as the entry point for
the application and orchestrates the overall application flow.

Key functionalities:
- Initialize the application environment
- Set up the main application window
- Manage page navigation and transitions
- Coordinate database connections
- Handle application startup and shutdown

The main module ties together all application components and provides
the overall structure for the climbing research software.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from gui.superuser.login_db_manager import LoginDatabaseManager
from gui.login_page import LoginPage
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
            db_manager=self.login_db_manager
        )

        # Add Pages to Stacked Widget
        self.stacked_widget.addWidget(self.login_page)          # Index 0

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

    # def show_login(self):
    #     """Switches to the login page."""
    #     self.stacked_widget.setCurrentIndex(0)
    #
    # def show_registration(self):
    #     """Switches to the registration page."""
    #     self.stacked_widget.setCurrentIndex(1)

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
        Ensures the database connection is closed and all threads are stopped 
        before closing the application.
        """
        # Ensure any active threads are stopped
        for i in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(i)
            if hasattr(widget, 'closeEvent'):
                widget.closeEvent(event)
        
        # Close db connection
        self.login_db_manager.close()
        
        # If any widget requested to ignore the close event
        if event.isAccepted():
            print("Application closing - all threads stopped")
        
        event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()