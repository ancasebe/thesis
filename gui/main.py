import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from db_manager import DatabaseManager
from login_page import LoginPage
from registration_page import RegistrationPage
from main_page import MainPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.setWindowTitle("Log in page")
        self.setMinimumSize(800, 600)  # Suitable starting size

        # Stacked Widget to hold different pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialize Pages
        self.login_page = LoginPage(
            switch_to_main=self.show_main_app,
            switch_to_register=self.show_registration,
            db_manager=self.db_manager
        )
        self.registration_page = RegistrationPage(
            switch_to_main=self.show_main_app,
            switch_to_login=self.show_login,
            db_manager=self.db_manager
        )

        # Add Pages to Stacked Widget
        self.stacked_widget.addWidget(self.login_page)          # Index 0
        self.stacked_widget.addWidget(self.registration_page)   # Index 1

        # Start with Login Page
        self.stacked_widget.setCurrentIndex(0)

    def create_application_widget(self, username):
        """Creates the main application interface with navigation."""
        app_widget = MainPage(username, self.logout)
        self.stacked_widget.addWidget(app_widget)  # Index 2
        self.stacked_widget.setCurrentWidget(app_widget)

    def show_main_app(self, username):
        """Switches to the main application after login or registration."""
        self.create_application_widget(username)

    def show_login(self):
        self.stacked_widget.setCurrentIndex(0)

    def show_registration(self):
        self.stacked_widget.setCurrentIndex(1)

    def logout(self):
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
        """Handle the window close event to ensure the database is closed."""
        self.db_manager.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
