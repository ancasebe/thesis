"""
This module defines the `ResearchMembersPage` class, which provides an interface
for managing research members (climbers).

Key functionalities:
- Displays a list of climbers linked to the logged-in admin.
- Allows the admin to view, adjust, add, or delete climber information.
- Integrates with the database to manage climber data for the specific admin.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt
from gui.research_members.edit_climber_info import EditClimberInfoPage
from gui.research_members.new_climber import NewClimber


class ResearchMembersPage(QWidget):
    """
    ResearchMembersPage allows the admin to manage registered climbers.
    Admins can view, adjust, add, or delete climber information.

    Args:
        db_manager (SuperDatabaseManager): Database manager to handle data retrieval and updates.
        admin_id (int): ID of the logged-in admin.
        main_stacked_widget (QStackedWidget): The main stacked widget to manage page transitions.
    """

    def __init__(self, db_manager, admin_id, main_stacked_widget):
        super().__init__()
        self.db_manager = db_manager
        self.admin_id = admin_id
        self.main_stacked_widget = main_stacked_widget

        self.climber_selector = None
        self.setup_ui()
        self.load_climbers()

    def setup_ui(self):
        """Sets up the user interface for the Research Members Page."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Title
        title_label = QLabel("Manage Research Members")
        title_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(title_label)

        # ComboBox for selecting climbers
        self.climber_selector = QComboBox()
        main_layout.addWidget(self.climber_selector)

        # Buttons for managing climbers
        button_layout = QHBoxLayout()

        # Button to adjust climber info
        adjust_button = QPushButton("Adjust Information")
        adjust_button.clicked.connect(self.adjust_climber_info)
        button_layout.addWidget(adjust_button)

        # Button to delete a climber
        delete_button = QPushButton("Delete Member")
        delete_button.clicked.connect(self.delete_climber)
        button_layout.addWidget(delete_button)

        # Button to add a new member
        add_button = QPushButton("Add New Member")
        add_button.clicked.connect(self.show_new_climber_page)
        button_layout.addWidget(add_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def load_climbers(self):
        """Loads all climbers registered by the current admin into the ComboBox."""
        climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
        self.climber_selector.clear()
        for climber in climbers:
            # Display the name but store email (unique identifier)
            name_display = f"{climber['name']} {climber['surname']}"
            self.climber_selector.addItem(name_display, climber["email"])

    def adjust_climber_info(self):
        """Opens SettingsPage to adjust the selected climber's information."""
        email = self.climber_selector.currentData()  # Get email of the selected climber
        if email:
            adjust_climber_info_page = AdjustClimberInfoPage(self.admin_id, email, self.reload_members, self.db_manager)
            self.main_stacked_widget.addWidget(adjust_climber_info_page)
            self.main_stacked_widget.setCurrentWidget(adjust_climber_info_page)
        else:
            QMessageBox.warning(self, "No Member Selected", "Please select a member to adjust.")

    def show_new_climber_page(self):
        """Opens RegistrationPage to add a new climber."""
        new_climber = NewClimber(self.admin_id, self.reload_members, self.db_manager)
        self.main_stacked_widget.addWidget(new_climber)
        self.main_stacked_widget.setCurrentWidget(new_climber)

    def delete_climber(self):
        """Deletes the selected climber after confirmation."""
        email = self.climber_selector.currentData()
        name = self.climber_selector.currentText()
        if email:
            reply = QMessageBox.question(
                self, "Delete Member", f"Are you sure you want to delete member {name}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.db_manager.delete_climber(email, self.admin_id):
                    QMessageBox.information(self, "Success", f"Member {name} deleted successfully.")
                    self.load_climbers()  # Refresh the ComboBox
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete member.")

    def reload_members(self):
        """Reloads the members list after adding a new member."""
        self.load_climbers()
        self.main_stacked_widget.setCurrentWidget(self)
