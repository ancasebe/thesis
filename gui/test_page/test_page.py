"""
This module defines the `TestPage` class, which provides a unified interface for managing climbers
and conducting tests.

Key functionalities:
- Allows admins to add, edit, and delete climbers.
- Enables selection of a test, climber, and arm for testing.
- Ensures validation before conducting a test and saves test data to the database.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QFormLayout, QStackedWidget, QSizePolicy, QDialog, QGridLayout
)
from PySide6.QtCore import Qt

from gui.research_members.edit_climber_info import EditClimberInfoPage
from gui.research_members.new_climber import NewClimber
from gui.test_page.data_communicator import CombinedDataCommunicator
import pandas as pd


class TestPage(QWidget):
    """
    The Test Page allows admins to manage climbers and conduct tests.

    Features:
    - Add, edit, delete climbers.
    - Select a climber, choose a test and arm, and save the test data.

    Args:
        db_manager (DatabaseManager): Database manager for data saving and retrieval.
        admin_id (int): ID of the logged-in admin.
        main_stacked_widget (QStackedWidget): Manages page transitions.
    """
    def __init__(self, db_manager, admin_id, main_stacked_widget):
        super().__init__()
        self.db_manager = db_manager
        self.admin_id = admin_id
        self.main_stacked_widget = main_stacked_widget
        # self.selected_test = None
        self.setup_ui()
        self.load_climbers()

    def setup_ui(self):
        """Sets up the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        title_label = QLabel("Manage Research Members & Testing")
        title_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(title_label)

        # Form layout for arm and climber selection
        form_layout = QGridLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setHorizontalSpacing(20)

        self.climber_selector = QComboBox()
        self.climber_selector.addItem("Select a climber", None)
        form_layout.addWidget(QLabel("Climber Tested:"), 1, 0)
        form_layout.addWidget(self.climber_selector, 1, 1)

        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["Select Arm", "Dominant", "Non-dominant"])
        form_layout.addWidget(QLabel("Arm Tested:"), 0, 0)
        form_layout.addWidget(self.arm_tested_combo, 0, 1)

        # New ComboBox for Test Data Type
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Select Data Type", "Force", "NIRS", "Force and NIRS"])
        form_layout.addWidget(QLabel("Test Data Type:"), 2, 0)
        form_layout.addWidget(self.data_type_combo, 2, 1)

        main_layout.addLayout(form_layout)

        # Climber management buttons
        climber_button_layout = QHBoxLayout()
        button_style = """
            QPushButton {
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                background-color: #007BFF;
                color: white;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003f7f;
            }
        """

        edit_button = QPushButton("Edit Climber Info")
        edit_button.setStyleSheet(button_style)
        edit_button.clicked.connect(self.edit_climber_info)
        climber_button_layout.addWidget(edit_button)

        delete_button = QPushButton("Delete Climber")
        delete_button.setStyleSheet(button_style)
        delete_button.clicked.connect(self.delete_climber)
        climber_button_layout.addWidget(delete_button)

        add_button = QPushButton("Add New Climber")
        add_button.setStyleSheet(button_style)
        add_button.clicked.connect(self.add_new_climber)
        climber_button_layout.addWidget(add_button)

        main_layout.addLayout(climber_button_layout)

        # Test selection buttons in two columns
        button_layout = QGridLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setHorizontalSpacing(20)
        button_layout.setVerticalSpacing(10)

        test_buttons = [
            "MVC",
            "AO",
            "IIT",
            "DH",
            "SIT",
            "IIRT"
        ]

        for i, test_name in enumerate(test_buttons):
            button = QPushButton(test_name)
            button.setStyleSheet(button_style)
            button.setMinimumSize(150, 75)
            button.clicked.connect(lambda _, name=test_name: self.select_test(name))
            row, col = divmod(i, 2)
            button_layout.addWidget(button, row, col)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_climbers(self):
        """Loads climbers registered by the current admin into the ComboBox."""
        climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
        self.climber_selector.clear()
        self.climber_selector.addItem("Select a climber", None)
        for climber in climbers:
            display_name = f"{climber['name']} {climber['surname']}"
            # Save both the id and email in a dict.
            self.climber_selector.addItem(display_name, climber["id"])

    def select_test(self, test_name):
        """Sets the selected test and validates climber and arm selections."""
        climber_id = self.climber_selector.currentData()
        selected_arm = self.arm_tested_combo.currentText()
        data_type = self.data_type_combo.currentText()

        if data_type == "Select Data Type":
            QMessageBox.warning(self, "No Data Type Selected", "Please select a test data type.")
            return

        if not climber_id:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber to proceed.")
            return

        if selected_arm == "Select Arm":
            QMessageBox.warning(self, "No Arm Selected", "Please select an arm to test.")
            return

        # self.selected_test = test_name
        test_id = test_name.lower()
        data_type = data_type.lower()
        if data_type == "force and nirs":
            data_type = "force_nirs"
        print(test_id, " _ ", data_type)
        self.launch_test_window(data_type, selected_arm, test_id)

    def launch_test_window(self, data_type, selected_arm, test_id):
        """Launches a new window for a new test."""
        climber_id = self.climber_selector.currentData()
        if not climber_id:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber.")
            return

        # # Compute a test label based on data type; using "ao" for All Out.
        # if data_type.lower() == "force":
        #     test_label = f"{test_id}_force"
        # elif data_type.lower() == "nirs":
        #     test_label = f"{test_id}_nirs"
        # elif data_type.lower() == "force and nirs":
        #     test_label = f"{test_id}_force_nirs"
        # else:
        #     QMessageBox.warning(self, "Some problem occurred during data type selection.")
        #     return
        #
        # print(test_label)

        # Create a dialog window for the test.
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Window")
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout(dialog)

        # Create the communicator instance with auto_start=False.
        communicator = CombinedDataCommunicator(
            # force_timestamps, force_values,
            # nirs_timestamps, nirs_values,
            admin_id=self.admin_id,
            climber_id=climber_id,
            arm_tested=selected_arm,
            window_size=60,
            auto_start=True,
            data_type=data_type,
            test_id=test_id
        )
        layout.addWidget(communicator)

        # Create Start and Stop buttons.
        button_layout = QHBoxLayout()
        start_button = QPushButton("Start")
        stop_button = QPushButton("Stop")
        button_layout.addWidget(start_button)
        button_layout.addWidget(stop_button)
        layout.addLayout(button_layout)

        start_button.clicked.connect(communicator.start_acquisition)

        def stop_and_close():
            communicator.stop_acquisition()
            dialog.accept()

        stop_button.clicked.connect(stop_and_close)

        # Override the closeEvent of the dialog to confirm before closing.
        def dialog_close_event(event):
            """Asks for confirmation before closing the All Out Test window."""
            if not communicator.finalized:  # If data isn't finalized, user might lose progress
                reply = QMessageBox.question(
                    dialog,
                    "Confirm Exit",
                    "Are you sure you want to close this All Out Test window?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    communicator.stop_acquisition()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()

        dialog.closeEvent = dialog_close_event

        dialog.exec()

    def edit_climber_info(self):
        """Opens an edit dialog for the selected climber."""
        climber_id = self.climber_selector.currentData()
        if climber_id:
            edit_dialog = QDialog(self)
            edit_dialog.setWindowTitle("Edit Climber Info")
            edit_dialog.setMinimumSize(600, 400)

            # from gui.research_members.edit_climber_info import EditClimberInfoPage
            edit_page = EditClimberInfoPage(self.admin_id, climber_id, self.reload_climbers, self.db_manager)
            edit_page.dialog = edit_dialog
            layout = QVBoxLayout()
            layout.addWidget(edit_page)
            edit_dialog.setLayout(layout)

            def dialog_close_event(event):
                """Asks for confirmation before closing the dialog."""
                if not edit_page.is_saving:
                    reply = QMessageBox.question(
                        edit_dialog,
                        "Confirm Exit",
                        "Are you sure you want to leave without saving?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.reload_climbers(editing=True)
                        event.accept()
                    else:
                        event.ignore()
                else:
                    event.accept()
            edit_dialog.closeEvent = dialog_close_event
            edit_dialog.exec()
        else:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber to edit.")

    def add_new_climber(self):
        """Opens a dialog to add a new climber."""
        add_dialog = QDialog(self)
        add_dialog.setWindowTitle("Add New Climber")
        add_dialog.setMinimumSize(600, 400)

        # from gui.research_members.new_climber import NewClimber
        add_page = NewClimber(self.admin_id, self.reload_climbers, self.db_manager)
        add_page.dialog = add_dialog
        layout = QVBoxLayout()
        layout.addWidget(add_page)
        add_dialog.setLayout(layout)

        def dialog_close_event(event):
            """Asks for confirmation before closing the dialog."""
            if not add_page.is_saving:
                reply = QMessageBox.question(
                    add_dialog,
                    "Confirm Exit",
                    "Are you sure you want to leave without saving?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.reload_climbers(editing=False)
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()
        add_dialog.closeEvent = dialog_close_event
        add_dialog.exec()

    def delete_climber(self):
        """Deletes the selected climber after confirmation."""
        email = self.climber_selector.currentData()
        name = self.climber_selector.currentText()
        if email:
            reply = QMessageBox.question(
                self, "Delete Climber", f"Are you sure you want to delete climber {name}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.db_manager.delete_climber(email["email"], self.admin_id):
                    QMessageBox.information(self, "Success", f"Climber {name} deleted successfully.")
                    self.load_climbers()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete climber.")

    def reload_climbers(self, editing):
        """Reloads the members list after adding a new member."""
        if not editing:
            self.load_climbers()
        self.main_stacked_widget.setCurrentWidget(self)
