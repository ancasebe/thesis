"""
This module defines the `TestPage` class, which provides a unified interface for managing climbers
and conducting tests.

Key functionalities:
- Allows admins to add, edit, and delete climbers.
- Enables selection of a test, climber, and arm for testing.
- Ensures validation before conducting a test and saves test data to the database.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox,  QStackedWidget,  QDialog, QGridLayout
)
from PySide6.QtCore import Qt, QTimer

from gui.research_members.edit_climber_info import EditClimberInfoPage
from gui.research_members.new_climber import NewClimber
from gui.test_page.data_communicator import CombinedDataCommunicator

from gui.test_page.data_gen import DataGenerator

import time

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
        super().__init__(main_stacked_widget)
        self.db_manager = db_manager
        self.admin_id = admin_id
        self.main_stacked_widget = main_stacked_widget
        self.data_generator = DataGenerator()
        self.button_styles = {
            'default': "QPushButton { background-color: #e74c3c; color: white; font-size: 16px; border: none; padding: 8px; border-radius: 4px; }",
            'connecting': "QPushButton { background-color: #f39c12; color: white; font-size: 16px; border: none; padding: 8px; border-radius: 4px; }",
            'success': "QPushButton { background-color: #2ecc71; color: white; font-size: 16px; border: none; padding: 8px; border-radius: 4px; }",
            'warning': "QPushButton { background-color: #f1c40f; color: white; font-size: 16px; border: none; padding: 8px; border-radius: 4px; }",
            'error': "QPushButton { background-color: #e74c3c; color: white; font-size: 16px; border: none; padding: 8px; border-radius: 4px; }"
        }
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
        self.data_type_combo.addItems(["Select Data Type", "Force", "Force and NIRS"])  # "NIRS",
        form_layout.addWidget(QLabel("Test Data Type:"), 2, 0)
        form_layout.addWidget(self.data_type_combo, 2, 1)

        main_layout.addLayout(form_layout)

        # Define button style once for reuse
        self.button_style = """
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

        # Climber management buttons
        climber_button_layout = QHBoxLayout()

        edit_button = QPushButton("Edit Climber Info")
        edit_button.setStyleSheet(self.button_style)
        edit_button.clicked.connect(self.edit_climber_info)
        climber_button_layout.addWidget(edit_button)

        delete_button = QPushButton("Delete Climber")
        delete_button.setStyleSheet(self.button_style)
        delete_button.clicked.connect(self.delete_climber)
        climber_button_layout.addWidget(delete_button)

        add_button = QPushButton("Add New Climber")
        add_button.setStyleSheet(self.button_style)
        add_button.clicked.connect(self.add_new_climber)
        climber_button_layout.addWidget(add_button)

        # NIRS connection button with status indication
        self.nirs_connect_button = QPushButton("NIRS disconnected")
        self.nirs_connect_button.setStyleSheet(self.button_styles['default'])
        self.nirs_connect_button.clicked.connect(self.nirs_connection)
        climber_button_layout.addWidget(self.nirs_connect_button)

        main_layout.addLayout(climber_button_layout)

        # Rest of the setup_ui method remains the same...
        # Test selection buttons in two columns
        button_layout = QGridLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setHorizontalSpacing(20)
        button_layout.setVerticalSpacing(10)

        test_buttons = [
            "MVC",
            "AO",
            "IIT",
            # "DH",
            "SIT",
            # "IIRT"
        ]

        for i, test_name in enumerate(test_buttons):
            button = QPushButton(test_name)
            button.setStyleSheet(self.button_style)
            button.setMinimumSize(150, 75)
            button.clicked.connect(lambda _, name=test_name: self.select_test(name))
            row, col = divmod(i, 2)
            button_layout.addWidget(button, row, col)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def nirs_connection(self):
        """
        Handles the NIRS connection process, including connection dialog and status updates.
        Provides user feedback on connection state and manages connection timeouts.
        """
        # Get current connection state
        # currently_connected = self.data_generator.is_nirs_connected()
        currently_connected = self.data_generator.bluetooth_com.running
        
        if currently_connected:
            # Disconnect if currently connected
            self.data_generator.stop(nirs=True, force=False)
            self.nirs_connect_button.setText("NIRS disconnected")
            self.nirs_connect_button.setStyleSheet(self.button_styles['default'])
            print("NIRS connection stopped")
        else:
            # Connect if not connected
            print("Starting new NIRS connection")
            # Pause data forwarding but continue collecting
            self.data_generator.pause_data_forwarding()
        
            # Clear previous data
            self.data_generator.clear_data()
        
            # Start the connection process
            success = self.data_generator.start_nirs_connection()
        
            if not success:
                self.nirs_connect_button.setText("NIRS disconnected")
                self.nirs_connect_button.setStyleSheet(self.button_styles['error'])
                return
        
            # Start monitoring connection state in a separate thread

            # Show connecting status in the UI
            self.nirs_connect_button.setText("Connecting...")
            self.nirs_connect_button.setStyleSheet(self.button_styles['connecting'])
        
            # Create and configure a timer to monitor connection state
            self.connection_timer = QTimer()
            self.connection_timer.timeout.connect(self._check_nirs_connection_state)
            self.connection_timer.start(500)  # Check every 500ms
        
    def _check_nirs_connection_state(self):
        """
        Checks the NIRS connection state and updates the button accordingly.
        Called by timer every 500ms during connection process.
        """
        # Get the latest connection state message
        state_msg = self.data_generator.bluetooth_com.connection_state

        # Check connection flags
        thread_running = self.data_generator.bluetooth_com.running
        real_connection = self.data_generator.bluetooth_com.connected
        simulated = self.data_generator.bluetooth_com.debugging

        # Print detailed connection state for debugging
        # print(
        #     f"NIRS Connection Check - Thread running: {thread_running}, Real connection: {real_connection}, Simulated: {simulated}, Message: {state_msg}")

        # Update button based on connection state
        if thread_running and real_connection and not simulated:
            # Real connection established and receiving data
            self.nirs_connect_button.setText("NIRS Connected")
            self.nirs_connect_button.setStyleSheet(self.button_styles['success'])
            self.connection_timer.stop()
        elif thread_running and simulated:
            # Simulation mode active
            self.nirs_connect_button.setText("NIRS Simulated")
            self.nirs_connect_button.setStyleSheet(self.button_styles['warning'])
            self.connection_timer.stop()
        # elif thread_running and "Connected to" in state_msg and not real_connection:
            # Connection message received but no data yet - keep waiting
            # Don't stop the timer, keep checking
        elif not thread_running:
            # Connection attempt failed or was stopped
            self.nirs_connect_button.setText("Connect NIRS")
            self.nirs_connect_button.setStyleSheet(self.button_styles['default'])
            self.connection_timer.stop()

    def _finish_nirs_disconnect(self, dialog, status_label):
        """
        Helper function to complete the NIRS disconnection process
        and update the UI accordingly.
        """
        # Update the display to show disconnection is complete
        status_label.setText("NIRS Disconnected ✓")
        status_label.setStyleSheet("color: red; font-weight: bold;")

        nirs_button_style = """
            QPushButton {
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                background-color: red;
                color: white;
            }
            QPushButton:hover {
                background-color: darkred;
            }
            QPushButton:pressed {
                background-color: darkred;
            }
        """

        # Reset the NIRS button to "Connect" state
        self.nirs_connect_button.setText("NIRS disconnected")
        self.nirs_connect_button.setStyleSheet(nirs_button_style)

        # Close the dialog after a brief delay
        QTimer.singleShot(1000, dialog.accept)

        # Log the disconnection
        print(f"{time.strftime('%H:%M:%S')} - NIRS disconnected manually by user")

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
        test_type = test_name.lower()
        data_type = data_type.lower()
        if data_type == "force and nirs":
            data_type = "force_nirs"
        self.launch_test_window(data_type, selected_arm, test_type)

    def launch_test_window(self, data_type, selected_arm, test_type):
        """Launches a new window for a new test."""
        if self.data_generator.is_nirs_connected():
            self.nirs_connect_button.setText("NIRS Connected")
            self.nirs_connect_button.setStyleSheet(self.button_styles['success'])

        climber_id = self.climber_selector.currentData()
        if not climber_id:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber.")
            return

        # Clear data before starting the test - this is necessary
        self.data_generator.clear_data()
        
        # Make sure data forwarding is paused until explicitly started
        self.data_generator.pause_data_forwarding()

        # Set calibration and create dialog window
        self.data_generator.update_calibration(43, 155)
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Window")
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout(dialog)

        # Create communicator with auto_start=False
        communicator = CombinedDataCommunicator(
            data_generator=self.data_generator,
            admin_id=self.admin_id,
            climber_id=climber_id,
            arm_tested=selected_arm,
            window_size=60,
            auto_start=False,  # Don't start automatically
            data_type=data_type,
            test_type=test_type,
            parent=None  # Don't set parent to avoid reference cycles
        )
        layout.addWidget(communicator)

        # Create Start and Stop buttons
        button_layout = QHBoxLayout()
        start_button = QPushButton("Start")
        stop_button = QPushButton("Stop")
        button_layout.addWidget(start_button)
        button_layout.addWidget(stop_button)
        layout.addLayout(button_layout)

        # Connect the start button correctly
        start_button.clicked.connect(communicator.start_acquisition)
        stop_button.clicked.connect(lambda: (communicator.stop_acquisition(), dialog.accept()))

        # Override dialog closeEvent
        def dialog_close_event(event):
            if not communicator.finalized:
                reply = QMessageBox.question(
                    dialog,
                    "Confirm Exit",
                    "Are you sure you want to close this Test window?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    communicator.stop_acquisition()
                    # Make sure to clear data when the window is closed
                    self.data_generator.clear_data()
                    event.accept()
            else:
                # Make sure to clear data even when finalized
                self.data_generator.clear_data()
                event.accept()

        dialog.closeEvent = dialog_close_event
        dialog.exec()

    def edit_climber_info(self):
        """Opens an edit dialog for the selected climber."""
        climber_id = self.climber_selector.currentData()
        name = self.climber_selector.currentText()
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
        climber_id = self.climber_selector.currentData()
        name = self.climber_selector.currentText()
        if climber_id:
            reply = QMessageBox.question(
                self, "Delete Climber", f"Are you sure you want to delete climber {name}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.db_manager.delete_climber(climber_id):
                    QMessageBox.information(self, "Success", f"Climber {name} deleted successfully.")
                    self.load_climbers()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete climber.")

    def reload_climbers(self, editing):
        """Reloads the members list after adding a new member."""
        if not editing:
            self.load_climbers()
        self.main_stacked_widget.setCurrentWidget(self)

    def load_climbers(self):
        """Loads all climbers for the current admin into the dropdown."""
        try:
            self.climber_selector.clear()
            climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
            if not climbers:
                self.climber_selector.addItem("No climbers found", -1)
                return
        
            # Add climbers to the dropdown
            self.climber_selector.addItem("Select a climber", -1)
            for climber in climbers:
                # The returned structure now has 'id', 'name', and 'surname'
                display_text = f"{climber['name']} {climber['surname']}"
                self.climber_selector.addItem(display_text, climber['id'])
        except Exception as e:
            print(f"Error loading climbers: {e}")

    def closeEvent(self, event):
        """
        Handles application closure by stopping all threads and connections.
        """
        # Stop the NIRS connection timer
        if hasattr(self, 'connection_timer') and self.connection_timer is not None:
            self.connection_timer.stop()
        
        # Stop any running data generators and their associated threads
        if hasattr(self, 'data_generator') and self.data_generator is not None:
            self.data_generator.stop(nirs=True, force=True)
        
        # Close database connections
        if hasattr(self, 'db_manager') and self.db_manager is not None:
            self.db_manager.close()
        
        event.accept()