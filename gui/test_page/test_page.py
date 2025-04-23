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
from PySide6.QtCore import Qt, QTimer, QEventLoop

from gui.research_members.edit_climber_info import EditClimberInfoPage
from gui.research_members.new_climber import NewClimber
from gui.test_page.data_communicator import CombinedDataCommunicator
import pandas as pd

from gui.test_page.data_gen import DataGenerator

import threading
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
        super().__init__()
        self.db_manager = db_manager
        self.admin_id = admin_id
        # self.is_superuser = is_superuser
        self.main_stacked_widget = main_stacked_widget
        # self.selected_test = None
        self.setup_ui()
        self.load_climbers()
        self.data_generator = DataGenerator()

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

        # NIRS connection button with status indication
        self.nirs_connect_button = QPushButton("NIRS x")
        self.nirs_connect_button.setStyleSheet(nirs_button_style)
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
        Toggle NIRS connection on/off.
        If already connected (real or simulation), this will disconnect.
        If not connected, this will attempt to connect.
        """
        print("=== NIRS CONNECTION TOGGLE ===")

        # Check current connection state
        bt_com = self.data_generator.bluetooth_com
        is_currently_connected = bt_com.running

        # If already connected, disconnect and update UI
        if is_currently_connected:
            print("Stopping active NIRS connection")

            # Create a simple dialog to show disconnection progress
            disconnect_dialog = QDialog(self)
            disconnect_dialog.setWindowTitle("NIRS Disconnection")
            disconnect_dialog.setFixedSize(400, 150)

            layout = QVBoxLayout(disconnect_dialog)

            # Status label
            status_label = QLabel("Disconnecting NIRS...")
            status_label.setAlignment(Qt.AlignCenter)
            status_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(status_label)

            # Stop the BLE thread
            self.data_generator.stop(nirs=True, force=False)

            # Wait briefly to ensure thread is stopped
            QTimer.singleShot(500, lambda: self._finish_nirs_disconnect(disconnect_dialog, status_label))

            # Show dialog (blocks until closed)
            disconnect_dialog.exec()
            return False

        # Not connected, so proceed with connection
        print("Starting new NIRS connection")

        # Pause forwarding data to external callbacks during connection
        self.data_generator.pause_data_forwarding()
        print("Data forwarding paused")

        # Create a dialog to show connection progress
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("NIRS Connection")
        progress_dialog.setFixedSize(400, 250)

        layout = QVBoxLayout(progress_dialog)

        # Status label
        status_label = QLabel("Attempting to connect to NIRS device...")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)

        # Progress messages
        progress_text = QLabel("")
        progress_text.setAlignment(Qt.AlignLeft)
        progress_text.setWordWrap(True)  # Allow text wrapping for longer messages
        layout.addWidget(progress_text)

        # Close button (hidden until connection process finishes)
        close_button = QPushButton("Close")
        close_button.setVisible(False)
        close_button.clicked.connect(progress_dialog.accept)
        layout.addWidget(close_button)

        # Thread control and connection state tracking
        monitor_active = {"value": True}
        connection_state = {
            "is_connected": False,
            "is_simulation": False,
            "final_state_determined": False,
            "connection_sequence_complete": False  # Flag to track if _run_loop has finished its attempts
        }

        # Debug output helper
        def debug_print(msg):
            print(f"NIRS Connection Debug: {msg}")

        # Update NIRS connection button based on the FINAL determined state
        def update_nirs_button():
            if not connection_state["connection_sequence_complete"]:
                debug_print(f"Button update prevented - connection sequence not complete")
                return

            if not connection_state["final_state_determined"]:
                debug_print(f"Button update prevented - final state not yet determined")
                return

            if hasattr(self, 'nirs_connect_button'):
                if connection_state["is_connected"]:
                    debug_print("Setting button to CONNECTED state (green)")
                    self.nirs_connect_button.setText("NIRS Connected ✓")
                    self.nirs_connect_button.setStyleSheet("""
                        QPushButton {
                            font-size: 16px;
                            padding: 10px;
                            border-radius: 5px;
                            background-color: #28a745;
                            color: white;
                        }
                        QPushButton:hover {
                            background-color: #218838;
                        }
                    """)
                elif connection_state["is_simulation"]:
                    debug_print("Setting button to SIMULATION state (yellow)")
                    self.nirs_connect_button.setText("NIRS Simulation")
                    self.nirs_connect_button.setStyleSheet("""
                        QPushButton {
                            font-size: 16px;
                            padding: 10px;
                            border-radius: 5px;
                            background-color: #ffc107;
                            color: black;
                        }
                        QPushButton:hover {
                            background-color: #e0a800;
                        }
                    """)
                else:
                    debug_print("Setting button to DEFAULT state")
                    self.nirs_connect_button.setText("Connect NIRS")
                    self.nirs_connect_button.setStyleSheet(self.button_style)

        # Start the connection process by launching the BLE thread
        debug_print("Clearing any previous data")
        self.data_generator.clear_data()

        debug_print("Starting NIRS connection process")
        self.data_generator.start_nirs_connection()

        # Connection monitoring function that runs in a separate thread
        def monitor_connection():
            try:
                debug_print("Monitor connection thread started")
                bt_com = self.data_generator.bluetooth_com

                # Initial state
                last_connection_state = ""
                max_attempts = bt_com.max_attempts
                attempts_seen = 0
                finished_message_seen = False

                # First, wait to see the connection attempts from _run_loop
                debug_print(f"Waiting for {max_attempts} connection attempts or success message")

                # Monitor connection progress and wait for connection sequence to complete
                while monitor_active["value"] and not connection_state["connection_sequence_complete"]:
                    current_conn_state = getattr(bt_com, 'connection_state', '')

                    # Update UI with connection messages
                    if current_conn_state and current_conn_state != last_connection_state:
                        debug_print(f"Connection state update: {current_conn_state}")
                        progress_text.setText(current_conn_state)
                        last_connection_state = current_conn_state

                    # Check if we've seen a connection attempt message
                    if "Reconnecting" in current_conn_state:
                        attempt_number = current_conn_state.split("(")[-1].split("/")[0]
                        try:
                            attempts_seen = max(attempts_seen, int(attempt_number))
                            debug_print(f"Detected connection attempt {attempts_seen}/{max_attempts}")
                        except ValueError:
                            pass

                    # Check if we've seen the successful connection message
                    if "Connected to" in current_conn_state:
                        debug_print("Detected successful connection message")
                        finished_message_seen = True

                    # Check if we've seen the "Failed to connect" message
                    if "Failed to connect after" in current_conn_state and "Starting simulation mode" in current_conn_state:
                        debug_print("Detected failed connection and simulation mode message")
                        finished_message_seen = True

                    # Connection sequence is complete when we've either:
                    # 1. Seen a successful connection message, or
                    # 2. Seen the max number of attempts and the failure message
                    if finished_message_seen or (attempts_seen >= max_attempts):
                        debug_print(
                            f"Connection sequence complete: attempts_seen={attempts_seen}, finished_message_seen={finished_message_seen}")
                        connection_state["connection_sequence_complete"] = True

                        # Wait a little longer to ensure the internal state has stabilized
                        time.sleep(1.0)

                        # Now determine the actual connection state
                        if bt_com.running and not bt_com.debugging:
                            debug_print("REAL CONNECTION DETECTED")
                            connection_state["is_connected"] = True
                            connection_state["is_simulation"] = False
                            status_label.setText("NIRS Connected Successfully ✓")
                            status_label.setStyleSheet("color: green; font-weight: bold;")
                        elif bt_com.running and bt_com.debugging:
                            debug_print("SIMULATION MODE ACTIVE")
                            connection_state["is_connected"] = False
                            connection_state["is_simulation"] = True
                            status_label.setText("NIRS Simulation Mode")
                            status_label.setStyleSheet("color: orange; font-weight: bold;")
                        else:
                            debug_print("CONNECTION FAILED")
                            connection_state["is_connected"] = False
                            connection_state["is_simulation"] = False
                            status_label.setText("NIRS Connection Failed ✗")
                            status_label.setStyleSheet("color: red; font-weight: bold;")

                        connection_state["final_state_determined"] = True
                        close_button.setVisible(True)
                        update_nirs_button()
                        return

                    # Brief pause between checks
                    time.sleep(0.2)

                debug_print(
                    f"Monitor thread exiting. connection_sequence_complete={connection_state['connection_sequence_complete']}")

            except Exception as e:
                debug_print(f"Error in monitor thread: {e}")
                if monitor_active["value"]:
                    status_label.setText("Error Monitoring NIRS ✗")
                    status_label.setStyleSheet("color: red; font-weight: bold;")
                    progress_text.setText(f"Error: {str(e)}")
                    close_button.setVisible(True)

                    # Mark as final state to allow button update
                    connection_state["connection_sequence_complete"] = True
                    connection_state["final_state_determined"] = True
                    connection_state["is_connected"] = False
                    connection_state["is_simulation"] = False
                    update_nirs_button()

        # Override dialog close event to clean up
        def on_dialog_close(event):
            debug_print("Dialog closing - cleaning up")
            monitor_active["value"] = False

            # Final check of connection state before closing
            bt_com = self.data_generator.bluetooth_com

            # If we never determined a final state, do it now
            if not connection_state["final_state_determined"]:
                debug_print("Final state determination on dialog close")
                connection_state["is_connected"] = bt_com.running and not bt_com.debugging
                connection_state["is_simulation"] = bt_com.running and bt_com.debugging
                connection_state["final_state_determined"] = True
                update_nirs_button()

            debug_print(
                f"Final connection state: connected={connection_state['is_connected']}, simulation={connection_state['is_simulation']}")
            debug_print("Resuming data forwarding")
            self.data_generator.resume_data_forwarding()
            event.accept()

        # Connect the close event
        progress_dialog.closeEvent = on_dialog_close

        # Start the monitoring thread
        monitoring_thread = threading.Thread(target=monitor_connection, daemon=True)
        monitoring_thread.start()

        # Show dialog (blocks until closed)
        debug_print("Showing connection dialog")
        progress_dialog.exec()

        # Dialog closed - clean up
        monitor_active["value"] = False
        debug_print("Dialog closed - ensuring data forwarding is resumed")
        self.data_generator.resume_data_forwarding()

        # Return true if we have either a real connection or simulation mode active
        result = connection_state["is_connected"] or connection_state["is_simulation"]
        debug_print(f"nirs_connection() returning {result}")
        return result

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
        self.nirs_connect_button.setText("Connect NIRS")
        self.nirs_connect_button.setStyleSheet(nirs_button_style)

        # Close the dialog after a brief delay
        QTimer.singleShot(1000, dialog.accept)

        # Log the disconnection
        print(f"{time.strftime('%H:%M:%S')} - NIRS disconnected manually by user")

    # def load_climbers(self):
    #     """Loads climbers registered by the current admin into the ComboBox."""
    #     # if self.admin_id == 1:
    #     #     climbers = self.db_manager.get_all_climbers()
    #     # else:
    #     climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
    #     self.climber_selector.clear()
    #     self.climber_selector.addItem("Select a climber", None)
    #     for climber in climbers:
    #         display_name = f"{climber['name']} {climber['surname']}"
    #         # Save both the id and email in a dict.
    #         self.climber_selector.addItem(display_name, climber["id"])

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
        print(test_type, " _ ", data_type)
        self.launch_test_window(data_type, selected_arm, test_type)

    def launch_test_window(self, data_type, selected_arm, test_type):
        """Launches a new window for a new test."""
        climber_id = self.climber_selector.currentData()
        if not climber_id:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber.")
            return

        # Clear data before starting the test - this is necessary
        self.data_generator.clear_data()
        
        # Make sure data forwarding is paused until explicitly started
        self.data_generator.pause_data_forwarding()

        # Set calibration and create dialog window
        self.data_generator.update_calibration(39, 155)
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