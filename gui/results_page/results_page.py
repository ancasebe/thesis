"""
Results page module for the Climbing Testing Application.

This module defines the ResultsPage class which displays test results for climbers.
It provides filtering options by climber, test type, and data type, and allows viewing
detailed test information, exporting data, and managing test records.

Key functionalities:
- Display test results in a filterable table
- View basic test information
- Access detailed test reports
- Export test data in various formats (CSV, XLSX, HDF5)
- Delete test records

The ResultsPage integrates with the database to fetch and display all tests conducted
by an admin, and provides navigation to more detailed analysis tools.
"""

import pandas as pd
import logging
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QGridLayout, QComboBox, QTableWidget,
    QTableWidgetItem, QPushButton, QDialog, QMessageBox, QFormLayout,
    QLabel, QHeaderView, QVBoxLayout, QFileDialog
)

from gui.results_page.report_window import TestReportWindow  # Using the provided report_window script
from gui.test_page.test_db_manager import ClimbingTestManager


class ResultsPage(QWidget):
    """
    ResultsPage allows the user to:
      1. Choose a climber (only those registered by the current admin for normal users, all for superuser).
      2. Choose a test label (MVC, AO, IIT, DH, SIT, IIRT or All Tests).
      3. See a table of tests with the following columns:
         Test ID, Test Name, NIRS, Arm Tested, Date, Time.
      4. Use bottom buttons to view basic test info, view detailed results (using the report window),
         access units (feature to be implemented later), export raw test data (via a dialog), or delete a test.
    """

    def __init__(self, admin_id, climber_db_manager):
        """
        Args:
            admin_id (int): The id of the currently logged in admin.
            climber_db_manager (ClimberDatabaseManager): Manager for climber data.
            test_db_manager (ClimbingTestManager): Manager for test result data.
        """
        super().__init__()
        self.selected_data_type = None
        self.admin_id = admin_id
        self.climber_db_manager = climber_db_manager
        self.test_db_manager = ClimbingTestManager()
        self.selected_climber_id = None
        self.selected_test_label = None
        self.setup_ui()
        self.load_climbers()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Top layout for climber and test selection
        top_layout = QHBoxLayout()

        # Climber selection combo box
        self.climber_selector = QComboBox()
        self.climber_selector.addItem("Select a climber", None)
        self.climber_selector.currentIndexChanged.connect(self.on_climber_changed)
        top_layout.addWidget(QLabel("Climber:"))
        top_layout.addWidget(self.climber_selector)

        # Test label selection combo box
        self.test_selector = QComboBox()
        # First option to show all tests, then individual test labels
        self.test_selector.addItem("All Tests")
        for test in ["MVC", "AO", "IIT", "SIT"]:
            self.test_selector.addItem(test)
        self.test_selector.currentIndexChanged.connect(self.on_test_filter_changed)
        top_layout.addWidget(QLabel("Test:"))
        top_layout.addWidget(self.test_selector)

        # Data type selection combo box
        self.data_type_selector = QComboBox()
        self.data_type_selector.addItem("All Data")
        for data in ["Force", "Force and NIRS"]:
            self.data_type_selector.addItem(data)
        self.data_type_selector.currentIndexChanged.connect(self.on_data_type_changed)
        top_layout.addWidget(QLabel("Data type:"))
        top_layout.addWidget(self.data_type_selector)

        main_layout.addLayout(top_layout)

        # Middle area: Table to display tests for the chosen climber
        self.tests_table = QTableWidget()
        self.tests_table.setColumnCount(9)
        self.tests_table.setHorizontalHeaderLabels(
            ["AdminID", "Test ID", "Name", "Surname", "Test Name", "NIRS", "Arm Tested", "Date", "Time"]
        )
        self.tests_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tests_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tests_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tests_table.setSelectionMode(QTableWidget.SingleSelection)
        main_layout.addWidget(self.tests_table)

        # Bottom area: Create action buttons using grid layout (similar to test_page)
        bottom_layout = QGridLayout()
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setHorizontalSpacing(20)
        bottom_layout.setVerticalSpacing(10)
        button_style = """
            QPushButton {
                font-size: 16px;        /* smaller font */
                padding: 8px;       /* reduced padding */
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

        # List of buttons: (Label, corresponding slot)
        bottom_buttons = [
            ("Show Info", self.show_info),
            ("Show Results", self.show_results),
            # ("Units", self.show_units),
            ("Export", self.export_data),
            ("Delete Test", self.delete_test)
        ]

        for i, (label, slot) in enumerate(bottom_buttons):
            button = QPushButton(label)
            button.setStyleSheet(button_style)
            button.setMinimumSize(150, 75)
            button.clicked.connect(slot)
            row, col = divmod(i, 2)
            bottom_layout.addWidget(button, row, col)

        main_layout.addLayout(bottom_layout)

    # Then update these methods:

    def load_climbers(self):
        """Load climbers for the current admin into the dropdown."""
        try:
            climbers = self.climber_db_manager.get_climbers_by_admin(self.admin_id)
            
            self.climber_selector.clear()
            if not climbers:
                self.climber_selector.addItem("No climbers found", None)
                return
            
            # Add "All climbers" option at the top
            self.climber_selector.addItem("All climbers", None)
            
            # Add individual climbers
            for climber in climbers:
                # The returned structure now has 'id', 'name', and 'surname'
                display_text = f"{climber['name']} {climber['surname']}"
                self.climber_selector.addItem(display_text, climber['id'])
            
        except Exception as e:
            logging.error(f"Error loading climbers: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load climbers: {str(e)}")

    def on_climber_changed(self, index):
        self.selected_climber_id = self.climber_selector.itemData(index)
        self.load_tests()

    def on_test_filter_changed(self):
        self.selected_test_label = self.test_selector.currentText()
        self.load_tests()

    def on_data_type_changed(self):
        self.selected_data_type = self.data_type_selector.currentText()
        self.load_tests()

    def load_tests(self):
        """
        Retrieves tests for the selected climber from the test database.
        If a test label (other than "All Tests") is chosen, the list is filtered accordingly.
        Expected test tuple structure (after updating the DB):
            0: id
            1: admin_id
            2: participant_id
            3: arm_tested
            4: data_type
            5: test_type
            6: timestamp
            7: file_paths
            8: test_results
        """
        self.tests_table.setRowCount(0)

        try:
            if self.selected_climber_id:
                tests = self.test_db_manager.fetch_results_by_participant(participant_id=self.selected_climber_id)
            else:
                tests = self.test_db_manager.fetch_results_by_admin(admin_id=self.admin_id)
            
            if tests is None:
                QMessageBox.warning(self, "Database Error", "Failed to fetch test results from the database.")
                return
                
            # If filtering by test label (other than "All Tests"), filter test rows.
            if self.selected_test_label and self.selected_test_label != "All Tests":
                filter_text = self.selected_test_label.lower()
                tests = [t for t in tests if filter_text in str(t["test_type"]).lower()]

            if self.selected_data_type and self.selected_data_type != "All Data":
                filter_text = self.selected_data_type.lower()
                tests = [t for t in tests if filter_text in str(t["data_type"]).lower()]

            for row_index, test in enumerate(tests):
                self.tests_table.insertRow(row_index)

                # Try to get climber data, with error handling
                try:
                    climber_data = self.climber_db_manager.get_user_data(self.admin_id, int(test["participant_id"]))
                    name = climber_data.get("name", "") if climber_data else ""
                    surname = climber_data.get("surname", "") if climber_data else ""
                except Exception as e:
                    logging.error(f"Error fetching climber data: {e}")
                    name = "Unknown"
                    surname = "Unknown"

                # Extract values from the test dictionary
                admin_id = str(test["admin_id"])
                test_id = str(test["id"])
                test_name = str(test["test_type"])
                data_type = str(test["data_type"])
                arm_tested = str(test["arm_tested"])
                raw_timestamp = str(test["timestamp"])

                date_str, time_str = self.format_timestamp(raw_timestamp)

                # Populate table columns:
                # Column 0: Admin ID, 1: Test ID, 2: Name, 3: Surname,
                # 4: Test Name, 5: NIRS (data_type), 6: Arm Tested, 7: Date, 8: Time.
                self.tests_table.setItem(row_index, 0, QTableWidgetItem(admin_id))
                self.tests_table.setItem(row_index, 1, QTableWidgetItem(test_id))
                self.tests_table.setItem(row_index, 2, QTableWidgetItem(name))
                self.tests_table.setItem(row_index, 3, QTableWidgetItem(surname))
                self.tests_table.setItem(row_index, 4, QTableWidgetItem(test_name))
                self.tests_table.setItem(row_index, 5, QTableWidgetItem(data_type))
                self.tests_table.setItem(row_index, 6, QTableWidgetItem(arm_tested))
                self.tests_table.setItem(row_index, 7, QTableWidgetItem(date_str))
                self.tests_table.setItem(row_index, 8, QTableWidgetItem(time_str))
                
        except Exception as e:
            logging.error(f"Error loading tests: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load test data: {str(e)}")

    @staticmethod
    def format_timestamp(raw_timestamp):
        """
        Given a Unix timestamp (as a float or string), returns a tuple:
          (date_str, time_str)
        where date_str is "YYYY-MM-DD" and time_str is "HH:MM".
        If the timestamp cannot be parsed, returns (raw_timestamp, "").
        """
        try:
            ts = float(raw_timestamp)
            dt = datetime.fromtimestamp(ts)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")
            return date_str, time_str
        except (ValueError, TypeError):
            return raw_timestamp, ""

    def get_selected_test(self):
        """
        Returns the test row (as a tuple) corresponding to the currently selected row in the tests table.
        If no row is selected, shows a warning.
        """
        selected_items = self.tests_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Test Selected", "Please select a test from the table.")
            return None
            
        try:
            row = selected_items[0].row()
            test_id_item = self.tests_table.item(row, 1)
            if not test_id_item:
                return None
            test_id = int(test_id_item.text())

            tests = self.test_db_manager.fetch_results_by_admin(admin_id=self.admin_id)
            if tests is None:
                QMessageBox.warning(self, "Database Error", "Failed to fetch test data from the database.")
                return None
            
            for t in tests:
                if t["id"] == test_id:
                    return t
                
            QMessageBox.warning(self, "Test Not Found", f"Test with ID {test_id} could not be found.")
            return None
        except Exception as e:
            logging.error(f"Error getting selected test: {e}")
            QMessageBox.critical(self, "Error", f"Failed to retrieve test data: {str(e)}")
            return None

    def show_info(self):
        """
        Displays a dialog with basic test information.
        Includes date/time in the new format (YYYY-MM-DD, HH:MM).
        """
        test = self.get_selected_test()
        if not test:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Info")
        layout = QFormLayout(dialog)

        # Retrieve participant info from test[2]
        participant_id = test["participant_id"]
        climber_data = self.climber_db_manager.get_user_data(self.admin_id, participant_id)
        name = climber_data.get("name", "") if climber_data else ""
        surname = climber_data.get("surname", "") if climber_data else ""

        # Basic fields
        layout.addRow("Test ID:", QLabel(str(test["id"])))
        layout.addRow("Name:", QLabel(name))
        layout.addRow("Surname:", QLabel(surname))
        layout.addRow("Test Name:", QLabel(str(test["test_type"])))
        layout.addRow("Data Type:", QLabel(str(test["data_type"])))
        layout.addRow("Arm Tested:", QLabel(str(test["arm_tested"])))
        layout.addRow("Number of Repetitions:", QLabel(str(test["number_of_reps"])))

        date_str, time_str = self.format_timestamp(str(test["timestamp"]))
        layout.addRow("Date:", QLabel(date_str))
        layout.addRow("Time:", QLabel(time_str))

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def show_results(self):
        """
        Opens the TestReportWindow showing a detailed report.
        Participant information is retrieved via the climber_db_manager.
        Here, test metrics are simulated and a dummy matplotlib figure is generated.
        """
        test = self.get_selected_test()
        if not test:
            return
    
        try:
            participant_id = test["participant_id"]
            participant_info = self.climber_db_manager.get_user_data(self.admin_id, participant_id)
            if not participant_info:
                participant_info = {"name": "Unknown", "surname": "Unknown"}
            
            db_data = self.test_db_manager.get_test_data(test_id=test['id'])
            if db_data is None:
                QMessageBox.warning(self, "Missing Data", 
                                    "Could not retrieve complete test data. The test may be corrupted.")
                return
            
            report_window = TestReportWindow(participant_info=participant_info,
                                            db_data=db_data,
                                            admin_id=self.admin_id,
                                            climber_db_manager=self.climber_db_manager,
                                            test_db_manager=self.test_db_manager,
                                            parent=self)
            report_window.show()
        except Exception as e:
            logging.error(f"Error showing results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to display test results: {str(e)}")

    # def show_units(self):
    #     """
    #     Placeholder slot for the Units feature.
    #     """
    #     QMessageBox.information(self, "Units", "Units feature not implemented yet.")

    def export_data(self):
        """
        Exports the selected test's raw data from separate files. If a force_file is present,
        it will be exported; similarly for a nirs_file. The exported filenames include the test id and the file type.
        """
        test = self.get_selected_test()
        if not test:
            return

        try:
            # Get the test id for filename pattern.
            test_id = test["id"]

            # Build a list of (file_type, path) tuples based on what is available.
            files_to_export = []
            if test.get("force_file") and test["force_file"].strip():
                files_to_export.append((test['test_type'], "force", test["force_file"]))
            if test.get("nirs_file") and test["nirs_file"].strip():
                files_to_export.append((test['test_type'], "nirs", test["nirs_file"]))

            if not files_to_export:
                QMessageBox.warning(self, "No Data Files", "No data file is associated with this test.")
                return

            # Create a dialog for the user to choose export format.
            dialog = QDialog(self)
            dialog.setWindowTitle("Export Test Data")
            vlayout = QVBoxLayout(dialog)
            vlayout.addWidget(QLabel("Select export format:"))
            format_combo = QComboBox()
            format_combo.addItems(["CSV", "XLSX", "HDF5"])
            vlayout.addWidget(format_combo)
            export_button = QPushButton("Export")
            vlayout.addWidget(export_button)

            def perform_export():
                export_format = format_combo.currentText().lower()  # "csv", "xlsx", or "hdf5"
                export_success = False
                
                for test_type, data_type, path in files_to_export:
                    # Attempt to read the file (assumes it's stored in Feather format)
                    try:
                        df = pd.read_feather(path)
                    except FileNotFoundError:
                        QMessageBox.warning(dialog, "File Not Found", 
                                           f"Could not find file: {path}\nThe file may have been moved or deleted.")
                        continue
                    except Exception as e:
                        QMessageBox.warning(dialog, "Error Reading File", 
                                           f"Could not read {path}:\n{str(e)}")
                        continue

                    # Create a suggested filename using the test id and file type
                    suggested_filename = f"test_{test_type}_{data_type}_{test_id}.{export_format}"
                    save_path, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save Converted File",
                        suggested_filename,
                        f"{export_format.upper()} Files (*.{export_format});;All Files (*)"
                    )
                    if not save_path:
                        # User canceled for this file
                        continue

                    # Export according to the chosen format
                    try:
                        if export_format == "csv":
                            df.to_csv(save_path, index=False)
                        elif export_format == "xlsx":
                            df.to_excel(save_path, index=False)
                        elif export_format == "hdf5":
                            df.to_hdf(save_path, key="data", mode="w")
                        export_success = True
                    except PermissionError:
                        QMessageBox.warning(dialog, "Permission Error", 
                                           f"Could not write to {save_path}. The file may be open in another program or you don't have permission.")
                        continue
                    except Exception as e:
                        QMessageBox.warning(dialog, "Export Error", 
                                           f"Could not export to {save_path}:\n{str(e)}")
                        continue
                        
                if export_success:
                    QMessageBox.information(dialog, "Export", "Export completed successfully.")
                else:
                    QMessageBox.warning(dialog, "Export", "No files were exported.")
                dialog.accept()

            export_button.clicked.connect(perform_export)
            dialog.exec()
            
        except Exception as e:
            logging.error(f"Error during export: {e}")
            QMessageBox.critical(self, "Export Error", f"An error occurred during export: {str(e)}")

    def delete_test(self):
        """
        Deletes the selected test from the test database after asking for confirmation.
        """
        test = self.get_selected_test()
        if not test:
            return
        
        reply = QMessageBox.question(
            self,
            "Delete Test",
            "Are you sure you want to delete this test?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                test_id = test['id']
                cursor = self.test_db_manager.connection.cursor()
                cursor.execute("DELETE FROM climbing_tests WHERE id = ?", (test_id,))
                self.test_db_manager.connection.commit()
                QMessageBox.information(self, "Deleted", "Test deleted successfully.")
                self.load_tests()
            except Exception as e:
                logging.error(f"Error deleting test: {e}")
                QMessageBox.critical(self, "Delete Error", f"Failed to delete test: {str(e)}")