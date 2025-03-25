# results_page.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QTableWidget,
    QTableWidgetItem, QPushButton, QDialog, QMessageBox, QFormLayout,
    QLabel, QHeaderView, QVBoxLayout, QFileDialog
)
from PySide6.QtCore import Qt

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
        for test in ["MVC", "AO", "IIT", "DH", "SIT", "IIRT"]:
            self.test_selector.addItem(test)
        self.test_selector.currentIndexChanged.connect(self.on_test_filter_changed)
        top_layout.addWidget(QLabel("Test:"))
        top_layout.addWidget(self.test_selector)

        main_layout.addLayout(top_layout)

        # Middle area: Table to display tests for the chosen climber
        self.tests_table = QTableWidget()
        self.tests_table.setColumnCount(6)
        self.tests_table.setHorizontalHeaderLabels(
            ["Test ID", "Test Name", "NIRS", "Arm Tested", "Date", "Time"]
        )
        self.tests_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
            ("Units", self.show_units),
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

    def load_climbers(self):
        """
        Loads climbers into the climber_selector combo box.
        Normal admins see only climbers registered with their admin_id;
        superuser (admin_id == 1) sees all climbers.
        """
        if self.admin_id == 1:
            climbers = self.climber_db_manager.get_all_climbers()
        else:
            climbers = self.climber_db_manager.get_climbers_by_admin(self.admin_id)
        self.climber_selector.clear()
        self.climber_selector.addItem("Select a climber", None)
        for climber in climbers:
            display_name = f"{climber['name']} {climber['surname']}"
            self.climber_selector.addItem(display_name, climber["id"])

    def on_climber_changed(self, index):
        self.selected_climber_id = self.climber_selector.itemData(index)
        self.load_tests()

    def on_test_filter_changed(self, index):
        self.selected_test_label = self.test_selector.currentText()
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
        if not self.selected_climber_id:
            return
        tests = self.test_db_manager.fetch_results_by_participant(self.selected_climber_id)
        # If filtering by test label (other than "All Tests"), filter test rows.
        if self.selected_test_label and self.selected_test_label != "All Tests":
            filter_text = self.selected_test_label.lower()
            # Assuming test name is stored at index 7
            tests = [t for t in tests if len(t) > 7 and filter_text in str(t[7]).lower()]

        for row_index, test in enumerate(tests):
            self.tests_table.insertRow(row_index)
            # # Extract columns
            # test_id = str(test[0])
            # test_name = str(test[7]) if len(test) > 7 else "N/A"
            # nirs = str(test[8]) if len(test) > 8 else "N/A"
            # arm_tested = str(test[3])
            #
            # # Format timestamp to Date (YYYY-MM-DD) and Time (HH:MM)
            # date_str, time_str = self.format_timestamp(str(test[4]))
            #
            # # Populate table
            # self.tests_table.setItem(row_index, 0, QTableWidgetItem(test_id))
            # self.tests_table.setItem(row_index, 1, QTableWidgetItem(test_name))
            # self.tests_table.setItem(row_index, 2, QTableWidgetItem(nirs))
            # self.tests_table.setItem(row_index, 3, QTableWidgetItem(arm_tested))
            # self.tests_table.setItem(row_index, 4, QTableWidgetItem(date_str))
            # self.tests_table.setItem(row_index, 5, QTableWidgetItem(time_str))

            test_id = str(test[0])  # ID
            arm_tested = str(test[3])  # arm_tested
            data_type = str(test[4])  # data_type
            test_name = str(test[5])  # test_type
            raw_timestamp = str(test[6])  # timestamp

            date_str, time_str = self.format_timestamp(raw_timestamp)

            # Populate the table
            self.tests_table.setItem(row_index, 0, QTableWidgetItem(test_id))
            self.tests_table.setItem(row_index, 1, QTableWidgetItem(test_name))  # "Test Name"
            self.tests_table.setItem(row_index, 2, QTableWidgetItem(data_type))  # "NIRS"
            self.tests_table.setItem(row_index, 3, QTableWidgetItem(arm_tested))
            self.tests_table.setItem(row_index, 4, QTableWidgetItem(date_str))
            self.tests_table.setItem(row_index, 5, QTableWidgetItem(time_str))

    # def format_timestamp(self, raw_timestamp):
    #     """
    #     Given a string like '20250319_132045',
    #     returns ('2025-03-19', '13:20') if possible.
    #     Otherwise returns (raw_timestamp, '') if we can't parse.
    #     """
    #     if "_" not in raw_timestamp:
    #         return raw_timestamp, ""
    #     date_part, time_part = raw_timestamp.split("_", 1)
    #     if len(date_part) == 8 and len(time_part) >= 4:
    #         # date_part: YYYYMMDD
    #         # time_part: HHMMSS or HHMM
    #         year = date_part[:4]
    #         month = date_part[4:6]
    #         day = date_part[6:8]
    #         date_str = f"{year}-{month}-{day}"
    #
    #         # If HHMMSS => keep only HH:MM
    #         hour = time_part[:2]
    #         minute = time_part[2:4]
    #         time_str = f"{hour}:{minute}"
    #         return date_str, time_str
    #     else:
    #         # fallback
    #         return date_part, time_part

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
        row = selected_items[0].row()
        test_id_item = self.tests_table.item(row, 0)
        if not test_id_item:
            return None
        test_id = int(test_id_item.text())
        # Look up the full test information by test id.
        tests = self.test_db_manager.fetch_results_by_participant(self.selected_climber_id)
        for t in tests:
            if t[0] == test_id:
                return t
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

        # Basic fields
        layout.addRow("Test ID:", QLabel(str(test[0])))
        # layout.addRow("Test Name:", QLabel(str(test[7]) if len(test) > 7 else "N/A"))
        # layout.addRow("NIRS:", QLabel(str(test[8]) if len(test) > 8 else "N/A"))
        # layout.addRow("Arm Tested:", QLabel(str(test[3])))
        #
        # # Format date/time for info window
        # date_str, time_str = self.format_timestamp(str(test[4]))
        # layout.addRow("Date:", QLabel(date_str))
        # layout.addRow("Time:", QLabel(time_str))

        layout.addRow("Test Name:", QLabel(str(test[5])))  # test_type
        layout.addRow("Data Type:", QLabel(str(test[4])))  # data_type
        layout.addRow("Arm Tested:", QLabel(str(test[3])))

        date_str, time_str = self.format_timestamp(str(test[6]))  # timestamp
        layout.addRow("Date:", QLabel(date_str))
        layout.addRow("Time:", QLabel(time_str))

        # Show the raw file paths or test results if desired
        # layout.addRow("File Paths:", QLabel(str(test[5])))
        # layout.addRow("Test Results:", QLabel(str(test[6])))

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
        participant_info = self.climber_db_manager.get_user_data(self.admin_id, self.selected_climber_id)
        if not participant_info:
            participant_info = {"Name": "Unknown"}
        test_metrics = {"Max Strength": "N/A", "Critical Force": "N/A", "W Prime": "N/A"}
        # fig, ax = plt.subplots()
        # ax.plot([0, 1, 2], [0, 1, 0])  # dummy data
        # ax.set_title("Test Graph")
        report_window = TestReportWindow(participant_info=participant_info,
                                         test_metrics=test_metrics,
                                         data_type="force",
                                         test_type="ao",
                                         force_file=None,
                                         nirs_file=None,
                                         parent=None)
        report_window.show()

    def show_units(self):
        """
        Placeholder slot for the Units feature.
        """
        QMessageBox.information(self, "Units", "Units feature not implemented yet.")

    def export_data(self):
        """
        Exports the selected test's raw data from HDF5 to the chosen format (CSV, XLSX, or HDF5).
        - We check the 'file_paths' field in the test record (index 5).
        - If there are two paths, they're separated by ';'.
        - For each path, we open a file dialog and save in the user-selected format.
        """
        test = self.get_selected_test()
        if not test:
            return

        file_paths_str = str(test[5])  # The 'file_paths' column
        if not file_paths_str.strip():
            QMessageBox.warning(self, "No Data Files", "No data file is associated with this test.")
            return

        # Parse potential multiple paths
        paths = [p.strip() for p in file_paths_str.split(";") if p.strip()]
        if not paths:
            QMessageBox.warning(self, "No Data Files", "No valid data files found in the database.")
            return

        # Dialog to choose export format
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
            for i, path in enumerate(paths):
                # Attempt to read HDF5 data with pandas
                try:
                    df = pd.read_feather(path)
                except Exception as e:
                    QMessageBox.warning(dialog, "Error Reading feather file", f"Could not read {path}:\n{str(e)}")
                    continue

                # Prompt the user for a save path
                suggested_filename = f"exported_file_{i}.{export_format}"
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
                except Exception as e:
                    QMessageBox.warning(dialog, "Export Error", f"Could not export to {save_path}:\n{str(e)}")
                    continue

            QMessageBox.information(dialog, "Export", "Export completed.")
            dialog.accept()

        export_button.clicked.connect(perform_export)
        dialog.exec()

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
            test_id = test[0]
            cursor = self.test_db_manager.connection.cursor()
            cursor.execute("DELETE FROM climbing_tests WHERE id = ?", (test_id,))
            self.test_db_manager.connection.commit()
            QMessageBox.information(self, "Deleted", "Test deleted successfully.")
            self.load_tests()
