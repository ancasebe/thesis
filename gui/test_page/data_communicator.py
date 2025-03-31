import os
import queue
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import QTimer

from gui.test_page.data_generator import BluetoothCommunicator, SerialCommunicator
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics
from gui.test_page.evaluations.rep_metrics import RepMetrics
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.results_page.report_window import TestReportWindow


class FeatherBinaryLogger:
    """
    Logs data points into an in-memory buffer and writes them to a Feather file.
    Note: Feather does not support appending, so all data is stored in memory
    and written out at once when close() is called.
    """
    def __init__(self, folder="tests", prefix="processed_data", timestamp="today"):
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        filename = f"{folder_path}/{prefix}_{timestamp}.feather"
        self.filename = filename
        self.buffer = []  # in-memory buffer for data points

    def log(self, ts, val):
        """Append a data point to the buffer."""
        self.buffer.append([ts, val])
        print(ts, "_", val)

    def flush(self):
        """Write the buffered data to the Feather file and clear the buffer."""
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=["timestamp", "value"])
            # Write the entire DataFrame to a Feather file
            df.reset_index(drop=True).to_feather(self.filename)
            self.buffer = []

    def close(self):
        """Flush any remaining data and write to the Feather file."""
        self.flush()


# --- Combined Data Communicator Class ---
class CombinedDataCommunicator(QMainWindow):
    """
    CombinedDataCommunicator collects and visualizes sensor data from real-time sources
    (force sensor via Serial and NIRS sensor via Bluetooth) instead of reading from Excel files.

    It supports:
      - Starting and stopping data acquisition (which launches the real-time communicators),
      - Live updating of a dual-axis plot via QTimers that poll a shared data queue,
      - Logging incoming data to Feather files,
      - Computing test evaluation metrics and saving results to the database.
    """
    def __init__(self, admin_id, climber_id, arm_tested, window_size=60, auto_start=False,
                 data_type="force", test_type="ao", parent=None):
        """
        Initializes the communicator.

        Parameters:
            admin_id (str): Administrator ID.
            climber_id (str): Climber's ID.
            arm_tested (str): "Dominant", "Non-dominant", or other.
            window_size (int): Time window (in seconds) for the x-axis.
            auto_start (bool): If True, acquisition starts immediately.
            data_type (str): "force", "nirs", or "force_nirs".
            test_type (str): Test type identifier.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.report_window = None
        self.timestamp = None
        self.force_file = None
        self.nirs_file = None
        self.force_timer = None
        self.nirs_timer = None

        self.admin_id = admin_id
        self.climber_id = climber_id
        self.test_type = test_type
        if arm_tested == "Dominant":
            self.arm_tested = "D"
        elif arm_tested == "Non-dominant":
            self.arm_tested = "N"
        else:
            self.arm_tested = "-"
        self.data_type = data_type  # new parameter indicating the test and data type

        self.x_min = -window_size
        self.x_max = 0

        # Initialize empty containers for real-time data (populated from the shared queue)
        self.force_data = {"timestamps": [], "values": []}
        self.nirs_data = {"timestamps": [], "values": []}
        # Default y-range (will be updated as data arrives)
        self.force_y_range = {"min": 0, "max": 100}
        self.nirs_y_range = {"min": 0, "max": 100}

        self.finalized = False
        self._setup_ui()

        # Create a shared queue for real-time data (both communicators will push into it)
        self.data_queue = queue.Queue()
        # Instantiate communicators if their sensor type is active.
        if self.data_type != "nirs":
            self.serial_comm = SerialCommunicator(self.data_queue, self.data_queue)
            self.force_timer = QTimer()
            self.force_timer.timeout.connect(self.update_force)
        if self.data_type != "force":
            self.bluetooth_comm = BluetoothCommunicator(self.data_queue, self.data_queue)
            self.nirs_timer = QTimer()
            self.nirs_timer.timeout.connect(self.update_nirs)

        if auto_start:
            self.start_acquisition()

    def _setup_ui(self):
        """Sets up the UI components and dual-axis plot layout."""
        main_widget = QWidget()
        main_widget.setWindowTitle("Combined Sensor Data Communicator")
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Main plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setXRange(self.x_min, self.x_max)

        # CASE 1: NIRS only.
        if self.data_type == "nirs":
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            # Clear left axis since force data is not used.
            self.plot_widget.getAxis('left').setLabel("")
            self.plot_widget.showAxis('right')
            self.plot_widget.getAxis('right').setLabel('NIRS (%)')

            # Add NIRS curve and text directly to the main view.
            self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
            self.plot_widget.addItem(self.nirs_curve)
            self.nirs_text = pg.TextItem("", anchor=(1, 0))
            self.plot_widget.addItem(self.nirs_text)

        # CASE 2: Force only.
        elif self.data_type == "force":
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            self.plot_widget.getAxis('left').setLabel("Force (kg)")
            # Hide the right axis since no NIRS data will be shown.
            self.plot_widget.hideAxis('right')
            self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])

            # Add force curve and text to the main view.
            self.force_curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
            self.force_text = pg.TextItem("", anchor=(0, 0))
            self.plot_widget.addItem(self.force_text)

        # CASE 3: Combined (Force and NIRS).
        else:
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            self.plot_widget.getAxis('left').setLabel("Force (kg)")
            self.plot_widget.showAxis('right')
            self.plot_widget.getAxis('right').setLabel("NIRS (%)")
            self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])

            # Create a secondary view for NIRS data.
            self.nirs_view = pg.ViewBox()
            self.plot_widget.scene().addItem(self.nirs_view)
            self.plot_widget.getAxis('right').linkToView(self.nirs_view)
            self.nirs_view.setYRange(self.nirs_y_range["min"], self.nirs_y_range["max"], padding=0)
            self.nirs_view.setXRange(self.x_min, self.x_max, padding=0)
            self.plot_widget.getViewBox().sigResized.connect(self._update_views)

            # Add Force and NIRS curves.
            self.force_curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
            self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
            self.nirs_view.addItem(self.nirs_curve)
            # Add text items.
            self.force_text = pg.TextItem("", anchor=(0, 0))
            self.plot_widget.addItem(self.force_text)
            self.nirs_text = pg.TextItem("", anchor=(1, 0))
            self.nirs_view.addItem(self.nirs_text)

        # Add a legend in the top center.
        legend = self.plot_widget.addLegend()
        legend.setParentItem(self.plot_widget.getViewBox())
        legend.anchor((0.5, 0), (0.5, 0))
        if self.data_type == "nirs":
            legend.addItem(self.nirs_curve, "NIRS (%)")
        elif self.data_type == "force":
            legend.addItem(self.force_curve, "Force [kg]")
        else:
            legend.addItem(self.force_curve, "Force [kg]")
            legend.addItem(self.nirs_curve, "NIRS (%)")

    def _update_views(self):
        """Synchronize the geometry of the secondary NIRS view with the main view."""
        self.nirs_view.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        # Link X movements
        self.nirs_view.linkedViewChanged(self.plot_widget.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        """
        Starts real-time data acquisition from the Serial and/or Bluetooth communicators.

        This function sets a timestamp, creates Feather loggers for saving data,
        starts the communicators, and begins QTimers that poll the shared data queue.
        """
        self.timestamp = str(time.time())
        # Initialize data loggers for saving the acquired data.
        if self.data_type in ["force", "force_nirs"]:
            self.force_file = FeatherBinaryLogger(folder="tests", prefix=f"{self.test_type}_force",
                                                  timestamp=self.timestamp)
        else:
            self.force_file = None

        if self.data_type in ["nirs", "force_nirs"]:
            self.nirs_file = FeatherBinaryLogger(folder="tests", prefix=f"{self.test_type}_nirs",
                                                 timestamp=self.timestamp)
        else:
            self.nirs_file = None

        # Start the communicators to collect real-time data.
        if self.data_type != "nirs":
            self.serial_comm.start_serial_collection()
            self.force_data = {"timestamps": [], "values": []}  # Reset force data container.
            self.force_timer.start(10)  # Poll every 10 ms.
        if self.data_type != "force":
            self.bluetooth_comm.start_bluetooth_collector()
            self.nirs_data = {"timestamps": [], "values": []}  # Reset NIRS data container.
            self.nirs_timer.start(10)

    def update_force(self):
        """
        Polls the shared data queue for new force data, updates the internal data container,
        logs the data to file, and updates the force plot.

        Expected data format (from SerialCommunicator): "sensorID_timestamp_value".
        """
        new_data = False
        # Process all items in the shared queue.
        while not self.data_queue.empty():
            data_str = self.data_queue.get()
            # Filter for force data (assumed sensor ID "1")
            if data_str.startswith(f"{1}_"):
                parts = data_str.split("_")
                if len(parts) >= 3:
                    try:
                        timestamp = float(parts[1])
                        value = float(parts[2])
                    except ValueError:
                        continue
                    self.force_data["timestamps"].append(timestamp)
                    self.force_data["values"].append(value)
                    if self.force_file is not None:
                        self.force_file.log(timestamp, value)
                    new_data = True
        # If new force data was received, update the plot.
        if new_data and self.force_data["timestamps"]:
            current_ts = self.force_data["timestamps"][-1]
            # Update dynamic y-range.
            self.force_y_range["min"] = np.min(self.force_data["values"]) - 5
            self.force_y_range["max"] = np.max(self.force_data["values"]) + 5
            # Compute x-values as time differences relative to the most recent timestamp.
            x_data = np.array(self.force_data["timestamps"]) - current_ts
            y_data = np.array(self.force_data["values"])
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.force_curve.setData(x_data, y_data)
            self.force_text.setText(f"Force Time: {current_ts:.2f} s\nForce: {y_data[-1]:.2f} kg")
            self.force_text.setPos(self.x_min, self.force_y_range["max"])
            self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])

    def update_nirs(self):
        """
        Polls the shared data queue for new NIRS data, updates the internal data container,
        logs the data to file, and updates the NIRS plot.

        Expected NIRS data formats are those generated by BluetoothCommunicator.
        """
        new_data = False
        while not self.data_queue.empty():
            data_str = self.data_queue.get()
            # Filter for NIRS data (sensor IDs assumed 2, 3, or 4)
            if data_str.startswith(f"{2}_") or data_str.startswith(f"{3}_") or data_str.startswith(f"{4}_"):
                parts = data_str.split("_")
                if len(parts) >= 3:
                    try:
                        timestamp = float(parts[1])
                        value = float(parts[2])
                    except ValueError:
                        continue
                    self.nirs_data["timestamps"].append(timestamp)
                    self.nirs_data["values"].append(value)
                    if self.nirs_file is not None:
                        self.nirs_file.log(timestamp, value)
                    new_data = True
        if new_data and self.nirs_data["timestamps"]:
            current_ts = self.nirs_data["timestamps"][-1]
            self.nirs_y_range["min"] = np.min(self.nirs_data["values"]) - 5
            self.nirs_y_range["max"] = np.max(self.nirs_data["values"]) + 5
            x_data = np.array(self.nirs_data["timestamps"]) - current_ts
            y_data = np.array(self.nirs_data["values"])
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.nirs_curve.setData(x_data, y_data)
            self.nirs_text.setText(f"NIRS Time: {current_ts:.2f} s\nNIRS: {y_data[-1]:.2f} %")
            self.nirs_text.setPos(self.x_max, self.nirs_y_range["max"])

    def stop_acquisition(self):
        """
        Stops data acquisition by stopping the QTimers, halting the communicators,
        closing the data log files, and finalizing the test (computing metrics and saving results).
        """
        if self.force_timer is not None and self.force_timer.isActive():
            self.force_timer.stop()
        if self.nirs_timer is not None and self.nirs_timer.isActive():
            self.nirs_timer.stop()
        if self.data_type != "nirs":
            self.serial_comm.stop_serial_collection()
        if self.data_type != "force":
            self.bluetooth_comm.stop_bluetooth_collector()
        if self.force_file is not None:
            self.force_file.close()
        if self.nirs_file is not None:
            self.nirs_file.close()
        self.finalize_acquisition()

    def finalize_acquisition(self):
        """
        Finalizes the test by computing evaluation metrics, saving results to the database,
        and opening a report window. This method is called once acquisition is stopped.
        """
        if not self.finalized:
            if self.data_type == "force":
                evaluator = ForceMetrics(file_path=self.force_file.filename,
                                         test_type=self.test_type,
                                         sampling_rate=100)
                force_df = pd.read_feather(self.force_file.filename)
                rep_evaluator = RepMetrics(force_df, sampling_rate=100)
                rep_results = rep_evaluator.compute_rep_metrics()
                test_results = evaluator.evaluate()
                print("Force Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            elif self.data_type == "nirs":
                test_results = {"evaluation": "placeholder"}
                rep_results = {"rep_evaluation": "none"}
            elif self.data_type == "force_nirs":
                evaluator = ForceMetrics(file_path=self.force_file.filename,
                                         test_type=self.test_type,
                                         sampling_rate=100)
                force_df = pd.read_feather(self.force_file.filename)
                rep_evaluator = RepMetrics(force_df, sampling_rate=100)
                rep_results = rep_evaluator.compute_rep_metrics()
                test_results = evaluator.evaluate()
                print("Mixed Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            else:
                QMessageBox.warning(self, "Error", "Unknown test type; cannot generate report.")
                raise ValueError("Unknown test type; cannot generate report.")

            db_manager = ClimbingTestManager()
            force_file = self.force_file.filename if self.force_file else ''
            nirs_file = self.nirs_file.filename if self.nirs_file else ''
            db_data = {
                'arm_tested': self.arm_tested,
                'data_type': self.data_type,
                'test_type': self.test_type,
                'timestamp': self.timestamp,
                'force_file': force_file,
                'nirs_file': nirs_file,
                'test_results': str(test_results),
                'rep_results': str(rep_results)
            }
            print(db_data)
            db_manager.add_test_result(admin_id=str(self.admin_id),
                                       participant_id=str(self.climber_id),
                                       db_data=db_data)
            db_manager.close_connection()
            QMessageBox.information(self, "Test saving", "Test was saved successfully.")
            self.finalized = True

            # Load climber data and show the report window.
            # from gui.research_members.climber_db_manager import ClimberDatabaseManager
            climber_db = ClimberDatabaseManager()
            climber_data = climber_db.get_user_data(self.admin_id, self.climber_id)
            climber_db.close()
            if not climber_data:
                climber_data = {
                    "name": "Unknown",
                    "surname": "Unknown",
                    "email": "N/A",
                    "gender": "N/A",
                    "dominant_arm": "N/A"
                }
            self.report_window = TestReportWindow(
                participant_info=climber_data,
                db_data=db_data,
                parent=self
            )
            self.report_window.show()

    def close_event(self, event):
        """
        Handles the window close event by stopping timers, finalizing acquisition,
        and closing log files.
        """
        if self.force_timer.isActive():
            self.force_timer.stop()
        if self.nirs_timer.isActive():
            self.nirs_timer.stop()
        if not self.finalized:
            self.finalize_acquisition()
        if self.force_file:
            self.force_file.close()
        if self.nirs_file:
            self.nirs_file.close()
        event.accept()
