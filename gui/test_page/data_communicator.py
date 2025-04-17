import os
import queue
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import QTimer, Signal, QThread

from gui.test_page.evaluations.nirs_evaluation import NIRSEvaluation
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics, find_test_interval
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.results_page.report_window import TestReportWindow


class FeatherBinaryLogger:
    """
    Logs data points into an in-memory buffer and writes them to a Feather file.
    Note: Feather does not support appending, so all data is stored in memory
    and written out at once when close() is called.
    """

    def __init__(self, folder="tests", prefix="processed_data", timestamp="today", db_queue=None):
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        filename = f"{folder_path}/{prefix}_{timestamp}.feather"
        print(f'{prefix} file was prepared:', filename)
        self.filename = filename
        self.db_queue = db_queue
        self.buffer = []  # in-memory buffer for data points

    def log(self, ts, val):
        """Append a data point to the buffer."""
        self.buffer.append([ts, val])
        # print(ts, "_", val)

    def flush(self):
        """Write the buffered data to the Feather file and clear the buffer."""
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=["time", "value"])
            # Write the entire DataFrame to a Feather file
            df.reset_index(drop=True).to_feather(self.filename)
            print(self.filename, 'was saved')
            self.buffer = []

    def close(self):
        """Flush any remaining data and write to the Feather file."""
        self.flush()


class DataPlotWorker(QThread):
    """
    DataPlotWorker is a QThread subclass that continuously polls a given visualization queue (viz_queue)
    for new sensor data. When new data is available, it emits the newData signal carrying the data string.

    This allows the user interface (UI) to update the plot in near-real-time without using fixed-interval QTimers.

    Attributes:
        newData (Signal): A Qt signal that emits a data string whenever new data is received from the viz_queue.
        viz_queue (queue.Queue): A Python queue containing downsampled sensor data for visualization.
        running (bool): Flag to indicate whether the thread should continue polling.
    """
    newData = Signal(str)

    def __init__(self, viz_queue, parent=None):
        """
        Initializes the DataPlotWorker.

        Args:
            viz_queue (queue.Queue): The queue to poll for new visualization data.
            parent (QObject, optional): The parent QObject. Defaults to None.
        """
        super().__init__(parent)
        self.viz_queue = viz_queue
        self.running = True

    def run(self):
        """Continuously poll the viz_queue. When data is found, emit the newData signal."""
        while self.running:
            if not self.viz_queue.empty():
                data = self.viz_queue.get()
                self.newData.emit(data)
            else:
                self.msleep(10)  # Sleep for 10 ms to prevent busy waiting

    def stop(self):
        """Signal the thread to stop and wait until it terminates."""
        self.running = False
        self.wait()


class CombinedDataCommunicator(QMainWindow):
    """
    CombinedDataCommunicator is responsible solely for visualizing sensor data
    using data provided via the downsampled viz_queue.

    It updates plots and logs data (using file loggers) that is being pushed by the
    DataGenerator. All sensor communicators (Serial and Bluetooth) are assumed to be started
    externally (via DataGenerator), which feeds data into the viz_queue (for plotting)
    and db_queue (for complete data logging).

    The update functions poll the viz_queue in real time and update the plots accordingly.
    """

    def __init__(self, admin_id, climber_id, arm_tested, window_size=60, auto_start=True,
                 viz_queue=None, db_queue=None, data_type="force", test_type="ao",
                 sensor_control=None, parent=None):
        super().__init__(parent)
        self.start_time = None
        self.report_window = None
        self.timestamp = None
        self.force_file = None
        self.nirs_file = None

        # # Create QTimers for updating the plot.
        # self.force_timer = QTimer()
        # self.force_timer.timeout.connect(self.update_force)
        # self.nirs_timer = QTimer()
        # self.nirs_timer.timeout.connect(self.update_nirs)

        # Use the queues passed from DataGenerator or create new ones if not provided.
        self.viz_queue = viz_queue if viz_queue is not None else queue.Queue()
        self.db_queue = db_queue if db_queue is not None else queue.Queue()

        # Optional reference to a sensor controller to stop sensor connections.
        self.sensor_control = sensor_control

        self.admin_id = admin_id
        self.climber_id = climber_id
        self.test_type = test_type
        if arm_tested == "Dominant":
            self.arm_tested = "D"
        elif arm_tested == "Non-dominant":
            self.arm_tested = "N"
        else:
            self.arm_tested = "-"
        self.data_type = data_type

        self.x_min = -window_size
        self.x_max = 0

        # Containers for sensor data that will be updated from the viz_queue
        self.force_data = {"timestamps": [], "values": []}
        self.nirs_data = {"timestamps": [], "values": []}
        self.force_y_range = {"min": 0, "max": 80}
        self.nirs_y_range = {"min": 0, "max": 100}

        self.finalized = False
        self._setup_ui()

        # Create a worker thread that polls viz_queue.
        self.plot_worker = DataPlotWorker(self.viz_queue)
        self.plot_worker.newData.connect(self.handle_new_data)
        self.plot_worker.start()

        if auto_start:
            self.start_acquisition()

    def _setup_ui(self):
        """Sets up the UI components and dual-axis plot layout."""
        main_widget = QWidget()
        main_widget.setWindowTitle("Combined Sensor Data Communicator")
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Main plot widget.
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setXRange(self.x_min, self.x_max)

        # Setup based on data_type.
        if self.data_type == "nirs":
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            # self.plot_widget.getAxis('left').setLabel("")
            self.plot_widget.showAxis('right')
            self.plot_widget.getAxis('right').setLabel('NIRS (%)')
            # self.plot_widget.setYRange(0, 100)
            self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
            self.plot_widget.addItem(self.nirs_curve)
            self.nirs_text = pg.TextItem("", anchor=(1, 0))
            self.plot_widget.addItem(self.nirs_text)
        elif self.data_type == "force":
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            self.plot_widget.getAxis('left').setLabel("Force (kg)")
            # self.plot_widget.hideAxis('right')
            # if self.force_y_range["max"] >= 80:
            #     self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])
            # else:
            #     self.plot_widget.setYRange(0, 80)
            self.force_curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
            self.force_text = pg.TextItem("", anchor=(0, 0))
            self.plot_widget.addItem(self.force_text)
        else:  # Combined force and NIRS.
            self.plot_widget.getAxis('bottom').setLabel("Time (s)")
            self.plot_widget.getAxis('left').setLabel("Force (kg)")
            self.plot_widget.showAxis('right')
            self.plot_widget.getAxis('right').setLabel("NIRS (%)")
            # if self.force_y_range["max"] >= 80:
            #     self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])
            # else:
            #     self.plot_widget.setYRange(0, 80)
            self.nirs_view = pg.ViewBox()
            self.plot_widget.scene().addItem(self.nirs_view)
            self.plot_widget.getAxis('right').linkToView(self.nirs_view)
            # self.nirs_view.setYRange(0, 105, padding=0)
            # self.nirs_view.setXRange(self.x_min, self.x_max, padding=0)
            # self.plot_widget.getViewBox().sigResized.connect(self._update_views)
            self.force_curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
            self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
            self.nirs_view.addItem(self.nirs_curve)
            self.force_text = pg.TextItem("", anchor=(0, 0))
            self.plot_widget.addItem(self.force_text)
            self.nirs_text = pg.TextItem("", anchor=(1, 0))
            self.nirs_view.addItem(self.nirs_text)

        # Add a legend.
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
        self.nirs_view.linkedViewChanged(self.plot_widget.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        """
        Initializes file loggers, resets internal data containers, and starts the timers
        for plotting. Data is assumed to be continuously pushed into self.viz_queue by the DataGenerator.
        """
        self.start_time = time.time()
        self.timestamp = str(self.start_time)
        # self.start_time = time.time()
        if self.data_type in ["force", "force_nirs"]:
            self.force_file = FeatherBinaryLogger(folder="tests",
                                                  prefix=f"{self.test_type}_force",
                                                  timestamp=self.timestamp,
                                                  db_queue=self.db_queue)
        else:
            self.force_file = None

        if self.data_type in ["nirs", "force_nirs"]:
            self.nirs_file = FeatherBinaryLogger(folder="tests",
                                                 prefix=f"{self.test_type}_nirs",
                                                 timestamp=self.timestamp,
                                                 db_queue=self.db_queue)

        else:
            self.nirs_file = None

        if self.data_type != "nirs":
            self.force_data = {"timestamps": [], "values": []}
            # self.force_timer.start(10)  # Timer interval for force plotting.
        if self.data_type != "force":
            self.nirs_data = {"timestamps": [], "values": []}
            # self.nirs_timer.start(10)  # Timer interval for NIRS plotting.

    def handle_new_data(self, data_str):
        """
        Slot that handles new data coming from the worker thread.
        The data string is expected in the format "sensorID_timestamp_value".
        Update internal data arrays and call updatePlot.
        """
        parts = data_str.split("_")
        print('Data to plot:', data_str)
        if len(parts) < 3:
            return
        try:
            sensor = int(parts[0])
            timestamp = float(parts[1])
            value = float(parts[2])
        except ValueError:
            return

        if sensor == 1:  # Force sensor data
            print('plotting Force:', timestamp, value)
            self.force_data["timestamps"].append(timestamp)
            self.force_data["values"].append(value)
        elif sensor == 2:  # NIRS (e.g., SMO2) data
            print('plotting NIRS:', timestamp, value)
            self.nirs_data["timestamps"].append(timestamp)
            self.nirs_data["values"].append(value)
        # (You can also check for battery data if sensor== 'b', etc.)
        # Optionally, also log every value to the file logger using db_queue is handled in DataGenerator.
        self.update_plot()

    def update_plot(self):
        """
        Update the plot curves using the data that has been collected.
        We adjust the x-axis to be relative to the most recent timestamp.
        """
        # For force data
        if self.data_type in ["force", "force_nirs"] and self.force_data["timestamps"]:
            current_ts = self.force_data["timestamps"][-1]
            # Optionally adjust y_range
            if np.max(self.force_data["values"]) >= 80:
                self.force_y_range["min"] = np.min(self.force_data["values"]) - 5
                self.force_y_range["max"] = np.max(self.force_data["values"]) + 5
            x_data = np.array(self.force_data["timestamps"]) - current_ts
            y_data = np.array(self.force_data["values"])
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.force_curve.setData(x_data, y_data)
            self.force_text.setText(f"Force: {y_data[-1]:.2f} kg")
            self.force_text.setPos(self.x_min, self.force_y_range["max"])
        # For NIRS data
        if self.data_type in ["nirs", "force_nirs"] and self.nirs_data["timestamps"]:
            current_ts = self.nirs_data["timestamps"][-1]
            x_data = np.array(self.nirs_data["timestamps"]) - current_ts
            y_data = np.array(self.nirs_data["values"])
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.nirs_curve.setData(x_data, y_data)
            self.nirs_text.setText(f"NIRS: {y_data[-1]:.2f} %")
            self.nirs_text.setPos(self.x_max, 100)
            if self.data_type == "force_nirs":
                # In combined mode, you might need to update secondary view geometry.
                self._update_views()

    # def update_force(self):
    #     """
    #     Polls the viz_queue for new force data (assumed to be in the format "1_timestamp_value"),
    #     updates the internal force_data container, logs data via the file logger, and updates the plot.
    #     """
    #     new_data = False
    #     while not self.viz_queue.empty():
    #         data_str = self.viz_queue.get()
    #         # print(data_str)
    #         # if data_str.startswith("2_"):
    #         parts = data_str.split("_")
    #         if len(parts) >= 3:
    #             try:
    #                 sensor = int(parts[0])
    #                 timestamp = float(parts[1])
    #                 value = float(parts[2])
    #             except ValueError:
    #                 continue
    #             if sensor == 1:
    #                 self.force_data["timestamps"].append(timestamp)
    #                 self.force_data["values"].append(value)
    #                 # if self.force_file is not None:
    #                 #     self.force_file.log(timestamp, value)
    #                 new_data = True
    #     if new_data and self.force_data["timestamps"]:
    #         current_ts = self.force_data["timestamps"][-1]
    #         self.force_y_range["min"] = np.min(self.force_data["values"]) - 5
    #         self.force_y_range["max"] = np.max(self.force_data["values"]) + 5
    #         x_data = np.array(self.force_data["timestamps"]) - current_ts
    #         y_data = np.array(self.force_data["values"])
    #         mask = (x_data >= self.x_min) & (x_data <= self.x_max)
    #         x_data = x_data[mask]
    #         y_data = y_data[mask]
    #         self.force_curve.setData(x_data, y_data)
    #         self.force_text.setText(f"Force Time: {(current_ts - self.start_time):.2f} s\nForce: {y_data[-1]:.2f} kg")
    #         if self.force_y_range["max"] >= 80:
    #             self.force_text.setPos(self.x_min, self.force_y_range["max"])
    #         else:
    #             self.force_text.setPos(self.x_min, 80)
    #         self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])
    #
    # def update_nirs(self):
    #     """
    #     Polls the viz_queue for new NIRS data (assumed to be in the format "2_timestamp_value"),
    #     updates the internal nirs_data container, logs data via the file logger, and updates the plot.
    #     """
    #     new_data = False
    #     while not self.viz_queue.empty():
    #         data_str = self.viz_queue.get()
    #         # print(data_str)
    #         # if data_str.startswith("2_"):
    #         parts = data_str.split("_")
    #         if len(parts) >= 3:
    #             try:
    #                 sensor = int(parts[0])
    #                 timestamp = float(parts[1])
    #                 value = float(parts[2])
    #             except ValueError:
    #                 continue
    #             if sensor == 2:
    #                 self.nirs_data["timestamps"].append(timestamp)
    #                 self.nirs_data["values"].append(value)
    #                 # if self.nirs_file is not None:
    #                 #     self.nirs_file.log(timestamp, value)
    #                 new_data = True
    #     if new_data and self.nirs_data["timestamps"]:
    #         current_ts = self.nirs_data["timestamps"][-1]
    #         x_data = np.array(self.nirs_data["timestamps"]) - current_ts
    #         y_data = np.array(self.nirs_data["values"])
    #         mask = (x_data >= self.x_min) & (x_data <= self.x_max)
    #         x_data = x_data[mask]
    #         y_data = y_data[mask]
    #         self.nirs_curve.setData(x_data, y_data)
    #         self.nirs_text.setText(f"NIRS Time: {(current_ts - self.start_time):.2f} s\nNIRS: {y_data[-1]:.2f} %")
    #         self.nirs_text.setPos(self.x_max, 100)

    # def log_value(self):
    #     # new_data = False
    #     while not self.db_queue.empty():
    #         data_str = self.db_queue.get()
    #         print("DB_QUEUE:", data_str)
    #         # print(data_str)
    #         parts = data_str.split("_")
    #         if len(parts) >= 3:
    #             try:
    #                 sensor = int(parts[0])
    #                 timestamp = float(parts[1])
    #                 value = float(parts[2])
    #             except ValueError:
    #                 print("ValueError in db_queue")
    #                 continue
    #             # if data_str.startswith("2_"):
    #             #     self.nirs_data["timestamps"].append(timestamp)
    #             #     self.nirs_data["values"].append(value)
    #             if sensor == 1 and self.force_file:
    #                 self.force_file.log(timestamp, value)
    #             elif sensor == 2 and self.nirs_file:
    #                 self.nirs_file.log(timestamp, value)
    #             # new_data = True

    def log_final_data(self):
        """
        Optionally, drain the db_queue and ensure all logged values are flushed to file.
        This can be called before final evaluation.
        """
        while not self.db_queue.empty():
            data_str = self.db_queue.get()
            parts = data_str.split("_")
            print('Data to log:', data_str)
            if len(parts) >= 3:
                try:
                    sensor = int(parts[0])
                    timestamp = float(parts[1])
                    value = float(parts[2])
                except ValueError:
                    continue
                if sensor == 1 and self.force_file is not None:
                    self.force_file.log(timestamp, value)
                elif sensor == 2 and self.nirs_file is not None:
                    self.nirs_file.log(timestamp, value)
        if self.force_file is not None:
            self.force_file.flush()
        if self.nirs_file is not None:
            self.nirs_file.flush()

    def stop_sensor_connections(self):
        """
        Stops sensor communications by calling the appropriate stop methods on the sensor controller.
        This requires that a reference to the DataGenerator (or similar) is passed via sensor_control.
        """
        if self.sensor_control is not None:
            # If the test uses force (or both), stop the serial sensor
            if self.data_type in ["force", "force_nirs"]:
                self.sensor_control.serial_communicator.stop_serial_collection()
            # If the test uses NIRS (or both), stop the Bluetooth sensor
            if self.data_type in ["nirs", "force_nirs"]:
                self.sensor_control.bluetooth_communicator.stop_bluetooth_collector()

    def stop_acquisition(self):
        """
        Stops the worker thread, logs any remaining data, stops sensor connections,
        and finalizes the test evaluation.
        """
        if self.plot_worker.isRunning():
            self.plot_worker.stop()
        self.log_final_data()
        self.stop_sensor_connections()
        self.finalize_acquisition()

    def finalize_acquisition(self):
        """
        Finalizes the test by computing evaluation metrics, saving results to the database,
        and opening a report window. This method is called once acquisition is stopped.
        """
        if not self.finalized:
            if self.data_type == "force":
                force_evaluator = ForceMetrics(file_path=self.force_file.filename,
                                               test_type=self.test_type,
                                               threshold_ratio=0.1)
                test_results, rep_results = force_evaluator.evaluate()
                number_of_reps = len(rep_results)
                nirs_results = None
                print('number of reps:', number_of_reps)
                print("Force Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            elif self.data_type == "nirs":
                nirs_results = {'Only NIRS': 'TODO'}
                test_results = rep_results = number_of_reps = None
            elif self.data_type == "force_nirs":
                force_evaluator = ForceMetrics(file_path=self.force_file.filename,
                                               test_type=self.test_type)
                test_results, rep_results = force_evaluator.evaluate()
                start_time, test_start_abs, test_end_abs = find_test_interval(self.force_file.filename)
                test_start_rel = test_start_abs - start_time
                test_end_rel = test_end_abs - start_time
                nirs_eval = NIRSEvaluation(self.nirs_file.filename, smoothing_window=25,
                                           baseline_threshold=0.1, recovery_tolerance=1.0)
                print(start_time, test_start_rel, test_end_rel)
                nirs_results = nirs_eval.evaluate(start_time, test_start_rel, test_end_rel)
                print("NIRS Evaluation Results:")
                print(nirs_results)
                number_of_reps = len(rep_results)
                print('number of reps:', number_of_reps)
                print("Force Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            else:
                QMessageBox.warning(self, "Error", "Unknown data type; cannot generate report")
                print(self.data_type)
                raise ValueError("Unknown data type; cannot generate report.")

            db_manager = ClimbingTestManager()
            force_file = self.force_file.filename if self.force_file else ''
            nirs_file = self.nirs_file.filename if self.nirs_file else ''
            db_data_save = {
                'arm_tested': self.arm_tested,
                'data_type': self.data_type,
                'test_type': self.test_type,
                'timestamp': self.timestamp,
                'number_of_reps': number_of_reps,
                'force_file': force_file,
                'nirs_file': nirs_file,
                'test_results': str(test_results),
                'nirs_results': str(nirs_results),
                'rep_results': str(rep_results)
            }
            print("data to db: ", db_data_save)
            test_id = db_manager.add_test_result(admin_id=str(self.admin_id),
                                                 participant_id=str(self.climber_id),
                                                 db_data=db_data_save)
            db_manager.close_connection()
            QMessageBox.information(self, "Test saving", "Test was saved successfully.")
            self.finalized = True

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
            db_data_save['id'] = test_id
            self.report_window = TestReportWindow(
                participant_info=climber_data,
                db_data=db_data_save,
                parent=self
            )
            self.report_window.show()

    def closeEvent(self, event):
        """
        Handles the window close event by stopping timers, finalizing acquisition,
        and closing any open log files.
        """
        # if not self.finalized:
        #     self.finalize_acquisition()
        # if self.force_file:
        #     self.force_file.close()
        # if self.nirs_file:
        #     self.nirs_file.close()
        self.stop_acquisition()
        event.accept()
