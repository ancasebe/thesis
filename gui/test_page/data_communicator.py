import os
import csv
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt
# from datetime import datetime
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.results_page.report_window import TestReportWindow

'''
# --- CSV Logger Helper Class ---
class CSVLogger:
    """
    Handles CSV logging with a timestamped filename.

    Parameters:
        folder (str): Directory in which to save the CSV file.
        prefix (str): Prefix for the filename.
    """
    def __init__(self, folder="tests", prefix="processed_data", timestamp="today"):
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        filename = f"{folder_path}/{prefix}_{timestamp}.csv"
        print(filename)
        self.filename = filename
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["force_timestamp", "value"])

    def log(self, ts, val):
        """
        Logs a single data point to the CSV file.

        Parameters:
            ts (float): The timestamp.
            val (float): The measured sensor value.
        """
        self.writer.writerow([f"force_{ts}", val])

    def close(self):
        """Closes the CSV file."""
        self.file.close()


class BufferedBinaryLogger:
    """
    Logs data points into an in-memory buffer and writes them in batches to an HDF5 file.
    """
    def __init__(self, folder="tests", prefix="processed_data", timestamp="today", flush_interval=1000):
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        filename = f"{folder_path}/{prefix}_{timestamp}.h5"
        self.filename = filename
        self.buffer = []  # in-memory buffer for data points.
        self.flush_interval = flush_interval
        self.key = "data"
        # Open an HDFStore in write mode with table format (to allow appending).
        self.store = pd.HDFStore(self.filename, mode='w', complib='blosc', complevel=9)
        # Initialize the table by writing an empty DataFrame with the proper columns.
        self.store.put(self.key, pd.DataFrame(columns=["timestamp", "value"]), format='table', append=False)

    def log(self, ts, val):
        """Append a data point to the buffer and flush if necessary."""
        self.buffer.append([f"timestamp_{ts}", val])
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Write the buffered data to the HDF5 file and clear the buffer."""
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=["timestamp", "value"])
            self.store.append(self.key, df, format='table', data_columns=True)
            self.buffer = []

    def close(self):
        """Flush any remaining data and close the HDF5 store."""
        self.flush()
        self.store.close()
'''

# Replace the HDF5-based BufferedBinaryLogger with this Feather version:


class FeatherBinaryLogger:
    """
    Logs data points into an in-memory buffer and writes them to a Feather file.
    Note: Feather does not support appending, so all data is stored in memory
    and written out at once when close() is called.
    """
    def __init__(self, folder="tests", prefix="processed_data", timestamp="today"):
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        # Use a .feather extension
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
    Modified CombinedDataCommunicator that does not auto-start the acquisition.
    It now accepts admin_id and climber_id parameters and exposes public
    methods to start and stop data acquisition.

    Parameters:
            force_timestamps (np.array): Array of Force sensor timestamps (seconds).
            force_values (np.array): Array of Force sensor values.
            nirs_timestamps (np.array): Array of NIRS sensor timestamps (seconds).
            nirs_values (np.array): Array of NIRS sensor values.
            window_size (int): Time window (in seconds) for the x-axis (default 60).
            fixed_offset_ratio (float): Ratio to determine where new points appear.
    """
    def __init__(self, admin_id, climber_id, arm_tested, window_size=60, auto_start=False, data_type="force", test_type="ao"):

        super().__init__()
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

        # Load sensor data from Excel files.
        script_dir = os.path.dirname(__file__)  # folder where test_page.py is located
        if self.data_type != "nirs":
            force_file = os.path.join(script_dir, "group3_ao_copy.xlsx")
            force_df = pd.read_excel(force_file).dropna()
            force_timestamps = force_df.iloc[:, 0].values
            force_values = force_df.iloc[:, 3].values

            self.force_y_range = {"min": np.min(force_values) - 5, "max": np.max(force_values) + 5}

            self.force_data = {
                "timestamps": force_timestamps,
                "values": force_values,
                "index": 0,
                "num_points": len(force_timestamps)
            }

        if self.data_type != "force":
            nirs_file = os.path.join(script_dir, "group3_ao_copy.xlsx")
            nirs_df = pd.read_excel(nirs_file).dropna()
            nirs_timestamps = nirs_df.iloc[:, 0].values
            nirs_values = nirs_df.iloc[:, 2].values

            self.nirs_y_range = {"min": np.min(nirs_values) - 5, "max": np.max(nirs_values) + 5}

            self.nirs_data = {
                "timestamps": nirs_timestamps,
                "values": nirs_values,
                "index": 0,
                "num_points": len(nirs_timestamps)
            }

        self.finalized = False
        self._setup_ui()

        if self.data_type != "nirs":
            self.force_timer = QTimer()
            self.force_timer.timeout.connect(self.update_force)
        if self.data_type != "force":
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
        """Starts the data acquisition by starting the timers."""
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # if self.data_type in ["force", "force_nirs"]:
        #     self.force_file = BufferedBinaryLogger(folder="tests", prefix=f"{self.test_type}_force",
        #                                            timestamp=self.timestamp, flush_interval=1000)
        # else:
        #     self.force_file = None
        #
        # if self.data_type in ["nirs", "force_nirs"]:
        #     self.nirs_file = BufferedBinaryLogger(folder="tests", prefix=f"{self.test_type}_nirs",
        #                                           timestamp=self.timestamp, flush_interval=1000)
        # else:
        #     self.nirs_file = None

        self.timestamp = str(time.time())

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

        if self.force_file is not None:
            self.force_timer.start(10)
        if self.nirs_file is not None:
            self.nirs_timer.start(10)

    def update_force(self):
        """
        Updates the Force data plot.

        Logs the current Force data point, transforms its x value so that the newest point appears at the fixed offset,
        and updates the force curve and live text readout.
        """
        if self.force_file is None:
            return
        if self.force_data["index"] >= (self.force_data["num_points"] - 1):
            self.force_timer.stop()
            self.finalize_acquisition()
        else:
            current_ts = self.force_data["timestamps"][self.force_data["index"]]
            current_val = self.force_data["values"][self.force_data["index"]]
            self.force_file.log(current_ts, current_val)
            x_data = (self.force_data["timestamps"][:self.force_data["index"] + 1] - current_ts)
            y_data = self.force_data["values"][:self.force_data["index"] + 1]
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.force_curve.setData(x_data, y_data)
            self.force_text.setText(f"Force Time: {current_ts:.2f} s\nForce: {current_val:.2f} kg")
            # self.force_text.setAnchor(0, 0)
            self.force_text.setPos(self.x_min, self.force_y_range["max"])
            self.force_data["index"] += 1

    def update_nirs(self):
        """
        Updates the NIRS data plot.

        Logs the current NIRS data point, transforms its x value so that the newest point appears at the fixed offset,
        and updates the NIRS curve and live text readout.
        """
        if self.nirs_file is None:
            return
        if self.nirs_data["index"] >= (self.nirs_data["num_points"] - 1):
            self.nirs_timer.stop()
            self.finalize_acquisition()
        else:
            current_ts = self.nirs_data["timestamps"][self.nirs_data["index"]]
            current_val = self.nirs_data["values"][self.nirs_data["index"]]
            self.nirs_file.log(current_ts, current_val)
            x_data = (self.nirs_data["timestamps"][:self.nirs_data["index"] + 1] - current_ts)
            y_data = self.nirs_data["values"][:self.nirs_data["index"] + 1]
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            self.nirs_curve.setData(x_data, y_data)
            self.nirs_text.setText(f"NIRS Time: {current_ts:.2f} s\nNIRS: {current_val:.2f} %")
            # self.nirs_text.setAnchor(1, 0)
            self.nirs_text.setPos(self.x_max, self.nirs_y_range["max"])
            self.nirs_data["index"] += 1

    def stop_acquisition(self):
        """Stops the data acquisition and finalizes the test."""
        if self.force_timer is not None and self.force_timer.isActive():
            self.force_timer.stop()
        if self.nirs_timer is not None and self.nirs_timer.isActive():
            self.nirs_timer.stop()
        if self.force_file is not None:
            self.force_file.close()
        if self.nirs_file is not None:
            self.nirs_file.close()
        self.finalize_acquisition()

    def finalize_acquisition(self):
        """Generate the final graph and saves data to a database, if not done yet."""
        # global test_results, evaluator
        if not self.finalized:

            if self.data_type == "force":
                # final_graph = generate_final_graph_force(self.force_file.filename)
                evaluator = ForceMetrics(self.force_file.filename)
                test_results = evaluator.evaluate()
                print("Force Evaluation Results:")
                print(test_results)
            elif self.data_type == "nirs":
                # final_graph = generate_final_graph_nirs(self.nirs_file.filename)
                test_results = {"evaluation": "placeholder"}
            elif self.data_type == "force_nirs":
                # final_graph = generate_final_combined_graph(self.force_file.filename, self.nirs_file.filename)
                evaluator = ForceMetrics(self.force_file.filename)
                test_results = evaluator.evaluate()
                print("Mixed Evaluation Results:")
                print(test_results)
            else:
                QMessageBox.warning(self, "Error", "Unknown test type; cannot generate report.")
                raise ValueError("Unknown test type; cannot generate report.")

            # Save filenames in the database using the provided admin_id and climber_id.
            db_manager = ClimbingTestManager()
            # Here you could include the test type in the stored file_paths if desired.
            file_paths = ""
            if self.force_file is not None:
                file_paths += self.force_file.filename
            if self.nirs_file is not None:
                file_paths += ("; " if file_paths else "") + self.nirs_file.filename

            db_data = {'arm_tested': self.arm_tested,
                       'data_type': self.data_type,
                       'test_type': self.test_type,
                       'timestamp': self.timestamp,
                       'file_paths': file_paths,
                       'test_results': str(test_results)}

            # db_manager.add_test_result(str(self.admin_id), str(self.climber_id), self.arm_tested,
            #                            self.timestamp, file_paths, str(test_results))
            db_manager.add_test_result(str(self.admin_id), str(self.climber_id), db_data)
            db_manager.close_connection()
            QMessageBox.information(self, "Test saving", "Test was saved successfully.")
            self.finalized = True

            # --- Load Real Climber Data Using ClimberDatabaseManager ---
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

            # --- Create and Show the Report Window ---
            # self.report_window = TestReportWindow(climber_data, test_results, final_graph)
            self.report_window = TestReportWindow(
                climber_data,
                test_results,
                data_type=self.data_type,
                test_type=self.test_type,
                force_file=(self.force_file.filename if self.force_file else None),
                nirs_file=(self.nirs_file.filename if self.nirs_file else None)
            )
            self.report_window.show()

    def close_event(self, event):
        """
        Handles the window close event.

        Stops both timers and, if finalization hasn't occurred yet, generates final graphs and saves the CSV filenames
        into the database in one row. Then closes both CSV files.

        Parameters:
            event (QCloseEvent): The close event.
        """
        if self.force_timer.isActive():
            self.force_timer.stop()
        if self.nirs_timer.isActive():
            self.nirs_timer.stop()
        if not self.finalized:
            self.finalize_acquisition()
        self.force_file.close()
        self.nirs_file.close()
        event.accept()


def generate_final_graph_force(force_file):
    """
    Generates a final static graph that plots force data.

    Parameters:
        force_file (str): Filename of the NIRS data h5.
    """
    # Read Force data.
    # force_df = pd.read_csv(force_file)
    # force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
    # force_timestamps = force_df['timestamp'].values
    # force_values = force_df['value'].values

    force_df = pd.read_hdf(force_file, key="data")
    force_df['timestamp'] = force_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
    force_timestamps = force_df['timestamp'].values
    force_values = force_df['value'].values

    # Create plot for Force only.
    fig, ax = plt.subplots()
    ax.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (kg)", color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True)
    fig.tight_layout()
    plt.title("Final Force Data")
    ax.legend(loc="upper right")
    return fig
    # plt.show()


def generate_final_graph_nirs(nirs_file):
    """
    Generates a final static graph that plots NIRS data.

    Parameters:
        nirs_file (str): Filename of the NIRS data h5.
    """
    # # Read NIRS data.
    # nirs_df = pd.read_csv(nirs_file)
    # nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
    # nirs_timestamps = nirs_df['timestamp'].values
    # nirs_values = nirs_df['value'].values

    # Read NIRS data.
    nirs_df = pd.read_hdf(nirs_file, key="data")
    nirs_df['timestamp'] = nirs_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
    nirs_timestamps = nirs_df['timestamp'].values
    nirs_values = nirs_df['value'].values

    # Create plot for NIRS only.
    fig, ax = plt.subplots()
    ax.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NIRS (%)", color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.grid(True)
    fig.tight_layout()
    plt.title("Final NIRS Data")
    ax.legend(loc="upper right")
    return fig
    # plt.show()


def generate_final_combined_graph(force_file, nirs_file):
    """
    Generates a final static graph that plots both Force and NIRS data on a single figure with two y-axes.

    Parameters:
        force_file (str): Filename of the Force data h5.
        nirs_file (str): Filename of the NIRS data h5.
    """
    # # Read Force data.
    # force_df = pd.read_csv(force_file)
    # force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
    # force_timestamps = force_df['timestamp'].values
    # force_values = force_df['value'].values
    #
    # # Read NIRS data.
    # nirs_df = pd.read_csv(nirs_file)
    # nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
    # nirs_timestamps = nirs_df['timestamp'].values
    # nirs_values = nirs_df['value'].values

    # Read Force data.
    force_df = pd.read_hdf(force_file, key="data")
    force_df['timestamp'] = force_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
    force_timestamps = force_df['timestamp'].values
    force_values = force_df['value'].values

    # Read NIRS data.
    nirs_df = pd.read_hdf(nirs_file, key="data")
    nirs_df['timestamp'] = nirs_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
    nirs_timestamps = nirs_df['timestamp'].values
    nirs_values = nirs_df['value'].values

    # Create a combined plot.
    fig, ax1 = plt.subplots()
    ax1.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (kg)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
    ax2.set_ylabel("NIRS (%)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Gather legend handles & labels from both axes and combine them:
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    plt.title("Final Combined Sensor Data")
    return fig
    # plt.show()


# The main function is omitted here since you now launch the CombinedDataCommunicator
# from your TestPage when needed.
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     # For standalone testing, load sensor data as before:
#     force_df = pd.read_excel("group2_ao_copy.xlsx").dropna()
#     force_timestamps = force_df.iloc[:, 0].values
#     force_values = force_df.iloc[:, 2].values
#
#     nirs_df = pd.read_excel("group3_ao_copy.xlsx").dropna()
#     nirs_timestamps = nirs_df.iloc[:, 0].values
#     nirs_values = nirs_df.iloc[:, 3].values
#
#     window = CombinedDataCommunicator(force_timestamps, force_values,
#                                       nirs_timestamps, nirs_values,
#                                       admin_id="TestAdmin", climber_id="TestClimber",
#                                       window_size=60, fixed_offset_ratio=1, auto_start=True)
#     window.show()
#     app.exec()
