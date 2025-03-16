import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QDialog, QPushButton, QHBoxLayout
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt
from datetime import datetime
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.all_out import AllOutTest


# ---------------------------
# BufferedBinaryLogger using HDF5
# ---------------------------
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
        self.store = pd.HDFStore(self.filename, mode='w', complib='blosc', complevel=9)
        # Initialize the table with an empty DataFrame
        self.store.put(self.key, pd.DataFrame(columns=["timestamp", "value"]), format='table', append=False)

    def log(self, ts, val):
        self.buffer.append([f"timestamp_{ts}", val])
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=["timestamp", "value"])
            self.store.append(self.key, df, format='table', data_columns=True)
            self.buffer = []

    def close(self):
        self.flush()
        self.store.close()


# ---------------------------
# Base GUI Setup
# ---------------------------
class CommunicatorGUI(QMainWindow):
    """
    Base GUI class that sets up the main window and PlotWidget.
    """
    def __init__(self, window_size=60):
        super().__init__()
        self.x_min = -window_size
        self.x_max = 0
        self._setup_ui()

    def _setup_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.plot_widget.setXRange(self.x_min, self.x_max)


# ---------------------------
# Force Data Communicator
# ---------------------------
class ForceDataCommunicator:
    """
    Handles acquisition and plotting for Force sensor data.
    """
    def __init__(self, force_data, force_y_range, x_min, x_max, flush_interval=1000):
        self.force_data = force_data  # dict with keys: timestamps, values, index, num_points
        self.force_y_range = force_y_range
        self.x_min = x_min
        self.x_max = x_max
        self.flush_interval = flush_interval
        self.logger = None  # To be set at start_acquisition
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.curve = None   # To be assigned by the GUI
        self.text_item = None  # To be assigned by the GUI

    def start(self, interval=10):
        self.timer.start(interval)

    def update(self):
        if self.force_data["index"] >= (self.force_data["num_points"] - 1):
            self.timer.stop()
        else:
            current_ts = self.force_data["timestamps"][self.force_data["index"]]
            current_val = self.force_data["values"][self.force_data["index"]]
            self.logger.log(current_ts, current_val)
            x_data = (self.force_data["timestamps"][:self.force_data["index"]+1] - current_ts)
            y_data = self.force_data["values"][:self.force_data["index"]+1]
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            if self.curve is not None:
                self.curve.setData(x_data, y_data)
            if self.text_item is not None:
                self.text_item.setText(f"Force Time: {current_ts:.2f} s\nForce: {current_val:.2f} kg")
                self.text_item.setPos(self.x_min, self.force_y_range["max"])
            self.force_data["index"] += 1

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        self.logger.close()

    def final_plot(self):
        # Read binary data from HDF5 file and plot.
        df = pd.read_hdf(self.logger.filename, key="data")
        df['timestamp'] = df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'].values, df['value'].values, 'b-', label="Force [kg]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (kg)", color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True)
        fig.tight_layout()
        plt.title("Final Force Data")
        ax.legend(loc="upper left")
        plt.show()

# ---------------------------
# NIRS Data Communicator
# ---------------------------
class NIRSDataCommunicator:
    """
    Handles acquisition and plotting for NIRS sensor data.
    """
    def __init__(self, nirs_data, nirs_y_range, x_min, x_max, flush_interval=1000):
        self.nirs_data = nirs_data  # dict with keys: timestamps, values, index, num_points
        self.nirs_y_range = nirs_y_range
        self.x_min = x_min
        self.x_max = x_max
        self.flush_interval = flush_interval
        self.logger = None  # To be set at start_acquisition
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.curve = None   # To be assigned by the GUI
        self.text_item = None  # To be assigned by the GUI

    def start(self, interval=10):
        self.timer.start(interval)

    def update(self):
        if self.nirs_data["index"] >= (self.nirs_data["num_points"] - 1):
            self.timer.stop()
        else:
            current_ts = self.nirs_data["timestamps"][self.nirs_data["index"]]
            current_val = self.nirs_data["values"][self.nirs_data["index"]]
            self.logger.log(current_ts, current_val)
            x_data = (self.nirs_data["timestamps"][:self.nirs_data["index"]+1] - current_ts)
            y_data = self.nirs_data["values"][:self.nirs_data["index"]+1]
            mask = (x_data >= self.x_min) & (x_data <= self.x_max)
            x_data = x_data[mask]
            y_data = y_data[mask]
            if self.curve is not None:
                self.curve.setData(x_data, y_data)
            if self.text_item is not None:
                self.text_item.setText(f"NIRS Time: {current_ts:.2f} s\nNIRS: {current_val:.2f} %")
                self.text_item.setPos(self.x_max, self.nirs_y_range["max"])
            self.nirs_data["index"] += 1

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        self.logger.close()

    def final_plot(self):
        df = pd.read_hdf(self.logger.filename, key="data")
        df['timestamp'] = df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'].values, df['value'].values, 'r-', label="NIRS (%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIRS (%)", color='r')
        ax.tick_params(axis='y', labelcolor='r')
        ax.grid(True)
        fig.tight_layout()
        plt.title("Final NIRS Data")
        ax.legend(loc="upper left")
        plt.show()


# ---------------------------
# Combined Communicator (Wrapper)
# ---------------------------
class CombinedDataCommunicator(CommunicatorGUI):
    """
    Sets up the GUI and instantiates one or both specialized communicators
    depending on the test type (force-only, NIRS-only, or combined).
    """
    def __init__(self, admin_id, climber_id, arm_tested, window_size=60, auto_start=False,
                 test_label="ao_force", test_id="ao"):
        # Save parameters.
        self.timestamp = None
        self.admin_id = admin_id
        self.climber_id = climber_id
        if arm_tested == "Dominant":
            self.arm_tested = "D"
        elif arm_tested == "Non-dominant":
            self.arm_tested = "N"
        else:
            self.arm_tested = "-"
        self.test_label = test_label  # "ao_force", "ao_nirs", or "ao_force_nirs"
        self.test_type = test_id
        script_dir = os.path.dirname(__file__)
        if test_label != "ao_nirs":  # Need force data.
            force_file = os.path.join(script_dir, "group2_ao_copy.xlsx")
            force_df = pd.read_excel(force_file).dropna()
            force_timestamps = force_df.iloc[:, 0].values
            force_values = force_df.iloc[:, 3].values
            self.force_y_range = {"min": np.min(force_values)-5, "max": np.max(force_values)+5}
            self.force_data = {
                "timestamps": force_timestamps,
                "values": force_values,
                "index": 0,
                "num_points": len(force_timestamps)
            }
        if test_label != "ao_force":  # Need NIRS data.
            nirs_file = os.path.join(script_dir, "group3_ao_copy.xlsx")
            nirs_df = pd.read_excel(nirs_file).dropna()
            nirs_timestamps = nirs_df.iloc[:, 0].values
            nirs_values = nirs_df.iloc[:, 2].values
            self.nirs_y_range = {"min": np.min(nirs_values)-5, "max": np.max(nirs_values)+5}
            self.nirs_data = {
                "timestamps": nirs_timestamps,
                "values": nirs_values,
                "index": 0,
                "num_points": len(nirs_timestamps)
            }
        super().__init__(window_size=window_size)
        self.force_comm = None
        self.nirs_comm = None
        if test_label in ["ao_force", "ao_force_nirs"]:
            self.force_comm = ForceDataCommunicator(self.force_data, self.force_y_range, self.x_min, self.x_max)
            self.force_comm.curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
            self.force_comm.text_item = pg.TextItem("", anchor=(0, 0))
            self.plot_widget.addItem(self.force_comm.text_item)
        if test_label in ["ao_nirs", "ao_force_nirs"]:
            if test_label == "ao_nirs":
                self.nirs_comm = NIRSDataCommunicator(self.nirs_data, self.nirs_y_range, self.x_min, self.x_max)
                self.nirs_comm.curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
                self.plot_widget.addItem(self.nirs_comm.curve)
                self.nirs_comm.text_item = pg.TextItem("", anchor=(1, 0))
                self.plot_widget.addItem(self.nirs_comm.text_item)
            else:
                self.nirs_comm = NIRSDataCommunicator(self.nirs_data, self.nirs_y_range, self.x_min, self.x_max)
                self.nirs_view = pg.ViewBox()
                self.plot_widget.scene().addItem(self.nirs_view)
                self.plot_widget.getAxis('right').linkToView(self.nirs_view)
                self.nirs_view.setYRange(self.nirs_y_range["min"], self.nirs_y_range["max"], padding=0)
                self.nirs_view.setXRange(self.x_min, self.x_max, padding=0)
                self.plot_widget.getViewBox().sigResized.connect(self._update_views)
                self.nirs_comm.curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
                self.nirs_view.addItem(self.nirs_comm.curve)
                self.nirs_comm.text_item = pg.TextItem("", anchor=(1, 0))
                self.nirs_view.addItem(self.nirs_comm.text_item)
        legend = self.plot_widget.addLegend()
        legend.setParentItem(self.plot_widget.getViewBox())
        legend.anchor((0.5, 0), (0.5, 0))
        if test_label == "ao_force":
            legend.addItem(self.force_comm.curve, "Force [kg]")
        elif test_label == "ao_nirs":
            legend.addItem(self.nirs_comm.curve, "NIRS (%)")
        else:
            legend.addItem(self.force_comm.curve, "Force [kg]")
            legend.addItem(self.nirs_comm.curve, "NIRS (%)")
        if auto_start:
            self.start_acquisition()

    def _update_views(self):
        self.nirs_view.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        self.nirs_view.linkedViewChanged(self.plot_widget.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.force_comm is not None:
            self.force_comm.logger = BufferedBinaryLogger(folder="tests", prefix="ao_force", timestamp=self.timestamp)
            self.force_comm.start(10)
        if self.nirs_comm is not None:
            self.nirs_comm.logger = BufferedBinaryLogger(folder="tests", prefix="ao_nirs", timestamp=self.timestamp)
            self.nirs_comm.start(10)

    def stop_acquisition(self):
        if self.force_comm is not None:
            self.force_comm.stop()
        if self.nirs_comm is not None:
            self.nirs_comm.stop()
        self.finalize_acquisition()

    def finalize_acquisition(self):
        if self.test_label == "ao_force":
            generate_final_graph_force(self.force_comm.logger.filename)
            # self.force_comm.final_plot()
            evaluator = AllOutTest(self.force_comm.logger.filename)
            test_results = evaluator.evaluate()
        elif self.test_label == "ao_nirs":
            generate_final_graph_nirs(self.nirs_comm.logger.filename)
            # self.nirs_comm.final_plot()
            test_results = str({"evaluation": "placeholder"})
        elif self.test_label == "ao_force_nirs":
            generate_final_combined_graph(self.force_comm.logger.filename, self.nirs_comm.logger.filename)
            # self.force_comm.final_plot()
            # self.nirs_comm.final_plot()
            evaluator = AllOutTest(self.force_comm.logger.filename)
            test_results = evaluator.evaluate()
        else:
            test_results = str({"evaluation": "placeholder"})
        db_manager = ClimbingTestManager()
        file_paths = ""
        if self.force_comm is not None:
            file_paths += self.force_comm.logger.filename
        if self.nirs_comm is not None:
            file_paths += ("; " if file_paths else "") + self.nirs_comm.logger.filename
        db_manager.add_test_result(str(self.admin_id), str(self.climber_id), self.arm_tested,
                                   self.timestamp, file_paths, str(test_results))
        db_manager.close_connection()

    def close_event(self, event):
        if self.force_comm is not None and self.force_comm.timer.isActive():
            self.force_comm.timer.stop()
        if self.nirs_comm is not None and self.nirs_comm.timer.isActive():
            self.nirs_comm.timer.stop()
        self.finalize_acquisition()
        if self.force_comm is not None:
            self.force_comm.logger.close()
        if self.nirs_comm is not None:
            self.nirs_comm.logger.close()
        event.accept()


def generate_final_graph_force(force_file):
    """
    Generates a final static graph that plots force data.

    Parameters:
        force_csv (str): Filename of the NIRS data CSV.
    """

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
    ax.legend(loc="upper left")
    plt.show()


def generate_final_graph_nirs(nirs_csv):
    """
    Generates a final static graph that plots NIRS data.

    Parameters:
        nirs_csv (str): Filename of the NIRS data CSV.
    """
    # Read NIRS data.
    nirs_df = pd.read_csv(nirs_csv)
    nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
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
    ax.legend(loc="upper left")
    plt.show()

def generate_final_combined_graph(force_csv, nirs_csv):
    """
    Generates a final static graph that plots both Force and NIRS data on a single figure with two y-axes.

    Parameters:
        force_csv (str): Filename of the Force data CSV.
        nirs_csv (str): Filename of the NIRS data CSV.
    """
    # Read Force data.
    force_df = pd.read_csv(force_csv)
    force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
    force_timestamps = force_df['timestamp'].values
    force_values = force_df['value'].values

    # Read NIRS data.
    nirs_df = pd.read_csv(nirs_csv)
    nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
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
    # plt.legend(loc="upper left")
    plt.show()