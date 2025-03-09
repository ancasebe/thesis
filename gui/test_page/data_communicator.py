import os
import csv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt
from datetime import datetime
from gui.test_page.test_db_manager import ClimbingTestManager


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
        print(folder_path)
        filename = f"{folder_path}/{prefix}_{timestamp}.csv"
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
    def __init__(self, force_timestamps, force_values, nirs_timestamps, nirs_values,
                 admin_id, climber_id, window_size=60, auto_start=False, test_label="ao_force"):
        super().__init__()
        self.timestamp = None
        self.force_csv = None
        self.nirs_csv = None

        self.admin_id = admin_id
        self.climber_id = climber_id
        self.auto_start = auto_start
        self.test_label = test_label  # new parameter indicating the test and data type

        self.x_min = -window_size
        self.x_max = 0
        # self.fixed_offset = self.x_min + window_size * fixed_offset_ratio

        self.force_y_range = {"min": np.min(force_values) - 5, "max": np.max(force_values) + 5}
        self.nirs_y_range = {"min": np.min(nirs_values) - 5, "max": np.max(nirs_values) + 5}

        self.force_data = {
            "timestamps": force_timestamps,
            "values": force_values,
            "index": 0,
            "num_points": len(force_timestamps)
        }
        self.nirs_data = {
            "timestamps": nirs_timestamps,
            "values": nirs_values,
            "index": 0,
            "num_points": len(nirs_timestamps)
        }

        # self.force_csv = CSVLogger(folder="tests", prefix="processed_force")
        # self.nirs_csv = CSVLogger(folder="tests", prefix="processed_nirs")

        self.finalized = False
        self._setup_ui()

        # # Create curves and live text items.
        # self.force_curve = self.plot_widget.plot([], [], pen='b', name="Force [kg]")
        # self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
        # self.nirs_view.addItem(self.nirs_curve)
        #
        # self.force_text = pg.TextItem("", anchor=(0, 0))
        # self.force_text.setPos(self.x_min, self.force_y_range["max"])
        # self.plot_widget.addItem(self.force_text)
        # self.nirs_text = pg.TextItem("", anchor=(1, 0))
        # self.nirs_text.setPos(self.x_max, self.nirs_y_range["max"])
        # self.nirs_view.addItem(self.nirs_text)

        # Set up timers but do not start automatically unless auto_start is True.
        self.force_timer = QTimer()
        self.force_timer.timeout.connect(self.update_force)
        self.nirs_timer = QTimer()
        self.nirs_timer.timeout.connect(self.update_nirs)

        if self.auto_start:
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

        # Force axis (left)
        self.plot_widget.setXRange(self.x_min, self.x_max)
        self.plot_widget.setYRange(self.force_y_range["min"], self.force_y_range["max"])
        self.plot_widget.getAxis('bottom').setLabel("Time (s)")
        self.plot_widget.getAxis('left').setLabel("Force (kg)")  # <-- explicit label
        self.plot_widget.showAxis('right')  # keep the right axis visible
        self.plot_widget.getAxis('right').setLabel('NIRS (%)')

        # Create a second ViewBox for NIRS data
        self.nirs_view = pg.ViewBox()
        self.plot_widget.scene().addItem(self.nirs_view)
        # Link the right axis to the NIRS view
        self.plot_widget.getAxis('right').linkToView(self.nirs_view)

        # Set Y range for the NIRS axis
        self.nirs_view.setYRange(self.nirs_y_range["min"], self.nirs_y_range["max"], padding=0)
        # Set X range for the NIRS axis
        self.nirs_view.setXRange(self.x_min, self.x_max, padding=0)

        # Make sure the NIRS view is resized with the main view
        self.plot_widget.getViewBox().sigResized.connect(self._update_views)

        # Create curves
        self.force_curve = self.plot_widget.plot([], [], pen='y', name="Force [kg]")
        self.nirs_curve = pg.PlotCurveItem(pen='r', name="NIRS (%)")
        self.nirs_view.addItem(self.nirs_curve)

        # Add a single legend to the main plot
        legend = self.plot_widget.addLegend()
        legend.setParentItem(self.plot_widget.getViewBox())
        legend.anchor((0.5, 0), (0.5, 0))
        # legend.setPos((self.x_min + self.x_max) / 2, self.force_y_range["max"])
        # Manually register both curves with the legend
        legend.addItem(self.force_curve, "Force [kg]")
        legend.addItem(self.nirs_curve, "NIRS (%)")

        # Force readout text
        self.force_text = pg.TextItem("", anchor=(0, 0))
        # self.force_text.setPos(self.x_min, self.force_y_range["max"])
        self.plot_widget.addItem(self.force_text)

        # NIRS readout text
        self.nirs_text = pg.TextItem("", anchor=(1, 0))
        # self.nirs_text.setPos(self.x_max, self.nirs_y_range["max"])
        self.nirs_view.addItem(self.nirs_text)

    def _update_views(self):
        """Synchronize the geometry of the secondary NIRS view with the main view."""
        self.nirs_view.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        # Link X movements
        self.nirs_view.linkedViewChanged(self.plot_widget.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        """Starts the data acquisition by starting the timers."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Depending on test_label, create CSVLogger(s)
        if self.test_label in ["ao_force", "ao_force_nirs"]:
            self.force_csv = CSVLogger(folder="tests", prefix=f"ao_force",
                                       timestamp=self.timestamp)
        else:
            self.force_csv = None

        if self.test_label in ["ao_nirs", "ao_force_nirs"]:
            self.nirs_csv = CSVLogger(folder="tests", prefix=f"ao_nirs",
                                      timestamp=self.timestamp)
        else:
            self.nirs_csv = None

        if self.force_csv is not None:
            self.force_timer.start(10)
        elif self.nirs_csv is not None:
            self.nirs_timer.start(10)
        # self.force_csv = CSVLogger(folder="tests", prefix="processed_force", timestamp=self.timestamp)
        # self.nirs_csv = CSVLogger(folder="tests", prefix="processed_nirs", timestamp=self.timestamp)
        # self.force_timer.start(10)
        # self.nirs_timer.start(10)

    def update_force(self):
        """
        Updates the Force data plot.

        Logs the current Force data point, transforms its x value so that the newest point appears at the fixed offset,
        and updates the force curve and live text readout.
        """
        if self.force_csv is None:
            return
        if self.force_data["index"] >= (self.force_data["num_points"] - 1):
            self.force_timer.stop()
            self.finalize_acquisition()
        else:
            current_ts = self.force_data["timestamps"][self.force_data["index"]]
            current_val = self.force_data["values"][self.force_data["index"]]
            self.force_csv.log(current_ts, current_val)
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
        if self.nirs_csv is None:
            return
        if self.nirs_data["index"] >= (self.nirs_data["num_points"] - 1):
            self.nirs_timer.stop()
            self.finalize_acquisition()
        else:
            current_ts = self.nirs_data["timestamps"][self.nirs_data["index"]]
            current_val = self.nirs_data["values"][self.nirs_data["index"]]
            self.nirs_csv.log(current_ts, current_val)
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
        if self.force_timer.isActive():
            self.force_timer.stop()
        if self.nirs_timer.isActive():
            self.nirs_timer.stop()
        self.finalize_acquisition()

    def finalize_acquisition(self):
        """Generate the final graph and saves data to a database, if not done yet."""
        if not self.finalized:

            # Generate a final graph depending on selected test type.
            if self.test_label == "ao_force":
                generate_final_graph_force(self.force_csv.filename)
            elif self.test_label == "ao_nirs":
                generate_final_graph_nirs(self.nirs_csv.filename)
            elif self.test_label == "ao_force_nirs":
                generate_final_combined_graph(self.force_csv.filename, self.nirs_csv.filename)
            else:
                print("Unknown test type; skipping graph generation.")
            # Save CSV filenames in the database using the provided admin_id and climber_id.
            db_manager = ClimbingTestManager()
            # Here you could include the test type in the stored file_paths if desired.
            file_paths = ""
            if self.force_csv is not None:
                file_paths += self.force_csv.filename
            if self.nirs_csv is not None:
                file_paths += ("; " if file_paths else "") + self.nirs_csv.filename
            print(file_paths)
            # For test_results, you can add further evaluation data.
            test_results = str({"evaluation": "placeholder"})
            db_manager.add_test_result(str(self.admin_id), str(self.climber_id), self.timestamp,
                                       file_paths, test_results)
            db_manager.close_connection()
            self.finalized = True

            # # Generate the final combined graph.
            # generate_final_combined_graph(self.force_csv.filename, self.nirs_csv.filename)
            # # Save CSV filenames in the database using the provided admin_id and climber_id.
            # db_manager = ClimbingTestManager()
            #
            # test_results = str({"maximal_force": 65.34, "critical_force": 34.54, "w_prime": 6543.43})
            # file_paths = str((self.force_csv.filename, self.nirs_csv.filename))
            # db_manager.add_test_result(str(self.admin_id), str(self.climber_id), self.timestamp,
            #                            file_paths, test_results)
            # db_manager.close_connection()
            # self.finalized = True

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
        self.force_csv.close()
        self.nirs_csv.close()
        event.accept()
def generate_final_graph_force(force_csv):
    # Read Force data.
    force_df = pd.read_csv(force_csv)
    force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
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


# def generate_final_graph(force_csv=None, nirs_csv=None, plot_type="combined"):
#     """
#     plot_type: "force", "nirs", or "combined"
#     """
#     if plot_type == "force":
#         if force_csv is None:
#             print("No force CSV provided")
#             return
#         # Read Force data.
#         force_df = pd.read_csv(force_csv)
#         force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
#         force_timestamps = force_df['timestamp'].values
#         force_values = force_df['value'].values
#
#         fig, ax = plt.subplots()
#         ax.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Force (kg)", color='b')
#         ax.tick_params(axis='y', labelcolor='b')
#         ax.grid(True)
#         fig.tight_layout()
#         plt.title("Final Force Data")
#         ax.legend(loc="upper left")
#         plt.show()
#
#     elif plot_type == "nirs":
#         if nirs_csv is None:
#             print("No NIRS CSV provided")
#             return
#         # Read NIRS data.
#         nirs_df = pd.read_csv(nirs_csv)
#         nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
#         nirs_timestamps = nirs_df['timestamp'].values
#         nirs_values = nirs_df['value'].values
#
#         fig, ax = plt.subplots()
#         ax.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("NIRS (%)", color='r')
#         ax.tick_params(axis='y', labelcolor='r')
#         ax.grid(True)
#         fig.tight_layout()
#         plt.title("Final NIRS Data")
#         ax.legend(loc="upper left")
#         plt.show()
#
#     elif plot_type == "combined":
#         if force_csv is None or nirs_csv is None:
#             print("Force CSV and/or NIRS CSV not provided")
#             return
#         # Read Force data.
#         force_df = pd.read_csv(force_csv)
#         force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
#         force_timestamps = force_df['timestamp'].values
#         force_values = force_df['value'].values
#
#         # Read NIRS data.
#         nirs_df = pd.read_csv(nirs_csv)
#         nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
#         nirs_timestamps = nirs_df['timestamp'].values
#         nirs_values = nirs_df['value'].values
#
#         fig, ax1 = plt.subplots()
#         ax1.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
#         ax1.set_xlabel("Time (s)")
#         ax1.set_ylabel("Force (kg)", color='b')
#         ax1.tick_params(axis='y', labelcolor='b')
#         ax1.grid(True)
#
#         ax2 = ax1.twinx()
#         ax2.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
#         ax2.set_ylabel("NIRS (%)", color='r')
#         ax2.tick_params(axis='y', labelcolor='r')
#
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
#
#         fig.tight_layout()
#         plt.title("Final Combined Sensor Data")
#         plt.show()
#
#     else:
#         print("Unknown plot type.")

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
