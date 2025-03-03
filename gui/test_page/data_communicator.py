"""
Merged Data Manager

This module integrates real-time data visualization and data storage for force and NIRS sensors.
It collects, processes, and visualizes incoming data from both Serial (Force) and Bluetooth (NIRS) sources.

Main Features:
- Collects force and NIRS data via Serial and Bluetooth.
- Stores collected data into CSV files.
- Provides real-time visualization with Matplotlib.
- Ensures synchronized data management.
- Generates a final summary graph after the test stops.
"""

import queue
import time
import csv
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
from data_generator import BluetoothCommunicator, SerialCommunicator, FORCE_SENSOR_ID, O2HB_SENSOR_ID, SMO2_SENSOR_ID, \
    HHB_SENSOR_ID


class DataManager:
    """
    Manages data collection, storage, and real-time visualization of force and NIRS data.

    Attributes:
    - force_comm: Handles force sensor communication (Serial)
    - nirs_comm: Handles NIRS sensor communication (Bluetooth)
    - db_queque, viz_queque: Queues for managing incoming data
    - force_filename, nirs_filename: CSV filenames for storing collected data
    - force_file, nirs_file: File handlers for writing data
    - force_values, nirs_values, time_values: Deques for storing time-series data for plotting
    """

    def __init__(self):
        """Initializes the DataManager, setting up communication, files, and visualization structures."""
        # Initialize queues for both NIRS and Force data
        self.db_queque = queue.Queue()
        self.viz_queque = queue.Queue()

        # Initialize both Serial (Force) and Bluetooth (NIRS) Communicators
        self.force_comm = SerialCommunicator(visualization_que=self.viz_queque, database_que=self.db_queque)
        self.nirs_comm = BluetoothCommunicator(visualization_que=self.viz_queque, database_que=self.db_queque)

        # Generate unique filenames for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.force_filename = f"tests/force_{timestamp}.csv"
        self.nirs_filename = f"tests/nirs_{timestamp}.csv"

        # Open CSV files and write headers
        self.force_file = open(self.force_filename, "w", newline="")
        self.force_writer = csv.writer(self.force_file)
        self.force_writer.writerow(["timestamp", "force_value"])

        self.nirs_file = open(self.nirs_filename, "w", newline="")
        self.nirs_writer = csv.writer(self.nirs_file)
        self.nirs_writer.writerow(["timestamp", "sensor_type", "value"])

        # Initialize real-time plot
        self.window_size = 100  # Number of points to display
        self.force_values = deque([0] * self.window_size, maxlen=self.window_size)
        self.time_values = deque([0] * self.window_size, maxlen=self.window_size)
        self.nirs_values = deque([0] * self.window_size, maxlen=self.window_size)

        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

        self.force_line, = self.ax1.plot(self.time_values, self.force_values, 'b-', label="Force (N)")
        self.nirs_line, = self.ax2.plot(self.time_values, self.nirs_values, 'r-', label="NIRS (%)")

        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(0, 100)
        self.ax2.set_ylim(50, 100)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Force (N)", color='b')
        self.ax2.set_ylabel("NIRS (%)", color='r')
        self.ax1.grid(True)
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

    def start_data_collection(self):
        """Starts data collection for both force and NIRS sensors in separate threads."""
        force_thread = threading.Thread(target=self.force_comm.start_serial_collection)
        nirs_thread = threading.Thread(target=self.nirs_comm.start_bluetooth_collector)

        force_thread.start()
        nirs_thread.start()

        print(f"Collecting data... Saving to {self.force_filename} and {self.nirs_filename}")

    def update_plot(self, frame):
        """Updates the Matplotlib plot with real-time force and NIRS data."""
        # Update Force Data
        if not self.force_comm.db_queque.empty():
            raw_data = self.force_comm.db_queque.get()
            try:
                _, ts, value = raw_data.split("_")
                self.force_writer.writerow([ts, value])
                self.force_values.append(float(value))
                self.time_values.append(float(ts))
            except ValueError:
                pass

        # Update NIRS Data
        if not self.nirs_comm.db_queque.empty():
            raw_data = self.nirs_comm.db_queque.get()
            try:
                _, ts, value, _, _ = raw_data.split("_")
                self.nirs_writer.writerow([ts, value])
                self.nirs_values.append(float(value))
            except ValueError:
                pass

        # Update plot data
        self.force_line.set_data(range(len(self.time_values)), self.force_values)
        self.nirs_line.set_data(range(len(self.time_values)), self.nirs_values)
        self.ax1.set_xlim(0, len(self.time_values))
        return self.force_line, self.nirs_line

    def start_visualization(self):
        """Starts the real-time Matplotlib animation for visualizing collected data."""
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)
        plt.show()

    def generate_final_graph(self):
        """Generates a final graph from all collected data when the test stops."""

        # Check if force file has data
        try:
            force_data = np.loadtxt(self.force_filename, delimiter=",", skiprows=1)
        except OSError:
            print(f"Warning: Force data file {self.force_filename} not found.")
            force_data = np.array([])

        # Check if nirs file has data
        try:
            nirs_data = np.loadtxt(self.nirs_filename, delimiter=",", skiprows=1)
        except OSError:
            print(f"Warning: NIRS data file {self.nirs_filename} not found.")
            nirs_data = np.array([])

        # Ensure data is not empty before plotting
        if force_data.size == 0 and nirs_data.size == 0:
            print("No valid data to plot.")
            return

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        if force_data.size > 0 and force_data.ndim > 1:
            ax1.plot(force_data[:, 0], force_data[:, 1], 'b-', label="Force (N)")

        if nirs_data.size > 0 and nirs_data.ndim > 1:
            ax2.plot(nirs_data[:, 0], nirs_data[:, 1], 'r-', label="NIRS (%)")

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Force (N)", color='b')
        ax2.set_ylabel("NIRS (%)", color='r')
        ax1.grid(True)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.title("Final Test Data")
        plt.show()

    def stop_collection(self):
        """Stops data collection, closes connections, and generates the final graph."""
        print("Stopping data collection...")
        self.force_comm.stop_serial_collection()
        self.nirs_comm.stop_bluetooth_collector()

        self.force_file.close()
        self.nirs_file.close()

        self.generate_final_graph()


if __name__ == "__main__":
    manager = DataManager()
    manager.start_data_collection()
    manager.start_visualization()
    manager.stop_collection()
