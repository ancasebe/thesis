"""
Data logging module for the Climbing Testing Application.

This module defines the FeatherBinaryLogger class which handles the efficient
storage of time series data collected during climbing tests. It buffers data
in memory and writes it to disk in the Apache Feather format for fast access
and efficient storage.

Key functionalities:
- Buffer data points in memory during acquisition
- Write data efficiently to disk in Feather format
- Handle timestamped data pairs
- Manage file paths and naming conventions
- Support for database queue integration

The Feather logger provides a performant data storage solution optimized for
the high-frequency sampling required by the climbing tests, balancing memory
usage and disk I/O.
"""
import os
import pandas as pd


class FeatherBinaryLogger:
    """
    Logs data points into an in-memory buffer and writes them to a Feather file.
    Note: Feather does not support appending, so all data is stored in memory
    and written out at once when close() is called.
    """

    def __init__(self, folder="tests", prefix="processed_data", timestamp="today", db_queue=None):
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder)
        self.filename = f"{prefix}_{timestamp}.feather"
        self.actual_filename = os.path.join(folder_path, self.filename)
        self.db_queue = db_queue
        self.buffer = []  # in-memory buffer for data points

    def log(self, ts, val):
        """Append a data point to the buffer."""
        self.buffer.append([ts, val])

    def flush(self):
        """Write the buffered data to the Feather file and clear the buffer."""
        if self.buffer:
            df = pd.DataFrame(self.buffer, columns=["time", "value"])
            # Write the entire DataFrame to a Feather file
            df.reset_index(drop=True).to_feather(self.actual_filename)
            self.buffer = []

    def close(self):
        """Flush any remaining data and write to the Feather file."""
        self.flush()