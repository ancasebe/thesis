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
        # filename = os.path.join(folder, f"{prefix}_{timestamp}.feather")
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