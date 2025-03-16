import os
import pandas as pd
import numpy as np
import peakutils
import matplotlib.pyplot as plt


class AllOutTest:
    """
    Class for evaluating all-out test data.

    It reads the force data from a CSV file and computes:
      - Maximum strength (the maximum force value)
      - Critical force (average of the last three detected peaks)
      - w_prime (an integrated measure of the force above the critical force)

    Additional evaluation functions can be added as methods.
    """

    def __init__(self, file_path):
        """
        Initialize the evaluator by reading the CSV file.

        Parameters:
            file_path (str): Path to the file containing force data.
                             Expected to have at least 'force_timestamp' and 'value' columns.
        """
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"File not found: {file_path}")
        # self.df = pd.read_csv(file_path)
        # # Ensure required columns exist
        # required_columns = ['force_timestamp', 'value']
        # for col in required_columns:
        #     if col not in self.df.columns:
        #         raise ValueError(f"Missing required column: {col}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            # Read from HDF5 using the key "data"
        self.df = pd.read_hdf(file_path, key="data")
        # Check for required columns.
        required_columns = ['timestamp', 'value']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def compute_max_strength(self):
        """
        Compute the maximum strength from the force data.

        Returns:
            float: The maximum force value.
        """
        return self.df['value'].max()

    def compute_endurance(self):
        """
        Compute the endurance metrics:
         - Critical force: the mean of the last three peaks in the force data.
         - w_prime: the integrated difference between the force values and the critical force,
                    summed over indices up to a calculated finish index.

        Returns:
            tuple: (critical_force, w_prime)
        """
        # Detect peaks in the force data
        force_values = self.df['value'].values
        peaks = peakutils.indexes(force_values, thres=0.05, min_dist=700)

        if len(peaks) < 3:
            # Not enough peaks to compute critical force reliably.
            critical_force = None
            w_prime = None
        else:
            # Compute critical force as the mean of the last three detected peaks.
            last_three = force_values[peaks[-3:]]
            critical_force = np.mean(last_three)

            # Define finish index (example: last three peaks index minus 300).
            finish = peaks[-3] - 300
            if finish < 1:
                finish = len(force_values)

            # Compute w_prime: sum the amount by which force exceeds the critical force
            # divided by 100 (adjust the divisor as needed).
            w_prime = 0
            for i in range(1, int(finish)):
                if force_values[i] > critical_force:
                    w_prime += (force_values[i] - critical_force) / 100
        return critical_force, w_prime

    def evaluate(self):
        """
        Evaluate the test by computing maximum strength, critical force, and w_prime.

        Returns:
            dict: A dictionary with the computed metrics.
                  For example:
                  {
                      'max_strength': 123.45,
                      'critical_force': 67.89,
                      'w_prime': 1234.56
                  }
        """
        max_strength = self.compute_max_strength()
        critical_force, w_prime = self.compute_endurance()
        return {
            'max_strength': max_strength,
            'critical_force': critical_force,
            'w_prime': w_prime
        }

    # (Optional) Additional evaluation functions can be added as methods.
    def plot_force(self):
        """
        Plot the force data with detected peaks and critical force line.
        """
        force_values = self.df['value'].values
        time = self.df['timestamp'].values
        peaks = peakutils.indexes(force_values, thres=0.05, min_dist=700)

        plt.figure(figsize=(10, 6))
        plt.plot(time, force_values, label='Force', color='darkblue', marker='o')

        if len(peaks) >= 3:
            critical_force = np.mean(force_values[peaks[-3:]])
            critical_force_line = [critical_force] * len(time)
            plt.plot(time, critical_force_line, label=f'Critical Force: {critical_force:.2f}', color='crimson')
            for i in peaks:
                if force_values[i] == max(force_values):
                    plt.plot(time[i], force_values[i], 'r.', label=f'Maximum Strength: {force_values[i]:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Force')
        plt.title('Force Data with Peaks')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Assume the CSV file is in the current directory under a 'files' folder.
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '..', 'tests', 'ao_force_20250312_125046.csv')
    evaluator = AllOutTest(file_path)
    results = evaluator.evaluate()
    print("Evaluation Results:")
    print(results)
    # Optionally, show a plot of the force data:
    evaluator.plot_force()
