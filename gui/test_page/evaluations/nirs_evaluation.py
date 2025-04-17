import numpy as np
import pandas as pd


class NIRSEvaluation:
    def __init__(self, nirs_file, smoothing_window=25, baseline_threshold=0.1, recovery_tolerance=1.0):
        """
        Parameters:
            nirs_file (str): Path to the NIRS Feather file.
            smoothing_window (int): Window size for the smoothing (default 25).
            baseline_threshold (float): Ratio threshold for spike correction (default 0.1).
            recovery_tolerance (float): Tolerance (in signal units) for detecting recovery (default 1.0).
        """
        self.nirs_file = nirs_file
        self.smoothing_window = smoothing_window
        self.baseline_threshold = baseline_threshold
        self.recovery_tolerance = recovery_tolerance
        self.results = {}  # to store computed metrics

    @staticmethod
    def smooth_data(data, window_size):
        if window_size < 2:
            return data
        half_win = (window_size - 1) // 2
        data_padded = np.pad(data, (half_win, half_win), mode='edge')
        kernel = np.ones(window_size) / window_size
        convolved = np.convolve(data_padded, kernel, mode='valid')
        return convolved

    @staticmethod
    def correct_baseline_spikes(data, times, test_start_time, threshold_ratio):
        """
        For all indices where times < test_start_time, replace a value with the overall
        baseline average if it deviates by more than threshold_ratio times that average.
        """
        corrected = data.copy()
        baseline_inds = np.where(times < test_start_time)[0]
        if baseline_inds.size == 0:
            return corrected
        baseline_mean = np.mean(data[baseline_inds])
        for i in baseline_inds:
            if baseline_mean != 0 and abs(corrected[i] - baseline_mean) > threshold_ratio * baseline_mean:
                corrected[i] = baseline_mean
        return corrected

    def evaluate(self, start_time, test_start_rel, test_end_rel):
        """
        Performs the NIRS evaluation.

        Parameters:
            start_time (float): The reference time (e.g. the first force timestamp).
            test_start_rel (float): The test start time (relative to start_time) determined via force data.
            test_end_rel (float): The test end time (relative to start_time) determined via force data.

        Returns:
            dict: Contains 'baseline_mean' and 'time_to_recovery'
        """
        try:
            # Load NIRS data
            nirs_df = pd.read_feather(self.nirs_file)
            nirs_df['time'] = nirs_df['time'].astype(float)
            nirs_time_absolute = nirs_df['time']
            # Convert to relative time based on the force start_time
            nirs_time_array = nirs_time_absolute - start_time

            # Get the NIRS signal – try 'smo2' if available; otherwise use 'value'
            if 'smo2' in nirs_df.columns:
                nirs_array = nirs_df['smo2'].values
            else:
                nirs_array = nirs_df['value'].values
            nirs_array = np.clip(nirs_array, 0, None)

            # Correct baseline spikes only in the baseline region (times < test_start_rel)
            nirs_array = self.correct_baseline_spikes(nirs_array, nirs_time_array, test_start_rel, self.baseline_threshold)

            # Smooth the corrected NIRS data
            nirs_array = self.smooth_data(nirs_array, self.smoothing_window)

            # Compute the baseline mean: average NIRS value before test start (relative time < test_start_rel)
            baseline_inds = np.where(nirs_time_array < test_start_rel)[0]
            if baseline_inds.size > 0:
                baseline_mean = np.mean(nirs_array[baseline_inds])
                baseline_mean = round(baseline_mean, 2)
            else:
                baseline_mean = np.nan

            # Compute time to recovery: starting at test_end_rel, find the first time point where
            # the NIRS value is within recovery_tolerance of the baseline_mean.
            recovery_inds = np.where(nirs_time_array >= test_end_rel)[0]
            time_to_recovery = None
            for idx in recovery_inds:
                if abs(nirs_array[idx] - baseline_mean) <= self.recovery_tolerance:
                    time_to_recovery = round((nirs_time_array[idx] - test_end_rel), 2)
                    break
            self.results = {
                'baseline_mean': baseline_mean,
                'time_to_recovery': time_to_recovery
            }
            return self.results
        except FileNotFoundError:
            print('FileNotFoundError:', self.nirs_file)
