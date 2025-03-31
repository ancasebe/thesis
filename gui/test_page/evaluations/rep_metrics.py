"""
Module for evaluating repetition-by-repetition metrics from force data,
reusing helper functions from metrics_helpers.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gui.test_page.evaluations.force_metrics import (
    compute_max_strength,
    compute_end_force,
    compute_force_drop,
    compute_work,
    compute_rfd
)


class RepMetrics:
    """
    Evaluates repetition-by-repetition metrics for a force test.

    This class segments the force data into repetitions based on a threshold
    and then uses helper functions from metrics_helpers to compute metrics
    for each repetition.
    """
    def __init__(self, force_df, sampling_rate=100, threshold_ratio=0.1, min_rep_sec=3, max_rep_sec=12):
        """
        Initialize the RepMetrics object.

        Parameters:
            force_df (pd.DataFrame): DataFrame containing force data with columns 'timestamp' and 'value'.
            sampling_rate (int): Sampling rate of the force data in Hz.
            threshold_ratio (float): Fraction of the maximum smoothed force used as a threshold to detect repetitions.
            min_rep_sec (float): Minimum duration (in seconds) for a valid repetition.
            max_rep_sec (float): Maximum duration (in seconds) for a valid repetition.
        """
        self.force_df = force_df
        self.sampling_rate = sampling_rate
        self.threshold_ratio = threshold_ratio
        self.min_rep_samples = int(min_rep_sec * sampling_rate)
        self.max_rep_samples = int(max_rep_sec * sampling_rate)
        self.reps = self._detect_reps()

    def _smooth(self, values, window_size=5):
        """
        Smooth the force data using a simple moving average.

        Parameters:
            values (np.array): Array of force values.
            window_size (int): Number of samples to use for smoothing.

        Returns:
            np.array: Smoothed force values.
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(values, kernel, mode='same')

    def _detect_reps(self):
        """
        Detect repetitions in the force data based on a threshold.

        Returns:
            list of tuples: A list of (start_index, end_index) for each detected repetition.
        """
        force_values = self.force_df['value'].values
        smoothed = self._smooth(force_values, window_size=5)
        threshold = self.threshold_ratio * np.max(smoothed)

        active_intervals = []
        start_idx = None
        for i, val in enumerate(smoothed):
            if val > threshold and start_idx is None:
                start_idx = i
            elif val <= threshold and start_idx is not None:
                active_intervals.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            active_intervals.append((start_idx, len(smoothed)-1))

        # Merge segments that are separated by small gaps.
        min_gap_samples = int(1.0 * self.sampling_rate)
        merged_intervals = []
        for interval in active_intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                prev_start, prev_end = merged_intervals[-1]
                cur_start, cur_end = interval
                if (cur_start - prev_end) < min_gap_samples:
                    merged_intervals[-1] = (prev_start, cur_end)
                else:
                    merged_intervals.append(interval)

        # Filter intervals based on expected repetition duration.
        rep_intervals = []
        for (s, e) in merged_intervals:
            if self.min_rep_samples <= (e - s) <= self.max_rep_samples:
                rep_intervals.append((s, e))
        return rep_intervals

    def compute_rep_metrics(self):
        """
        Compute metrics for each detected repetition by reusing helper functions.

        For each repetition, a DataFrame slice is created and helper functions
        are called to compute maximum strength, average end force, force drop,
        work, and rate of force development.

        Returns:
            list of dict: A list where each dictionary contains metrics for a repetition.
        """
        rep_metrics = []
        # Create a time axis from the timestamp column (assuming timestamp values are in seconds)
        time_axis = self.force_df['timestamp'].values
        for i, (s, e) in enumerate(self.reps, start=1):
            # Create a DataFrame slice for this repetition.
            rep_df = self.force_df.iloc[s:e+1].copy()
            # Optionally, compute a local time axis for the rep.
            rep_df['local_time'] = rep_df['timestamp'] - rep_df['timestamp'].iloc[0]

            # Use the helper functions on the rep slice.
            mvc = compute_max_strength(rep_df)
            # For rep-level average end force, you might use a larger fraction (e.g., last 20% of data)
            end_force = compute_end_force(rep_df, end_portion=0.2)
            force_drop = compute_force_drop(rep_df)
            work = compute_work(rep_df, self.sampling_rate)
            rfd = compute_rfd(rep_df, self.sampling_rate)
            avg_force = rep_df['value'].mean()
            # Compute the pull time (duration of the repetition) in milliseconds.
            duration_ms = (rep_df['timestamp'].iloc[-1] - rep_df['timestamp'].iloc[0]) * 1000

            rep_metrics.append({
                "Rep": i,
                "MVC (kg)": round(mvc, 2) if mvc is not None else None,
                "End Force (kg)": round(end_force, 2) if end_force is not None else None,
                "Force Drop (%)": round(force_drop, 2) if force_drop is not None else None,
                "Avg Force (kg)": round(avg_force, 2) if avg_force is not None else None,
                "Work (kg·s)": round(work, 2) if work is not None else None,
                "Pull Time (ms)": round(duration_ms, 2) if duration_ms is not None else None,
                "RFD (kg/s)": round(rfd, 2) if rfd is not None else None,
            })
        return rep_metrics

    def plot_rep_graphs(self):
        """
        Plot each repetition in a separate subplot.

        This function creates a grid of subplots where each subplot represents one repetition.
        """
        reps = self.reps
        n_reps = len(reps)
        cols = 6
        rows = (n_reps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True, sharey=True)
        axes = axes.flatten()
        time_axis = self.force_df['timestamp'].values
        force_values = self.force_df['value'].values

        for i, (s, e) in enumerate(reps):
            ax = axes[i]
            rep_time = time_axis[s:e+1] - time_axis[s]
            rep_force = force_values[s:e+1]
            ax.plot(rep_time, rep_force, label=f"Rep {i+1}")
            ax.set_title(f"Rep {i+1}")
            ax.legend(fontsize=8)
        # Turn off any extra subplots.
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
