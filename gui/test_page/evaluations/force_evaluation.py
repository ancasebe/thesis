"""
Combined Metrics Module

This module merges the overall force evaluation functions with the repetition‐based metrics.
It defines:
  - Overall helper functions: for computing metrics from the entire force data.
  - Rep-based helper functions: for computing metrics from segmented repetitions.
  - The RepMetrics class: segments the force data into repetitions and computes rep‐based metrics.
  - The CombinedForceMetrics class: computes all desired metrics (overall and rep‐based) in one evaluation.
All computed numerical values are rounded to two decimals.
"""

import os
import numpy as np
import pandas as pd


# ------------------------------
# Overall Helper Functions
# ------------------------------

def compute_max_force(force_df):
    """
    Compute the maximal force (MVC) from the force data.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.

    Returns:
        float: Maximum force value, rounded to two decimals.
    """
    max_force = round(force_df['value'].max(), 2)
    max_force_idx = force_df['value'].idxmax()
    return max_force, max_force_idx

# def compute_end_force(rep_df, max_idx, window=5, drop_multiplier=2):
#     """
#     Compute the end force as the last sample before a sudden drop after maximum force.
#
#     Approach:
#       1. Find the maximum force in the rep.
#       2. Compute the derivative (difference between consecutive samples)
#          for the portion of the rep after the maximum force.
#       3. Define a drop threshold as -drop_multiplier times the standard deviation
#          of these derivative values.
#       4. Identify the first index (after the maximum) where the derivative falls below
#          that threshold and return the force value from the sample immediately before the drop.
#       5. If no significant drop is found, return the last sample.
#
#     Parameters:
#         rep_df (pd.DataFrame): DataFrame for a single rep (should have a 'value' column).
#         window (int): Window size for any optional smoothing (not used in this example).
#         drop_multiplier (float): Multiplier for the standard deviation to define the drop threshold.
#
#     Returns:
#         tuple: (end_force, end_index)
#             - end_force (float): The computed end force, rounded to two decimals.
#             - end_index: The DataFrame index corresponding to that force.
#     """
#     try:
#         forces = rep_df['value'].values
#         indices = rep_df.index
#         if len(forces) < 2:
#             return round(forces[-1], 2), indices[-1]
#
#         # Find the maximum force and its index.
#         # max_idx = forces.argmax()
#         # max_force = forces[max_idx]
#
#         # Consider only the portion after the max force.
#         post_max = forces[max_idx:]
#         if len(post_max) < 2:
#             return round(forces[-1], 2), indices[-1]
#
#         # Compute the first-order derivative for the post-max region.
#         derivatives = np.diff(post_max)
#
#         # Define a drop threshold: e.g., if a drop exceeds drop_multiplier * (std of derivatives)
#         # We use a negative threshold since drops are negative differences.
#         threshold = -drop_multiplier * np.std(derivatives)
#
#         # Look for the first index where the derivative is less than the threshold.
#         drop_idx = None
#         for i, d in enumerate(derivatives):
#             if d < threshold:
#                 drop_idx = i
#                 break
#
#         # if drop_idx is None:
#         #     # If no significant drop is detected, return the last sample.
#         #     end_index = len(forces) - 1
#         #     return round(forces[end_index], 2), indices[end_index]
#         # else:
#             # The drop_idx is relative to post_max.
#             # We take the plateau edge as the sample immediately before the drop.
#         plateau_edge_local = max_idx + drop_idx
#         return round(forces[plateau_edge_local], 2), indices[plateau_edge_local]
#
#     except Exception as e:
#         print(f"Error computing end force (new approach): {e}")
#         return None, None


def compute_end_force(rep_df, derivative_threshold=0.4, plateau_fraction=0.8,
                      min_fraction=0.2, max_force_ratio=0.98, stable_required=2):
    """
    Compute the end force (plateau edge) by scanning backwards from the end of the rep,
    requiring that the difference between a sample and the one three indices earlier is small
    for a consecutive block of samples.

    Approach:
      1. Compute the maximum force and define the plateau region as values >= (plateau_fraction * max_force).
      2. Only consider candidate samples that occur in the last min_fraction of the rep and are below max_force_ratio*max_force.
      3. Scan backward from the rep's end and, for each sample (where i>=3), check if the difference between
         forces[i] and forces[i-3] is <= derivative_threshold.
      4. Group consecutive samples meeting this condition into a "stable block."
      5. Once a stable block of length >= stable_required is found, take the earliest sample of that block
         (i.e. the one closest to the front of the rep) as the plateau edge.
      6. If no such block is found, fall back to using the last sample.

    Parameters:
        rep_df (pd.DataFrame): DataFrame for a single rep (must have a 'value' column).
        derivative_threshold (float): Maximum allowed difference between a sample and the sample three indices earlier.
        plateau_fraction (float): Fraction of max force to define the plateau region.
        min_fraction (float): The end force must be detected within the last min_fraction portion of the rep.
        max_force_ratio (float): The candidate end force must be less than this fraction of the max force.
        stable_required (int): Number of consecutive samples (with the 3-sample gap check) required for stability.

    Returns:
        tuple: (end_force, end_index)
            - end_force (float): The computed end force, rounded to two decimals.
            - end_index: The DataFrame index corresponding to that force.
    """
    try:
        forces = rep_df['value'].values
        indices = rep_df.index
        N = len(forces)
        if N < 3:
            return None, None

        max_force = forces.max()
        min_plateau = plateau_fraction * max_force

        # Define the region toward the end where we expect the end force.
        min_valid_idx = int(N * (1 - min_fraction))

        stable_block = []  # to hold indices that meet the 3-sample difference condition
        candidate_idx = None

        # Scan backwards within the valid region.
        for i in range(N - 1, min_valid_idx - 1, -1):
            # Only consider points in the plateau region and that show some drop from max.
            if min_plateau <= forces[i] < max_force_ratio * max_force:
                # Ensure we have enough earlier samples to compare.
                if i >= 3 and abs(forces[i] - forces[i - 3]) <= derivative_threshold:
                    stable_block.append(i)
                else:
                    # If the current sample fails the condition, check if the stable block so far is long enough.
                    if len(stable_block) >= stable_required:
                        candidate_idx = min(stable_block)  # choose the earliest index in this block
                        break
                    else:
                        stable_block = []  # reset if the condition is broken
            else:
                if len(stable_block) >= stable_required:
                    candidate_idx = min(stable_block)
                    break
                else:
                    stable_block = []

        if candidate_idx is not None and candidate_idx >= min_valid_idx:
            return round(forces[candidate_idx], 2), indices[candidate_idx]
        else:
            # Fallback: if no valid stable plateau edge is found, return the last sample.
            return round(forces[-1], 2), indices[-1]

    except Exception as e:
        print(f"Error computing end force: {e}")
        return None, None



def compute_time_between_max_and_end(max_force_idx, end_force_idx, rep_df):
    """
    Compute the time interval (in milliseconds) between the point of maximal and the end force.

    Parameters:
        max_force_idx (float): Index of determined max force.
        end_force_idx (float): Index of determined end force.
        rep_df (pd.DataFrame): DataFrame with a 'value' and 'timestamp' column.

    Returns:
        float: Time in milliseconds, rounded to two decimals.
    """
    try:
        time_values = rep_df['timestamp'].values
        duration_s = (time_values[end_force_idx] - time_values[max_force_idx])
        return round(duration_s * 1000, 2)
    except Exception as e:
        print(f"Error computing time between max and end force: {e}")
        return None


def compute_force_drop(max_force, end_force):
    """
    Compute the percentage drop in force from the maximal to the end force.

    Parameters:
        max_force (float): Maximal force.
        end_force (float): End force.

    Returns:
        float: Force drop percentage, rounded to two decimals (or None if MVC is zero).
    """
    # mvc = compute_max_force(force_df)
    # end_force = force_df['value'].iloc[-1]
    if max_force > 0:
        return round(100 * (max_force - end_force) / max_force, 2)
    return None


def compute_work(force_df, sampling_rate=100):
    """
    Compute the total work (kg·s) as the area under the force-time curve.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Total work, rounded to two decimals.
    """
    force_values = force_df['value'].values
    return round(np.trapezoid(force_values, dx=1 / sampling_rate), 2)


def compute_rfd(force_df, max_force):
    """
    Compute the time required to go from 20% of MVC to 80% of MVC.

    This function calculates the maximal voluntary contraction (MVC) from the force data,
    then determines the first time at which the force exceeds 20% of MVC and the first time
    after that when the force exceeds 80% of MVC. The difference in time between these two events
    is returned, representing the time needed to develop force from 20% to 80% MVC.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column representing force.
        max_force (float): Maximal force.

    Returns:
        float: Time (in seconds) required to go from 20% MVC to 80% MVC, rounded to two decimals.
               Returns None if either threshold is not reached.
    """
    try:
        force_values = force_df['value'].values
        time_values = force_df['timestamp'].values
        if len(force_values) == 0:
            return None

        threshold20 = 0.2 * max_force
        threshold80 = 0.8 * max_force

        idx_20 = None
        idx_80 = None
        # Find the first index where force exceeds or equals 20% MVC.
        for i, val in enumerate(force_values):
            if idx_20 is None and val >= threshold20:
                idx_20 = i
            # After finding idx_20, look for the first index where force exceeds or equals 80% MVC.
            if idx_20 is not None and val >= threshold80:
                idx_80 = i
                break

        if idx_20 is None or idx_80 is None:
            return None

        # Compute the time difference in seconds.
        rfd = (time_values[idx_80] - time_values[idx_20]) * 1000  # to ms
        # Compute normalized rate force development
        rfd_norm = rfd / (threshold80 - threshold20)
        return round(rfd, 2), round(rfd_norm, 2)

    except Exception as e:
        print(f"Error computing rfd: {e}")
        return None


def compute_critical_force(rep_metrics, peak_window=3):
    """
    Compute the critical force (CF) as the mean of the maximal force (MVC)
    values from the last few repetitions.

    Parameters:
        rep_metrics (list of dict): List of repetition metric dictionaries,
            each containing an "Max Force (kg)" key.
        peak_window (int): Number of last repetitions to average (default is 3).

    Returns:
        float: Critical force, rounded to two decimals, or None if there are insufficient repetitions.
    """
    if len(rep_metrics) < peak_window:
        return None
    # Get the last `peak_window` repetitions.
    last_reps = rep_metrics[-peak_window:]
    mvc_values = [rep["Max Force (kg)"] for rep in last_reps if rep.get("Max Force (kg)") is not None]
    if not mvc_values:
        return None
    return round(sum(mvc_values) / len(mvc_values), 2)


def compute_work_above_cf(force_df, cf, sampling_rate=100):
    """
    Compute the total work performed above the critical force (CF).

    The work above CF is calculated as the area under the curve of (force - CF),
    but only for force values that exceed CF. Values below CF contribute 0 to the area.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column representing force.
        cf (float): Critical force value.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Total work above CF (kg·s), rounded to two decimals.
    """
    force_values = force_df['value'].values
    # Compute the difference between force and critical force, but only when force > cf.
    adjusted_force = np.maximum(force_values - cf, 0)
    # Compute the area under the adjusted curve.
    work_above = np.trapezoid(adjusted_force, dx=1/sampling_rate)
    return round(work_above, 2)


def compute_sum_work_above_cf(force_df, cf, sampling_rate=100):
    """
    Compute the total work performed above the critical force (CF).

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        cf (float): Critical force.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Sum of work above CF, rounded to two decimals.
    """
    force_values = force_df['value'].values
    dt = 1.0 / sampling_rate
    return round(np.trapezoid(np.maximum(force_values - cf, 0), dx=dt), 2)


def compute_avg_work_above_cf(force_df, cf, sampling_rate=100):
    """
    Compute the average work above CF over the duration of the test.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        cf (float): Critical force.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Average work above CF, rounded to two decimals.
    """
    total_time = len(force_df) / sampling_rate
    sum_work = compute_sum_work_above_cf(force_df, cf, sampling_rate)
    return round(sum_work / total_time, 2) if total_time > 0 else None


def compute_average_rep_force(rep_metrics):
    """
    Compute the average repetition force from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "Avg Force (kg)".

    Returns:
        float: Average rep force, rounded to two decimals, or None if no valid values exist.
    """
    forces = [rep["Avg Force (kg)"] for rep in rep_metrics if rep.get("Avg Force (kg)") is not None]
    return round(sum(forces) / len(forces), 2) if forces else None


def compute_average_end_force(rep_metrics):
    """
    Compute the average repetition end force from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "End Force (kg)".

    Returns:
        float: Average end force, rounded to two decimals, or None if no valid values exist.
    """
    forces = [rep["End Force (kg)"] for rep in rep_metrics if rep.get("End Force (kg)") is not None]
    return round(sum(forces) / len(forces), 2) if forces else None


def compute_average_max_end_time(rep_metrics):
    """
    Compute the average repetition end force from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "End Force (kg)".

    Returns:
        float: Average end force, rounded to two decimals, or None if no valid values exist.
    """
    avg_time = [rep["Time Between Max - End Force (ms)"] for rep
                in rep_metrics if rep.get("Time Between Max - End Force (ms)") is not None]
    return round(sum(avg_time) / len(avg_time), 2) if avg_time else None


def compute_average_force_drop(rep_metrics):
    """
    Compute the average repetition force drop from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "Force Drop (%)".

    Returns:
        float: Average rep force drop, rounded to two decimals, or None if no valid values exist.
    """
    force_drop = [rep["Force Drop (%)"] for rep in rep_metrics if rep.get("Force Drop (%)") is not None]
    return round(sum(force_drop) / len(force_drop), 2) if force_drop else None


def compute_average_pulling_time(rep_metrics):
    """
    Compute the average pulling time (ms) from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "Pull Time (ms)".

    Returns:
        float: Average pulling time in milliseconds, rounded to two decimals, or None if not available.
    """
    times = [rep["Pull Time (ms)"] for rep in rep_metrics if rep.get("Pull Time (ms)") is not None]
    return round(sum(times) / len(times), 2) if times else None


def compute_avg_sum_work(rep_metrics):
    """
    Compute the average work from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "Work (kg/s)".

    Returns:
        float: Average work, rounded to two decimals, or None if not available.
    """
    work = [rep["Work (kg/s)"] for rep in rep_metrics if rep.get("Work (kg/s)") is not None]
    avg_work = round(sum(work) / len(work), 2) if work else None
    sum_work = round(sum(work), 2) if work else None
    return avg_work, sum_work


def compute_reps_to_cf(rep_metrics, cf):
    """
    Compute the number of repetitions until the maximal force falls below CF.

    Parameters:
        rep_metrics (list of dict): Each dict should contain the key "Max Force (kg)".
        cf (float): Critical force.

    Returns:
        int: Number of reps until MVC < CF, or the total number of reps if none drop below CF.
    """
    for idx, rep in enumerate(rep_metrics, start=1):
        cf_threshold = cf * 1.1
        if rep.get("Max Force (kg)", 0) < cf_threshold:
            return idx
    return None


def compute_cf_mvc(cf, mvc):
    """
    Compute the ratio of critical force to maximal force as a percentage.

    Parameters:
        cf (float): Critical force.
        mvc (float): Maximal force.

    Returns:
        float: CF/MVC percentage, rounded to two decimals, or None if mvc is zero.
    """
    if mvc > 0:
        return round((cf / mvc) * 100, 2)
    return None


def compute_rfd_subset(rep_metrics, indices):
    """
    Compute the average RFD for a subset of repetitions.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "RFD (kg/s)".
        indices (list of int): Zero-based indices of reps to include.

    Returns:
        float: Average RFD for the selected reps, rounded to two decimals, or None if not available.
    """
    values = [rep["RFD (ms)"] for i, rep in enumerate(rep_metrics) if
              i in indices and rep.get("RFD (ms)") is not None]
    return round(sum(values) / len(values), 2) if values else None


def compute_rfd_subset_normalized(rep_metrics, indices):
    """
    Compute the average normalized RFD for a subset of repetitions.

    Normalized RFD for each repetition is defined as (RFD / MVC).

    Parameters:
        rep_metrics (list of dict): Each dict should contain "RFD (kg/s)" and "Max Force (kg)".
        indices (list of int): Zero-based indices of reps to include.

    Returns:
        float: Average normalized RFD for the selected reps, rounded to two decimals, or None if not available.
    """
    # values = []
    # for i, rep in enumerate(rep_metrics):
    #     if i in indices:
    #         mvc = rep.get("Max Force (kg)")
    #         rfd = rep.get("RFD (ms/kg)")
    #         if mvc and mvc > 0 and rfd is not None:
    #             values.append(rfd / mvc)
    values = [rep["RFD Norm (ms/kg)"] for i, rep in enumerate(rep_metrics) if
              i in indices and rep.get("RFD Norm (ms/kg)") is not None]
    return round(sum(values) / len(values), 2) if values else None


# ------------------------------
# RepMetrics Class
# ------------------------------

class RepMetrics:
    """
    Segments force data into repetitions and computes repetition-based metrics.

    The segmentation is performed by smoothing the force data and detecting when it
    exceeds a threshold defined as a fraction of the maximum smoothed force.
    """

    def __init__(self, force_df, sampling_rate=100, threshold_ratio=0.1, min_rep_sec=5, max_rep_sec=12):
        """
        Initialize the RepMetrics object.

        Parameters:
            force_df (pd.DataFrame): Force data with 'timestamp' and 'value' columns.
            sampling_rate (int): Sampling rate in Hz.
            threshold_ratio (float): Fraction of the maximum smoothed force used as threshold.
            min_rep_sec (float): Minimum valid repetition duration (in seconds).
            max_rep_sec (float): Maximum valid repetition duration (in seconds).
        """
        self.force_df = force_df
        self.sampling_rate = sampling_rate
        self.threshold_ratio = threshold_ratio
        self.min_rep_samples = int(min_rep_sec * sampling_rate)
        self.max_rep_samples = int(max_rep_sec * sampling_rate)
        self.reps = self._detect_reps()

    @staticmethod
    def _smooth(values, window_size=5):
        """
        Smooth the data using a simple moving average.

        Parameters:
            values (np.array): Array of force values.
            window_size (int): Number of samples for the moving average.

        Returns:
            np.array: Smoothed force values.
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(values, kernel, mode='same')

    def _detect_reps(self):
        """
        Detect repetitions in the force data based on a threshold.

        Returns:
            list of tuple: Each tuple is (start_index, end_index) for a detected repetition.
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
            active_intervals.append((start_idx, len(smoothed) - 1))

        # Merge segments with small gaps.
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
        print(rep_intervals)
        return rep_intervals

    def compute_rep_metrics(self):
        """
        Compute metrics for each detected repetition.

        Returns:
            list of dict: Each dictionary contains:
                "Rep": repetition number,
                "Max Force (kg)": maximum force,
                "End Force (kg)": average force at the end of the rep,
                "Force Drop (%)": force drop percentage,
                "Avg Force (kg)": average force,
                "Work (kg·s)": work done during the rep,
                "Pull Time (ms)": duration of the rep in ms,
                "RFD (kg/s)": rate of force development.
            All values (except repetition number) are rounded to two decimals.
        """
        rep_metrics = []
        # time_axis = self.force_df['timestamp'].values
        for i, (s, e) in enumerate(self.reps, start=1):
            # rep_df = self.force_df.iloc[s:e + 1].copy()
            rep_df = self.force_df.iloc[s:e + 1].copy().reset_index(drop=True)
            rep_df['local_time'] = rep_df['timestamp'] - rep_df['timestamp'].iloc[0]
            max_force, max_force_idx = compute_max_force(rep_df)
            end_force, end_force_idx = compute_end_force(rep_df)
            # end_force, end_force_idx = compute_end_force(rep_df, derivative_threshold=0.3, plateau_fraction=0.8)
            print('max idx:', max_force_idx, 'end idx:', end_force_idx)
            force_drop = compute_force_drop(max_force, end_force)
            work = compute_work(rep_df, self.sampling_rate)
            # w_prime = compute_work_above_cf(rep_df, cf, self.sampling_rate)
            rfd, rfd_norm = compute_rfd(rep_df, max_force)
            avg_force = rep_df['value'].mean()
            print('end time:', rep_df['timestamp'].iloc[-1])
            duration_ms = (rep_df['timestamp'].iloc[-1] - rep_df['timestamp'].iloc[0]) * 1000
            max_end_time = compute_time_between_max_and_end(max_force_idx, end_force_idx, rep_df)
            rep_metrics.append({
                "Rep": i,
                "Max Force (kg)": max_force,
                "End Force (kg)": end_force,
                "Force Drop (%)": force_drop,
                "Avg Force (kg)": round(avg_force, 2),
                "Work (kg/s)": work,
                # "Work above CF (kg.s)": w_prime,
                "Pull Time (ms)": round(duration_ms, 2),
                "RFD (ms)": rfd,
                "RFD Norm (ms/kg)": rfd_norm,
                "Time Between Max - End Force (ms)": max_end_time
            })
        return rep_metrics


# ------------------------------
# CombinedForceMetrics Class
# ------------------------------

class ForceMetrics:
    """
    Computes both overall and repetition-based force metrics from a force data file.

    This class loads the force data from a Feather file, performs repetition segmentation once,
    and then computes a set of overall metrics (e.g. maximal force, average end-force, RFD, etc.)
    and rep-based metrics (e.g. average rep force, pulling time, RFD subsets, etc.).
    All computed values are rounded to two decimals.
    """

    def __init__(self, file_path, test_type="ao", sampling_rate=100,
                 threshold_ratio=0.1, min_rep_sec=3, max_rep_sec=12):
        """
        Initialize the CombinedForceMetrics object.

        Parameters:
            file_path (str): Path to the Feather file containing force data.
            test_type (str): Test type identifier (e.g., "ao").
            sampling_rate (int): Sampling rate of the force data (Hz).
            threshold_ratio (float): Fraction of maximum smoothed force used as rep threshold.
            min_rep_sec (float): Minimum repetition duration (in seconds).
            max_rep_sec (float): Maximum repetition duration (in seconds).
        """
        self.rep_results = None
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.df = pd.read_feather(file_path)
        for col in ['timestamp', 'value']:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.test_type = test_type
        self.sampling_rate = sampling_rate
        self.threshold_ratio = threshold_ratio
        self.min_rep_sec = min_rep_sec
        self.max_rep_sec = max_rep_sec

        # Perform repetition segmentation once.
        self.rep_evaluator = RepMetrics(self.df, sampling_rate, threshold_ratio, min_rep_sec, max_rep_sec)
        self.rep_results = self.rep_evaluator.compute_rep_metrics()

    def evaluate(self):
        """
        Compute overall and rep-based metrics from the force data.

        Returns:
            dict: A dictionary containing all computed metrics. Keys include:
                'max_strength', 'avg_end_force', 'time_between_max_end_ms', 'force_drop_pct',
                'work', 'rfd_overall', 'critical_force', 'avg_rep_force', 'avg_pulling_time_ms',
                'reps_to_cf', 'cf_mvc_pct', 'avg_work_above_cf', 'sum_work', 'sum_work_above_cf',
                'rfd_norm_overall', 'rfd_first3', 'rfd_last3', 'rfd_norm_first3', 'rfd_norm_last3',
                'rfd_first6', 'rfd_norm_first6'
            All numerical values are rounded to two decimals.
        """
        avg_work, sum_work = compute_avg_sum_work(self.rep_results)
        results = {}

        # Overall metrics computed from the entire force DataFrame.
        results['max_strength'], _ = compute_max_force(self.df)
        print('Max force:', results['max_strength'])
        results['work'] = avg_work
        results['sum_work'] = sum_work

        # Rep-based metrics computed from the segmented repetitions (self.rep_results).
        results['avg_end_force'] = compute_average_end_force(self.rep_results)
        results['time_between_max_end_ms'] = compute_average_max_end_time(self.rep_results)
        results['force_drop_pct'] = compute_average_force_drop(self.rep_results)
        results['avg_rep_force'] = compute_average_rep_force(self.rep_results)
        results['avg_pulling_time_ms'] = compute_average_pulling_time(self.rep_results)

        # Compute critical force from rep metrics (average MVC of the last few reps)
        results['critical_force'] = compute_critical_force(self.rep_results)
        print('Critical force:', results['critical_force'])
        print('Number of repetitions:', len(self.rep_results))

        # Additional metrics that depend on previously computed values.
        if results['critical_force']:
            results['reps_to_cf'] = compute_reps_to_cf(self.rep_results, results['critical_force'])
            results['cf_mvc_pct'] = compute_cf_mvc(results['critical_force'], results['max_strength'])
            results['avg_work_above_cf'] = compute_avg_work_above_cf(self.df, results['critical_force'], self.sampling_rate)
            results['sum_work_above_cf'] = compute_sum_work_above_cf(self.df, results['critical_force'], self.sampling_rate)

        n_reps = len(self.rep_results)

        results['rfd_overall'] = compute_rfd_subset(self.rep_results, list(range(len(self.rep_results))))
        results['rfd_norm_overall'] = compute_rfd_subset_normalized(self.rep_results, list(range(len(self.rep_results))))

        if n_reps >= 3:
            rfd_first3 = compute_rfd_subset(self.rep_results, list(range(3)))
            rfd_last3 = compute_rfd_subset(self.rep_results, list(range(n_reps - 3, n_reps)))
            rfd_norm_first3 = compute_rfd_subset_normalized(self.rep_results, list(range(3)))
            rfd_norm_last3 = compute_rfd_subset_normalized(self.rep_results, list(range(n_reps - 3, n_reps)))
            results['rfd_first3'] = round(rfd_first3, 2) if rfd_first3 is not None else None
            results['rfd_last3'] = round(rfd_last3, 2) if rfd_last3 is not None else None
            results['rfd_norm_first3'] = round(rfd_norm_first3, 2) if rfd_norm_first3 is not None else None
            results['rfd_norm_last3'] = round(rfd_norm_last3, 2) if rfd_norm_last3 is not None else None

        if n_reps >= 6:
            rfd_first6 = compute_rfd_subset(self.rep_results, list(range(6)))
            rfd_norm_first6 = compute_rfd_subset_normalized(self.rep_results, list(range(6)))
            results['rfd_first6'] = round(rfd_first6, 2) if rfd_first6 is not None else None
            results['rfd_norm_first6'] = round(rfd_norm_first6, 2) if rfd_norm_first6 is not None else None

        return results, self.rep_results
