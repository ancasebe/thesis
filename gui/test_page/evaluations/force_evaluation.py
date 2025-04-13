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
        tuple: (maximum force value, index of maximum), rounded to two decimals.
    """
    try:
        max_val = force_df['value'].max()
        max_idx = force_df['value'].idxmax()
        return round(max_val, 2), max_idx
    except Exception as e:
        print(f"Error computing max force: {e}")
        return None, None


def compute_end_force(rep_df, derivative_threshold=0.2, min_fraction=0.3,
                      stable_required=2, start_offset=10, limit_ratio=0.3):
    """
    Compute the end force (plateau edge) by scanning backward from a point near the end
    of the rep. The function starts scanning from 'start_offset' samples from the end and
    moves backward until it finds at least 'stable_required' consecutive samples where the
    difference between the current sample and the sample three indices earlier is <= derivative_threshold.

    Additionally, a candidate is only accepted if its force is at least limit_ratio * max_force,
    otherwise the search continues.

    Parameters:
        rep_df (pd.DataFrame): DataFrame for a single rep (must have a 'value' column).
        derivative_threshold (float): Maximum allowed difference between a sample and the sample 3 indices earlier.
        min_fraction (float): Only search in the last min_fraction portion of the rep.
        stable_required (int): Number of consecutive stable samples required.
        start_offset (int): Number of samples from the end at which to start the backward scan.
        limit_ratio (float): Minimum acceptable candidate force is limit_ratio * max_force.

    Returns:
        tuple: (end_force, index), where end_force is rounded to two decimals and index is the DataFrame index;
               returns (None, None) if no stable candidate meeting the criteria is found.
    """
    try:
        forces = rep_df['value'].values
        indices = rep_df.index
        N = len(forces)
        if N < 3:
            return None, None

        max_force = forces.max()
        # Define the lower bound for scanning: only search in the last min_fraction of the rep.
        min_valid_idx = int(N * (1 - min_fraction))
        # Set starting index: start_offset samples from the end (or the very last sample if that would be too low)
        start_index = N - start_offset if (N - start_offset) >= min_valid_idx else N - 1

        consecutive = 0
        candidate_idx = None

        # Scan backward from start_index down to min_valid_idx.
        for i in range(start_index, min_valid_idx - 1, -1):
            # Ensure we can compare with the sample three indices earlier.
            if i < 3:
                break
            # Check the condition based on the sample three indices earlier.
            if abs(forces[i] - forces[i - 3]) <= derivative_threshold:
                consecutive += 1
                candidate_idx = i
                if consecutive >= stable_required:
                    # Only accept candidate if it meets the minimum force limit.
                    if forces[candidate_idx] >= limit_ratio * max_force:
                        break
                    else:
                        # Candidate's force is too low; reset and continue searching.
                        consecutive = 0
                        candidate_idx = None
            else:
                consecutive = 0
                candidate_idx = None

        if candidate_idx is not None and consecutive >= stable_required:
            return round(forces[candidate_idx], 2), indices[candidate_idx]
        else:
            return None, None

    except Exception as e:
        print(f"Error computing end force: {e}")
        return None, None


def compute_time_between_max_and_end(max_force_idx, end_force_idx, rep_df):
    """
    Compute the time interval (in milliseconds) between the point of maximal and the end force.

    Parameters:
        max_force_idx (float): Index of determined max force.
        end_force_idx (float): Index of determined end force.
        rep_df (pd.DataFrame): DataFrame with 'time' column.

    Returns:
        float: Time in milliseconds, rounded to two decimals.
    """
    try:
        time_values = rep_df['time'].values
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
        float: Force drop percentage, rounded to two decimals (or None if max_force is zero).
    """
    try:
        if max_force > 0:
            return round(100 * (max_force - end_force) / max_force, 2)
        return None
    except Exception as e:
        print(f"Error computing force drop: {e}")
        return None


def compute_work(force_df):
    """
    Compute the total work (kg·s) as the area under the force-time curve using actual times.

    Parameters:
        force_df (pd.DataFrame): DataFrame with 'value' and 'time' columns.

    Returns:
        float: Total work, rounded to two decimals.
    """
    try:
        force_values = force_df['value'].values
        time_values = force_df['time'].values
        work = np.trapezoid(force_values, x=time_values)
        return round(work, 2)
    except Exception as e:
        print(f"Error computing work: {e}")
        return None


def compute_rfd(force_df, max_force):
    """
    Compute the time required to go from 20% of MVC to 80% of MVC.

    Parameters:
        force_df (pd.DataFrame): DataFrame with 'value' and 'time' columns.
        max_force (float): Maximal force.

    Returns:
        tuple: (Time in ms to go from 20% MVC to 80% MVC, normalized RFD), both rounded to two decimals,
               or None if thresholds are not reached.
    """
    try:
        force_values = force_df['value'].values
        time_values = force_df['time'].values
        if len(force_values) == 0:
            return None

        threshold20 = 0.2 * max_force
        threshold80 = 0.8 * max_force

        idx_20 = None
        idx_80 = None
        for i, val in enumerate(force_values):
            if idx_20 is None and val >= threshold20:
                idx_20 = i
            if idx_20 is not None and val >= threshold80:
                idx_80 = i
                break

        if idx_20 is None or idx_80 is None:
            return None

        rfd = (time_values[idx_80] - time_values[idx_20]) * 1000  # in ms
        rfd_norm = rfd / (threshold80 - threshold20)
        return round(rfd, 2), round(rfd_norm, 2)
    except Exception as e:
        print(f"Error computing rfd: {e}")
        return None


def compute_critical_force(rep_metrics, peak_window=3):
    """
    Compute the critical force (CF) as the mean of the maximal force (MVC) values from the last few repetitions.

    Parameters:
        rep_metrics (list of dict): List of repetition metric dictionaries.
        peak_window (int): Number of last repetitions to average.

    Returns:
        float: Critical force, rounded to two decimals, or None if insufficient data.
    """
    try:
        if len(rep_metrics) < peak_window:
            return None
        last_reps = rep_metrics[-peak_window:]
        mvc_values = [rep["Max Force (kg)"] for rep in last_reps if rep.get("Max Force (kg)") is not None]
        if not mvc_values:
            return None
        return round(sum(mvc_values) / len(mvc_values), 2)
    except Exception as e:
        print(f"Error computing critical force: {e}")
        return None


def compute_work_above_cf(force_df, cf):
    """
    Compute the total work performed above the critical force (CF).

    Parameters:
        force_df (pd.DataFrame): DataFrame with 'value' and 'time' columns.
        cf (float): Critical force value.

    Returns:
        float: Total work above CF, rounded to two decimals.
    """
    try:
        force_values = force_df['value'].values
        times = force_df['time'].values
        adjusted_force = np.maximum(force_values - cf, 0)
        work_above = np.trapezoid(adjusted_force, x=times)
        return round(work_above, 2)
    except Exception as e:
        print(f"Error computing work above CF: {e}")
        return None

#
# def compute_sum_work_above_cf(force_df, cf):
#     """
#     Compute the total work performed above the critical force (CF) as the sum.
#
#     Parameters:
#         force_df (pd.DataFrame): DataFrame with 'value' and 'time' columns.
#         cf (float): Critical force.
#
#     Returns:
#         float: Sum of work above CF, rounded to two decimals.
#     """
#     try:
#         force_values = force_df['value'].values
#         times = force_df['time'].values
#         work_above = np.trapezoid(np.maximum(force_values - cf, 0), x=times)
#         return round(work_above, 2)
#     except Exception as e:
#         print(f"Error computing sum work above CF: {e}")
#         return None


def compute_avg_work_above_cf(rep_metrics):
    """
    Compute the average work above CF over the test duration using rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Work Above CF (kg·s)".

    Returns:
        float: Average work above CF, rounded to two decimals, or None.
    """
    try:
        works = [rep["Work Above CF (kg·s)"] for rep in rep_metrics if rep.get("Work Above CF (kg·s)") is not None]
        return round(sum(works) / len(works), 2) if works else None
    except Exception as e:
        print(f"Error computing average work above CF: {e}")
        return None


def compute_average_rep_force(rep_metrics):
    """
    Compute the average repetition force from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Avg Force (kg)".

    Returns:
        float: Average rep force, rounded to two decimals, or None.
    """
    try:
        forces = [rep["Avg Force (kg)"] for rep in rep_metrics if rep.get("Avg Force (kg)") is not None]
        return round(sum(forces) / len(forces), 2) if forces else None
    except Exception as e:
        print(f"Error computing average rep force: {e}")
        return None


def compute_average_end_force(rep_metrics):
    """
    Compute the average repetition end force from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "End Force (kg)".

    Returns:
        float: Average end force, rounded to two decimals, or None.
    """
    try:
        forces = [rep["End Force (kg)"] for rep in rep_metrics if rep.get("End Force (kg)") is not None]
        return round(sum(forces) / len(forces), 2) if forces else None
    except Exception as e:
        print(f"Error computing average end force: {e}")
        return None


def compute_average_max_end_time(rep_metrics):
    """
    Compute the average time between max force and end force from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Time Between Max - End Force (ms)".

    Returns:
        float: Average time, rounded to two decimals, or None.
    """
    try:
        avg_time = [rep["Time Between Max - End Force (ms)"] for rep in rep_metrics if rep.get("Time Between Max - End Force (ms)") is not None]
        return round(sum(avg_time) / len(avg_time), 2) if avg_time else None
    except Exception as e:
        print(f"Error computing average max-end time: {e}")
        return None


def compute_average_force_drop(rep_metrics):
    """
    Compute the average repetition force drop from a list of rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Force Drop (%)".

    Returns:
        float: Average force drop percentage, rounded to two decimals, or None.
    """
    try:
        force_drop = [rep["Force Drop (%)"] for rep in rep_metrics if rep.get("Force Drop (%)") is not None]
        return round(sum(force_drop) / len(force_drop), 2) if force_drop else None
    except Exception as e:
        print(f"Error computing average force drop: {e}")
        return None


def compute_average_pulling_time(rep_metrics):
    """
    Compute the average pulling time (ms) from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Pull Time (ms)".

    Returns:
        float: Average pulling time in ms, rounded to two decimals, or None.
    """
    try:
        times = [rep["Pull Time (ms)"] for rep in rep_metrics if rep.get("Pull Time (ms)") is not None]
        return round(sum(times) / len(times), 2) if times else None
    except Exception as e:
        print(f"Error computing average pulling time: {e}")
        return None


def compute_avg_sum_work(rep_metrics):
    """
    Compute the average and sum of work from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Work (kg/s)".

    Returns:
        tuple: (average work, sum work), both rounded to two decimals, or (None, None).
    """
    try:
        work = [rep["Work (kg/s)"] for rep in rep_metrics if rep.get("Work (kg/s)") is not None]
        avg_work = round(sum(work) / len(work), 2) if work else None
        sum_work = round(sum(work), 2) if work else None
        return avg_work, sum_work
    except Exception as e:
        print(f"Error computing average sum work: {e}")
        return None, None


def compute_reps_to_cf(rep_metrics, cf):
    """
    Compute the number of repetitions until the maximal force falls below CF.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "Max Force (kg)".
        cf (float): Critical force.

    Returns:
        int: Number of reps until MVC < CF, or None.
    """
    try:
        for idx, rep in enumerate(rep_metrics, start=1):
            cf_threshold = cf * 1.1
            if rep.get("Max Force (kg)", 0) < cf_threshold:
                return idx
        return None
    except Exception as e:
        print(f"Error computing reps to CF: {e}")
        return None


def compute_cf_mvc(cf, mvc):
    """
    Compute the ratio of critical force to maximal force as a percentage.

    Parameters:
        cf (float): Critical force.
        mvc (float): Maximal force.

    Returns:
        float: CF/MVC percentage, rounded to two decimals, or None.
    """
    try:
        if mvc > 0:
            return round((cf / mvc) * 100, 2)
        return None
    except Exception as e:
        print(f"Error computing CF/MVC ratio: {e}")
        return None


def compute_rfd_subset(rep_metrics, indices):
    """
    Compute the average RFD for a subset of repetitions.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "RFD (ms)".
        indices (list of int): Indices of reps to include.

    Returns:
        float: Average RFD, rounded to two decimals, or None.
    """
    try:
        values = [rep["RFD (ms)"] for i, rep in enumerate(rep_metrics) if i in indices and rep.get("RFD (ms)") is not None]
        return round(sum(values) / len(values), 2) if values else None
    except Exception as e:
        print(f"Error computing RFD subset: {e}")
        return None


def compute_rfd_subset_normalized(rep_metrics, indices):
    """
    Compute the average normalized RFD for a subset of repetitions.

    Parameters:
        rep_metrics (list of dict): Each dict should contain "RFD Norm (ms/kg)".
        indices (list of int): Indices of reps to include.

    Returns:
        float: Average normalized RFD, rounded to two decimals, or None.
    """
    try:
        values = [rep["RFD Norm (ms/kg)"] for i, rep in enumerate(rep_metrics) if i in indices and rep.get("RFD Norm (ms/kg)") is not None]
        return round(sum(values) / len(values), 2) if values else None
    except Exception as e:
        print(f"Error computing normalized RFD subset: {e}")
        return None


# ------------------------------
# RepMetrics Class
# ------------------------------

class RepMetrics:
    """
    Segments force data into repetitions and computes repetition-based metrics.

    The segmentation is performed by smoothing the force data and detecting when it
    exceeds a threshold defined as a fraction of the maximum smoothed force.
    """

    def __init__(self, force_df, threshold_ratio=0.1, min_rep_sec=5, max_rep_sec=12):
        """
        Initialize the RepMetrics object.

        Parameters:
            force_df (pd.DataFrame): Force data with 'time' and 'value' columns.
            threshold_ratio (float): Fraction of the maximum smoothed force used as threshold.
            min_rep_sec (float): Minimum valid repetition duration (in seconds).
            max_rep_sec (float): Maximum valid repetition duration (in seconds).
        """
        # self.force_df = force_df
        # self.sampling_rate = sampling_rate
        # self.threshold_ratio = threshold_ratio
        # self.min_rep_samples = int(min_rep_sec * sampling_rate)
        # self.max_rep_samples = int(max_rep_sec * sampling_rate)
        # self.reps = self._detect_reps()

        self.force_df = force_df
        # self.sampling_rate = sampling_rate
        self.threshold_ratio = threshold_ratio
        self.min_rep_sec = min_rep_sec
        self.max_rep_sec = max_rep_sec
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
        Detect repetitions in the force data using actual times.

        Process:
          1. Smooth the force values.
          2. Determine a threshold as a fraction of the maximum smoothed value.
          3. Detect contiguous intervals where the smoothed force exceeds the threshold.
          4. Merge intervals that are separated by a gap shorter than a constant (e.g. 1.0 sec).
          5. For each merged interval, compute the duration using the actual times.
             Only retain intervals whose duration is between min_rep_sec and max_rep_sec.

        Returns:
            list of tuple: Each tuple is (start_index, end_index) for a detected repetition.
        """
        try:
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

            # Merge segments with small gaps based on time difference.
            min_gap_sec = 1.0  # Merge segments separated by less than 1.0 second.
            merged_intervals = []
            times = self.force_df['time'].values
            for interval in active_intervals:
                if not merged_intervals:
                    merged_intervals.append(interval)
                else:
                    prev_start, prev_end = merged_intervals[-1]
                    cur_start, cur_end = interval
                    gap = times[cur_start] - times[prev_end]
                    if gap < min_gap_sec:
                        merged_intervals[-1] = (prev_start, cur_end)
                    else:
                        merged_intervals.append(interval)

            # Filter intervals based on actual repetition duration.
            rep_intervals = []
            for (s, e) in merged_intervals:
                duration = times[e] - times[s]
                if self.min_rep_sec <= duration <= self.max_rep_sec:
                    rep_intervals.append((s, e))
            print("Detected repetition intervals (using time):", rep_intervals)
            return rep_intervals
        except Exception as e:
            print(f"Error detecting repetitions: {e}")
            return []

    # def _detect_reps(self):
    #     """
    #     Detect repetitions in the force data based on a threshold.
    #
    #     Returns:
    #         list of tuple: Each tuple is (start_index, end_index) for a detected repetition.
    #     """
    #     try:
    #         force_values = self.force_df['value'].values
    #         smoothed = self._smooth(force_values, window_size=5)
    #         threshold = self.threshold_ratio * np.max(smoothed)
    #         active_intervals = []
    #         start_idx = None
    #         for i, val in enumerate(smoothed):
    #             if val > threshold and start_idx is None:
    #                 start_idx = i
    #             elif val <= threshold and start_idx is not None:
    #                 active_intervals.append((start_idx, i))
    #                 start_idx = None
    #         if start_idx is not None:
    #             active_intervals.append((start_idx, len(smoothed) - 1))
    #
    #         # Merge segments with small gaps.
    #         min_gap_samples = int(1.0 * self.sampling_rate)
    #         merged_intervals = []
    #         for interval in active_intervals:
    #             if not merged_intervals:
    #                 merged_intervals.append(interval)
    #             else:
    #                 prev_start, prev_end = merged_intervals[-1]
    #                 cur_start, cur_end = interval
    #                 if (cur_start - prev_end) < min_gap_samples:
    #                     merged_intervals[-1] = (prev_start, cur_end)
    #                 else:
    #                     merged_intervals.append(interval)
    #
    #         # Filter intervals based on expected repetition duration.
    #         rep_intervals = []
    #         for (s, e) in merged_intervals:
    #             if self.min_rep_samples <= (e - s) <= self.max_rep_samples:
    #                 rep_intervals.append((s, e))
    #         print(rep_intervals)
    #         return rep_intervals
    #     except Exception as e:
    #         print(f"Error detecting repetitions: {e}")
    #         return []

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
        # time_axis = self.force_df['time'].values
        print("self.reps: ", len(self.reps))
        try:
            for i, (s, e) in enumerate(self.reps, start=1):
                print('repetition n.', i, 'indexes: ', (s, e))
                rep_df = self.force_df.iloc[s:e + 1].copy().reset_index(drop=True)
                rep_df['local_time'] = rep_df['time'] - rep_df['time'].iloc[0]
                max_force, max_force_idx = compute_max_force(rep_df)
                end_force, end_force_idx = compute_end_force(rep_df)
                # end_force, end_force_idx = compute_end_force(rep_df, derivative_threshold=0.3, plateau_fraction=0.8)
                force_drop = compute_force_drop(max_force, end_force)
                work = compute_work(rep_df)
                rfd, rfd_norm = compute_rfd(rep_df, max_force)
                avg_force = rep_df['value'].mean()
                duration_ms = (rep_df['time'].iloc[-1] - rep_df['time'].iloc[0]) * 1000
                max_end_time = compute_time_between_max_and_end(max_force_idx, end_force_idx, rep_df)
                rep_metrics.append({
                    "Rep": int(i),
                    "Max Force (kg)": max_force,
                    "End Force (kg)": end_force,
                    "Force Drop (%)": force_drop,
                    "Avg Force (kg)": round(avg_force, 2),
                    "Pull Time (ms)": round(duration_ms, 2),
                    "Time Between Max - End Force (ms)": max_end_time,
                    "RFD (ms)": rfd,
                    "RFD Norm (ms/kg)": rfd_norm,
                    "Work (kg/s)": work
                })

            return rep_metrics
        except Exception as e:
            print(f"Error computing rep metrics {e}")


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

    def __init__(self, file_path, test_type="ao", threshold_ratio=0.1, min_rep_sec=3, max_rep_sec=12):
        """
        Initialize the CombinedForceMetrics object.

        Parameters:
            file_path (str): Path to the Feather file containing force data.
            test_type (str): Test type identifier (e.g., "ao")..
            threshold_ratio (float): Fraction of maximum smoothed force used as rep threshold.
            min_rep_sec (float): Minimum repetition duration (in seconds).
            max_rep_sec (float): Maximum repetition duration (in seconds).
        """
        self.rep_metrics = None
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.df = pd.read_feather(file_path)
        for col in ['time', 'value']:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.test_type = test_type
        self.threshold_ratio = threshold_ratio
        if test_type == "mvc":
            min_rep_sec = 2
            max_rep_sec = 8
        elif test_type == "sit":
            min_rep_sec = 2
            max_rep_sec = 300
        elif test_type == "iit":
            min_rep_sec = 4
            max_rep_sec = 12
        else:
            min_rep_sec = 3
            max_rep_sec = 11

        # Perform repetition segmentation once.
        self.rep_evaluator = RepMetrics(force_df=self.df,
                                        threshold_ratio=threshold_ratio,
                                        min_rep_sec=min_rep_sec,
                                        max_rep_sec=max_rep_sec)
        self.rep_metrics = self.rep_evaluator.compute_rep_metrics()

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
        try:
            avg_work, sum_work = compute_avg_sum_work(self.rep_metrics)
            results = {}

            # Overall metrics computed from the entire force DataFrame.
            results['max_strength'], _ = compute_max_force(self.df)
            print('Max force:', results['max_strength'])
            results['avg_pulling_time_ms'] = compute_average_pulling_time(self.rep_metrics)
            results['sum_work'] = sum_work
            n_reps = len(self.rep_metrics)

            results['rfd_overall'] = compute_rfd_subset(self.rep_metrics, list(range(len(self.rep_metrics))))
            results['rfd_norm_overall'] = compute_rfd_subset_normalized(self.rep_metrics, list(range(len(self.rep_metrics))))

            if self.test_type not in ['mvc', 'sit', 'dh']:
                results['work'] = avg_work
                if n_reps >= 3:
                    rfd_first3 = compute_rfd_subset(self.rep_metrics, list(range(3)))
                    rfd_last3 = compute_rfd_subset(self.rep_metrics, list(range(n_reps - 3, n_reps)))
                    rfd_norm_first3 = compute_rfd_subset_normalized(self.rep_metrics, list(range(3)))
                    rfd_norm_last3 = compute_rfd_subset_normalized(self.rep_metrics, list(range(n_reps - 3, n_reps)))
                    results['rfd_first3'] = round(rfd_first3, 2) if rfd_first3 is not None else None
                    results['rfd_last3'] = round(rfd_last3, 2) if rfd_last3 is not None else None
                    results['rfd_norm_first3'] = round(rfd_norm_first3, 2) if rfd_norm_first3 is not None else None
                    results['rfd_norm_last3'] = round(rfd_norm_last3, 2) if rfd_norm_last3 is not None else None

                if n_reps >= 6:
                    rfd_first6 = compute_rfd_subset(self.rep_metrics, list(range(6)))
                    rfd_norm_first6 = compute_rfd_subset_normalized(self.rep_metrics, list(range(6)))
                    results['rfd_first6'] = round(rfd_first6, 2) if rfd_first6 is not None else None
                    results['rfd_norm_first6'] = round(rfd_norm_first6, 2) if rfd_norm_first6 is not None else None

            if self.test_type == 'ao':
                # Rep-based metrics computed from the segmented repetitions (self.rep_results).
                results['avg_end_force'] = compute_average_end_force(self.rep_metrics)
                results['time_between_max_end_ms'] = compute_average_max_end_time(self.rep_metrics)
                results['force_drop_pct'] = compute_average_force_drop(self.rep_metrics)
                results['avg_rep_force'] = compute_average_rep_force(self.rep_metrics)

                # Compute critical force from rep metrics (average MVC of the last few reps)
                results['critical_force'] = compute_critical_force(self.rep_metrics)
                print('Critical force:', results['critical_force'])
                print('Number of repetitions:', len(self.rep_metrics))

                # Additional metrics that depend on previously computed values.
                if results['critical_force']:
                    results['reps_to_cf'] = compute_reps_to_cf(self.rep_metrics, results['critical_force'])
                    results['cf_mvc_pct'] = compute_cf_mvc(results['critical_force'], results['max_strength'])
                    # Loop over the rep intervals (they are in the same order as rep_metrics)
                    for i, (s, e) in enumerate(self.rep_evaluator.reps):
                        # Extract the rep data and reset its index for local integration.
                        rep_df = self.df.iloc[s:e + 1].copy().reset_index(drop=True)
                        # Compute the work above CF for this rep using the times from rep_df.
                        rep_work_above_cf = compute_work_above_cf(rep_df, results['critical_force'])
                        # Add the computed work above CF to the corresponding rep metric.
                        self.rep_metrics[i]['Work Above CF (kg/s)'] = rep_work_above_cf

                    # results['avg_work_above_cf'] = compute_avg_work_above_cf(self.rep_metrics)
                    results['sum_work_above_cf'] = compute_work_above_cf(self.df, results['critical_force'])

                # n_reps = len(self.rep_metrics)
                #
                # results['rfd_overall'] = compute_rfd_subset(self.rep_metrics, list(range(len(self.rep_metrics))))
                # results['rfd_norm_overall'] = compute_rfd_subset_normalized(self.rep_metrics, list(range(len(self.rep_metrics))))

            return results, self.rep_metrics

        except Exception as e:
            print("Error evaluating force metrics:", e)
            return None, None
