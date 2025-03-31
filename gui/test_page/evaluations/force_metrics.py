"""
Helper functions for computing force evaluation metrics.

This module provides functions to compute overall test metrics and additional
rep-based metrics. The functions here include both calculations that operate
directly on the entire force DataFrame and those that require rep segmentation.
"""

import numpy as np
# import pandas as pd

# Overall metrics computed from the whole force data


def compute_max_strength(force_df):
    """
    Compute the maximal force (MVC) from the force data.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column representing force.

    Returns:
        float: Maximum force value.
    """
    return force_df['value'].max()


def compute_avg_end_force(force_df, end_portion=0.1):
    """
    Compute the average end-force by averaging the last portion of force values.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        end_portion (float): Fraction of the data at the end to average.

    Returns:
        float: Average end-force.
    """
    n = int(len(force_df) * end_portion)
    if n < 1:
        n = 1
    return force_df['value'].iloc[-n:].mean()


def compute_time_between_max_and_end(force_df, sampling_rate=100):
    """
    Compute the time interval (ms) between the point of maximal force and the end.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Time in milliseconds.
    """
    idx_max = force_df['value'].idxmax()
    duration_s = (len(force_df) - idx_max) / sampling_rate
    return duration_s * 1000


def compute_force_drop(force_df):
    """
    Compute the percentage drop in force from the maximal force to the final force.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.

    Returns:
        float: Percentage force drop.
    """
    mvc = force_df['value'].max()
    end_force = force_df['value'].iloc[-1]
    return 100 * (mvc - end_force) / mvc if mvc > 0 else None


def compute_work(force_df, sampling_rate=100):
    """
    Compute the total work (kg·s) as the area under the force-time curve.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Total work.
    """
    force_values = force_df['value'].values
    return np.trapezoid(force_values, dx=1/sampling_rate)


def compute_rfd(force_df, sampling_rate=100):
    """
    Compute the overall Rate of Force Development (RFD).

    RFD is defined as the maximal force divided by the time to reach it.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: RFD value in kg/s.
    """
    force_values = force_df['value'].values
    idx_peak = np.argmax(force_values)
    time_to_peak = idx_peak / sampling_rate
    mvc = force_values[idx_peak]
    return mvc / time_to_peak if time_to_peak > 0 else None


def compute_critical_force(force_df, peak_window=3):
    """
    Compute the critical force (CF) as the mean of the last few peaks.

    Note: In this simple version, we average the last 'peak_window' force values.
          You may replace this with a peak-detection algorithm if needed.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        peak_window (int): Number of last points to average.

    Returns:
        float: Critical force value.
    """
    if len(force_df) < peak_window:
        return None
    return force_df['value'].iloc[-peak_window:].mean()


# --- Rep-based Metrics (used after segmentation) ---

def compute_average_rep_force(rep_metrics):
    """
    Compute the average repetition force from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict contains an "Avg Force (kg)" key.

    Returns:
        float: Average repetition force.
    """
    forces = [rep["Avg Force (kg)"] for rep in rep_metrics if rep.get("Avg Force (kg)") is not None]
    return sum(forces) / len(forces) if forces else None


def compute_average_pulling_time(rep_metrics):
    """
    Compute the average pulling time (ms) from rep metrics.

    Parameters:
        rep_metrics (list of dict): Each dict contains a "Pull Time (ms)" key.

    Returns:
        float: Average pulling time in milliseconds.
    """
    times = [rep["Pull Time (ms)"] for rep in rep_metrics if rep.get("Pull Time (ms)") is not None]
    return sum(times) / len(times) if times else None


def compute_reps_to_cf(rep_metrics, cf):
    """
    Compute the number of repetitions until the maximal force falls below CF.

    Parameters:
        rep_metrics (list of dict): Each dict contains an "MVC (kg)" key.
        cf (float): Critical force value.

    Returns:
        int: Number of reps until MVC < CF (or total reps if none drop below CF).
    """
    for idx, rep in enumerate(rep_metrics, start=1):
        if rep.get("MVC (kg)", 0) < cf:
            return idx
    return len(rep_metrics)


def compute_cf_mvc(cf, mvc):
    """
    Compute the ratio of CF to MVC expressed as a percentage.

    Parameters:
        cf (float): Critical force.
        mvc (float): Maximal force.

    Returns:
        float: CF/MVC percentage.
    """
    if mvc > 0:
        return (cf / mvc) * 100
    return None


def compute_sum_work_above_cf(force_df, cf, sampling_rate=100):
    """
    Compute the total work performed above the critical force (CF).

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        cf (float): Critical force value.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Sum of work above CF.
    """
    force_values = force_df['value'].values
    dt = 1.0 / sampling_rate
    return np.trapezoid(np.maximum(force_values - cf, 0), dx=dt)


def compute_avg_work_above_cf(force_df, cf, sampling_rate=100):
    """
    Compute the average work above CF (kg·s⁻¹) over the duration of the test.

    Parameters:
        force_df (pd.DataFrame): DataFrame with a 'value' column.
        cf (float): Critical force.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        float: Average work above CF.
    """
    total_time = len(force_df) / sampling_rate
    sum_work = compute_sum_work_above_cf(force_df, cf, sampling_rate)
    return sum_work / total_time if total_time > 0 else None


def compute_rfd_subset(rep_metrics, indices):
    """
    Compute the average RFD for a subset of repetitions.

    Parameters:
        rep_metrics (list of dict): Each dict contains a "RFD (kg/s)" key.
        indices (list of int): Zero-based indices of reps to include.

    Returns:
        float: Average RFD for the selected reps.
    """
    values = [rep_metrics[i]["RFD (kg/s)"] for i in indices if rep_metrics[i].get("RFD (kg/s)") is not None]
    return sum(values) / len(values) if values else None


def compute_rfd_subset_normalized(rep_metrics, indices):
    """
    Compute the average normalized RFD for a subset of repetitions.

    Normalized RFD for each rep is defined as RFD / MVC.

    Parameters:
        rep_metrics (list of dict): Each dict contains "RFD (kg/s)" and "MVC (kg)" keys.
        indices (list of int): Zero-based indices of reps to include.

    Returns:
        float: Average normalized RFD for the selected reps.
    """
    values = []
    for i in indices:
        rep = rep_metrics[i]
        mvc = rep.get("MVC (kg)")
        rfd = rep.get("RFD (kg/s)")
        if mvc and mvc > 0 and rfd is not None:
            values.append(rfd / mvc)
    return sum(values) / len(values) if values else None
