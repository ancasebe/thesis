"""
Module for evaluating overall force test metrics.

The ForceMetrics class reads force data from a Feather file and computes various
metrics using helper functions from metrics_helpers.
"""

import os
import pandas as pd
from gui.test_page.evaluations.force_metrics import (
    compute_max_strength,
    compute_avg_end_force,
    compute_time_between_max_and_end,
    compute_force_drop,
    compute_work,
    compute_rfd,
    compute_critical_force,
    compute_sum_work_above_cf,
    compute_avg_work_above_cf,
    compute_average_rep_force,
    compute_reps_to_cf,
    compute_cf_mvc,
    compute_average_pulling_time,
    compute_rfd_subset,
    compute_rfd_subset_normalized
)
from gui.test_page.evaluations.rep_metrics import RepMetrics  # Assumes you have a RepMetrics class for rep segmentation


class ForceMetrics:
    """
    Evaluates overall force test metrics in a modular way.

    Computes a total of 21 parameters including:
      1. Maximal Force – MVC (KG)
      2. Average End-Force (KG)
      3. Average Time btw Max- and End-Force (ms)
      4. Average Force Drop (%)
      5. Average Rep. Force (KG)
      6. Critical Force – CF (KG)
      7. Repetitions to CF
      8. CF/MVC (%)
      9. Average W (KG.s⁻¹)
      10. Sum W (KG.s⁻¹)
      11. Average W above CF (KG.s⁻¹)
      12. Sum W above CF (KG.s⁻¹)
      13. Average Pulling Time (ms)
      14. Rate of Force Development – RFD (ms) [overall]
      15. RFD first three repetitions (ms)
      16. RFD first six repetitions (ms)
      17. RFD last three repetitions (ms)
      18. RFD normalized to force (ms·KG⁻¹) [overall average from reps]
      19. RFD norm. first three rep. (ms·KG⁻¹)
      20. RFD norm. first six rep. (ms·KG⁻¹)
      21. RFD norm. last three rep. (ms·KG⁻¹)
    """

    def __init__(self, file_path, test_type="ao", sampling_rate=100):
        """
        Initialize the evaluator by loading the force data.

        Parameters:
            file_path (str): Path to the Feather file containing force data.
            test_type (str): The test type (e.g., "ao").
            sampling_rate (int): Sampling rate of the data in Hz.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.df = pd.read_feather(file_path)
        # Verify required columns.
        for col in ['timestamp', 'value']:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.test_type = test_type
        self.sampling_rate = sampling_rate

    def evaluate(self):
        """
        Evaluate the test by computing all 21 desired force metrics.

        Returns:
            dict: A dictionary with all computed metrics.
        """
        results = {'max_strength': compute_max_strength(self.df), 'avg_end_force': compute_avg_end_force(self.df),
                   'time_between_max_end_ms': compute_time_between_max_and_end(self.df, self.sampling_rate),
                   'force_drop_pct': compute_force_drop(self.df), 'work': compute_work(self.df, self.sampling_rate),
                   'rfd_overall': compute_rfd(self.df, self.sampling_rate),
                   'critical_force': compute_critical_force(self.df)}
        # Overall metrics computed directly from the entire force signal.

        # Instantiate rep evaluator to compute rep-level metrics.
        rep_evaluator = RepMetrics(self.df, sampling_rate=self.sampling_rate)
        rep_results = rep_evaluator.compute_rep_metrics()

        # Rep-based metrics.
        results['avg_rep_force'] = compute_average_rep_force(rep_results)
        results['avg_pulling_time_ms'] = compute_average_pulling_time(rep_results)
        results['reps_to_cf'] = compute_reps_to_cf(rep_results, results['critical_force'])
        results['cf_mvc_pct'] = compute_cf_mvc(results['critical_force'], results['max_strength'])
        results['avg_work_above_cf'] = compute_avg_work_above_cf(self.df, results['critical_force'], self.sampling_rate)
        results['sum_work'] = compute_work(self.df, self.sampling_rate)  # Sum W is the same as overall work.
        results['sum_work_above_cf'] = compute_sum_work_above_cf(self.df, results['critical_force'], self.sampling_rate)

        # RFD metrics from reps:
        n_reps = len(rep_results)
        if n_reps >= 1:
            # Overall normalized RFD computed over all reps.
            norm_rfd_all = [rep["RFD (kg/s)"] / rep["MVC (kg)"] for rep in rep_results if
                            rep.get("MVC (kg)", 0) > 0 and rep.get("RFD (kg/s)") is not None]
            results['rfd_norm_overall'] = sum(norm_rfd_all) / len(norm_rfd_all) if norm_rfd_all else None
        else:
            results['rfd_norm_overall'] = None

        # Compute RFD for rep subsets (if enough reps exist)
        if n_reps >= 3:
            results['rfd_first3'] = compute_rfd_subset(rep_results, list(range(3)))
            results['rfd_last3'] = compute_rfd_subset(rep_results, list(range(n_reps - 3, n_reps)))
            results['rfd_norm_first3'] = compute_rfd_subset_normalized(rep_results, list(range(3)))
            results['rfd_norm_last3'] = compute_rfd_subset_normalized(rep_results, list(range(n_reps - 3, n_reps)))
        # else:
        #     results['rfd_first3'] = results['rfd_last3'] = results['rfd_norm_first3'] = results['rfd_norm_last3'] = None

        if n_reps >= 6:
            results['rfd_first6'] = compute_rfd_subset(rep_results, list(range(6)))
            results['rfd_norm_first6'] = compute_rfd_subset_normalized(rep_results, list(range(6)))
        # else:
            # results['rfd_first6'] = results['rfd_norm_first6'] =

        return results
