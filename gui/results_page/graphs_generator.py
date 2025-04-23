import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager


def smooth_data(data, window_size=11):
    if window_size < 2:
        return data

    half_win = (window_size - 1) // 2
    # Pad the data on both ends using the edge values
    data_padded = np.pad(data, (half_win, half_win), mode='edge')
    kernel = np.ones(window_size) / window_size

    # 'valid' mode ensures the output has the same size as the original data
    # after we manually pad; if you do 'same', it tries zero-padding internally.
    convolved = np.convolve(data_padded, kernel, mode='valid')
    return convolved


def create_combined_figure(force_file, nirs_file, test_metrics):
    """
    Creates a combined figure with force data (left y-axis) and NIRS (SmO2) data (right y-axis).

    Processing steps:
      1. Load the force data and identify the test interval using the original (absolute) timestamps.
         The test interval is defined as the time range during which the force is above 10% of its maximum.
      2. Convert the timestamps for both force and NIRS data to relative times by subtracting the start time.
      3. Replace any negative values in force and NIRS data with 0.
      4. Smooth the NIRS data to reduce noise.
      5. Shade the baseline (before test start) and recovery (after test end) areas on the NIRS plot,
         using the relative times computed from the absolute test start and end.
      6. Fix the NIRS y-axis limits from 0 to 100 and adjust the layout so the title does not overlap.

    Parameters:
        force_file (str): Path to the force data (feather file) with a 'time' column and 'value' column.
        nirs_file (str): Path to the NIRS data (feather file) with a 'time' column and optionally 'smo2'.
        test_metrics (dict): Dictionary containing test metrics (max_strength, critical_force).

    Returns:
        matplotlib.figure.Figure: The generated combined figure.
    """

    def correct_baseline_spikes(data, times, test_start_time, threshold_ratio=0.1):
        """
        Corrects outlier spikes in the baseline region of the NIRS data.
        The baseline region is defined by time values less than test_start_time.
        For all indices in that region, if a sample deviates from the average of all baseline samples
        by more than threshold_ratio * baseline_average, it is replaced by the baseline average.
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

    # --- Load and prepare Force data (absolute timestamps) ---
    force_df = pd.read_feather(force_file)
    force_df['time'] = force_df['time'].astype(float)
    # Do not subtract the start time immediately.
    force_absolute_time = force_df['time']  # Absolute time values from the force file
    force_array = force_df['value'].values
    force_array = np.clip(force_array, 0, None)  # Set negative values to 0
    force_array = smooth_data(force_array, window_size=11)

    # --- Identify test start and end times from force data (absolute timestamps) ---
    # max_force = force_array.max()
    max_force = test_metrics.get("max_strength")
    threshold = 0.1 * max_force
    above_threshold_indices = np.where(force_array >= threshold)[0]
    if above_threshold_indices.size > 0:
        # Use the absolute times directly from the force file
        test_start_abs = force_absolute_time.iloc[above_threshold_indices[0]]
        test_end_abs = force_absolute_time.iloc[above_threshold_indices[-1]]
    else:
        test_start_abs = force_absolute_time.iloc[0]
        test_end_abs = force_absolute_time.iloc[-1]

    # --- Convert force time values to relative times for plotting ---
    start_time = force_absolute_time.iloc[0]
    time_array = force_absolute_time - start_time

    # Convert the test boundaries to relative time
    test_start_rel = test_start_abs - start_time
    test_end_rel = test_end_abs - start_time

    # --- Load and prepare NIRS data ---
    nirs_df = pd.read_feather(nirs_file)
    nirs_df['time'] = nirs_df['time'].astype(float)
    # Use the same reference (first force time) to create relative time for NIRS
    nirs_time_absolute = nirs_df['time']
    nirs_time_array = nirs_time_absolute - start_time
    if 'smo2' in nirs_df.columns:
        nirs_array = nirs_df['smo2'].values
    else:
        nirs_array = nirs_df['value'].values
    nirs_array = np.clip(nirs_array, 0, None)
    # --- Correct baseline spikes in NIRS data (only before test start) ---
    test_start_nirs_rel = force_absolute_time.iloc[above_threshold_indices[0] + 100] - start_time
    nirs_array = correct_baseline_spikes(nirs_array, nirs_time_array, test_start_nirs_rel, threshold_ratio=0.1)
    # --- Smooth the (corrected) NIRS data ---
    nirs_array = smooth_data(nirs_array, window_size=25)

    # --- Create the figure with two y-axes ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the force data on the left y-axis
    ax1.plot(time_array, force_array, label='Force [kg]', color='darkblue')
    max_idx = force_array.argmax()
    ax1.plot(time_array.iloc[max_idx] if hasattr(time_array, 'iloc') else time_array[max_idx],
             max_force, 'r.', label=f'Maximum strength: {max_force:.2f}')
    # ax1.text(time_array.iloc[max_idx] if hasattr(time_array, 'iloc') else time_array[max_idx],
    #          max_force, f'{max_force:.2f}', fontsize=10, ha='left', va='bottom')
    ax1.set_xlabel('Time [s]', fontsize=14)
    ax1.set_ylabel('Force [kg]', fontsize=14, color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.set_ylim(0, max_force + 5)
    ax1.grid(True)

    critical_force = test_metrics.get("critical_force")
    if critical_force is not None:
        ax1.axhline(critical_force, color='crimson', linestyle='--', alpha=0.7,
                    label=f'Critical force: {critical_force:.2f}')

    # --- Create the secondary axis for NIRS ---
    ax2 = ax1.twinx()
    ax2.plot(nirs_time_array, nirs_array, label='SmO2 (%)', color='darkgreen', linestyle=':')
    ax2.set_ylabel('SmO2 (%)', fontsize=14, color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylim(0, 100)

    # --- Determine overall time range (relative) from both datasets ---
    combined_min_time = min(time_array.iloc[0] if hasattr(time_array, 'iloc') else time_array[0],
                            nirs_time_array.iloc[0] if hasattr(nirs_time_array, 'iloc') else nirs_time_array[0])
    combined_max_time = max(time_array.iloc[-1] if hasattr(time_array, 'iloc') else time_array[-1],
                            nirs_time_array.iloc[-1] if hasattr(nirs_time_array, 'iloc') else nirs_time_array[-1])

    # --- Shade baseline and recovery regions on the NIRS axis (using the relative time values) ---
    # Baseline: Everything before test_start_rel
    if test_start_rel > combined_min_time:
        ax2.axvspan(combined_min_time, test_start_rel, color='green', alpha=0.1, label='Baseline')
    # Recovery: Everything after test_end_rel
    if combined_max_time > test_end_rel:
        ax2.axvspan(test_end_rel, combined_max_time, color='cyan', alpha=0.1, label='Recovery')

    # --- Combine legends from both axes and place them at the bottom ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)

    # --- Set main title and adjust layout ---
    fig.suptitle("Combined Force and SmO2 Data", fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.3, left=0.08, right=0.92)

    return fig


def create_force_figure(force_file, test_metrics):
    """
    Creates a matplotlib Figure showing:
      - The force vs. time curve
      - A horizontal line for critical force
      - A red dot & label for maximum strength
      - A shaded area for w_prime
    time_array and force_array should be NumPy arrays (or similar),
    and times are in seconds from start (or however you store them).
    """

    # Read Force data
    force_df = pd.read_feather(force_file)
    force_df['time'] = force_df['time'].astype(float)
    start_time_force = force_df['time'].iloc[0]
    time_array = force_df['time'] - start_time_force
    force_array = force_df['value'].values
    force_array = np.clip(force_array, 0, None)  # Set negative values to 0
    force_array = smooth_data(force_array, window_size=11)

    critical_force = test_metrics.get("critical_force")
    max_strength = test_metrics.get("max_strength")
    # w_prime = test_metrics.get("sum_work_above_cf")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_array, force_array, label='Duration of test', color='darkblue')

    # Plot critical force as a horizontal line
    if critical_force is not None:
        ax.axhline(critical_force, color='crimson',
                   label=f'Critical force: {critical_force:.2f}')

    # Find the index of maximum strength for labeling (if it exists)
    if max_strength is not None:
        max_index = force_array.argmax()
        ax.plot(time_array[max_index], max_strength, 'r.',
                label=f'Maximum strength: {max_strength:.2f}')
        # Optionally annotate the exact value near the point
        ax.text(time_array[max_index], max_strength,
                f'{max_strength:.2f}', fontsize=10, ha='left', va='bottom')

    # # Shade area above critical force for w_prime
    # # only if critical_force is valid
    # if (critical_force is not None) and (w_prime is not None):
    #     ax.fill_between(
    #         time_array, force_array, critical_force,
    #         where=(force_array > critical_force),
    #         color='lightblue', alpha=0.8,
    #         label=f'Work above CF: {w_prime:.2f} [kg/s]'
    #     )

    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Force [kg]', fontsize=14)
    ax.set_ylim(0, max_strength + 5)
    # Gather legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Place the legend at the bottom
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=3, fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    return fig


def create_nirs_figure(nirs_file):
        """
        Creates a matplotlib Figure showing:
          - The nirs vs. time curve

        time_array and nirs_array should be NumPy arrays (or similar),
        and times are in seconds from start (or however you store them).
        """

        # Read Force data
        nirs_df = pd.read_feather(nirs_file)
        nirs_df['time'] = nirs_df['time'].astype(float)
        start_time_force = nirs_df['time'].iloc[0]
        time_array = nirs_df['time'] - start_time_force
        nirs_array = nirs_df['value'].values
        nirs_array = np.clip(nirs_array, 0, None)  # Set negative values to 0
        nirs_array = smooth_data(nirs_array, window_size=11)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_array, nirs_array, label='Duration of test', color='darkgreen')

        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('NIRS (%)', fontsize=14)
        # ax.legend(fontsize=12, loc='upper right')
        # Gather legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Place the legend at the bottom
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  ncol=3, fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        return fig


def plot_normalized_max_force(climber_manager, test_manager, admin_id, current_test_id=None):
    """
    Plot normalized max strength data against climbing grade (IRCRA).
    
    1) Scatter plot with regression line
    2) Histogram of normalized max strength values
    
    Only uses tests with valid max_strength, weight, and IRCRA values.
    
    Parameters:
        climber_manager: Manager for climber data
        test_manager: Manager for test data
        admin_id: ID of the admin user
        current_test_id: ID of the current test to highlight
        
    Returns:
        fig: A matplotlib Figure containing both plots (or None if data insufficient)
    """
    # Get all tests from database
    all_tests = test_manager.fetch_results_by_admin(str(admin_id))
    print(f"Retrieved {len(all_tests)} tests from database")
    
    # Process tests to find valid ones with all required data
    valid_tests = []
    
    for test in all_tests:
        try:
            test_id = test['id']
            participant_id = test.get('participant_id')
            
            # Skip if missing participant ID
            if not participant_id:
                print(f"Test ID {test_id}: Missing participant ID")
                continue
                
            # Try to get test_results - handle as string or dict
            test_results = None
            if isinstance(test.get('test_results'), dict):
                test_results = test.get('test_results')
            elif isinstance(test.get('test_results'), str):
                try:
                    test_results = json.loads(test.get('test_results') or "{}")
                except json.JSONDecodeError:
                    print(f"Test ID {test_id}: Invalid JSON in test_results")
                    continue
            
            # Skip if no test_results or missing max_strength
            if not test_results or 'max_strength' not in test_results:
                print(f"Test ID {test_id}: Missing max_strength data")
                continue
                
            # Get user data
            user = climber_manager.get_user_data(str(admin_id), participant_id)
            
            # Skip if user is missing weight
            if not user or not user.get('weight'):
                print(f"Test ID {test_id}: User missing weight data")
                continue
                
            # Skip if IRCRA is missing or not a number
            ircra = user.get('ircra')
            if not ircra or ircra == 'N/A':
                print(f"Test ID {test_id}: User missing IRCRA grade")
                continue
                
            # Try to convert IRCRA to a number
            try:
                ircra = float(ircra)
            except (ValueError, TypeError):
                print(f"Test ID {test_id}: Invalid IRCRA grade format")
                continue
                
            # Calculate normalized force
            weight = float(user['weight'])
            max_strength = float(test_results['max_strength'])
            norm_max_force = max_strength / weight
            
            # Add to valid tests
            valid_tests.append({
                'test_id': test_id,
                'ircra': ircra,
                'norm_max_force': norm_max_force,
                'max_strength': max_strength,
                'weight': weight
            })
            print(f"Test ID {test_id}: Valid data (IRCRA={ircra}, Weight={weight}, Max strength={max_strength})")
            
        except Exception as e:
            print(f"Error processing test {test.get('id')}: {e}")
    
    # Create DataFrame from valid tests
    df = pd.DataFrame(valid_tests)
    
    # Check if we have any valid data
    if df.empty:
        print("No valid tests found with all required data (max_strength, weight, IRCRA).")
        return None
    
    print(f"\nFound {len(df)} valid tests with all required data")
    print(f"Valid test IDs: {df['test_id'].tolist()}")
    
    # If no current_test_id specified or the specified one isn't valid,
    # use the first valid test instead
    if current_test_id is None or current_test_id not in df['test_id'].values:
        current_test_id = df['test_id'].iloc[0]
        print(f"Using test ID {current_test_id} as current test")
    
    # Get the current test data
    curr = df[df['test_id'] == current_test_id].iloc[0]
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(16, 6))
    
    # PLOT 1: Scatter plot with regression line (left subplot)
    if len(df) >= 2:  # Need at least 2 points for regression
        ax1 = fig.add_subplot(121)  # Left subplot
        
        x = df['ircra']
        y = df['norm_max_force']
        
        # Calculate regression line
        slope, intercept = np.polyfit(x, y, 1)
        line = np.poly1d((slope, intercept))
        
        ax1.scatter(x, y, color='skyblue', alpha=0.7, edgecolor='k', s=70, label='All tests')

        # Plot regression line
        xs = np.linspace(x.min(), x.max(), 100)
        ax1.plot(xs, line(xs), linestyle=':', linewidth=2, color='darkblue',
                 label=f'Trend line (slope={slope:.3f})')

        # Highlight current test
        ax1.scatter(curr['ircra'], curr['norm_max_force'], s=70,
                    color='crimson', label=f'Current test (ID={current_test_id})')
        
        ax1.set_xlabel('Climbing Grade (IRCRA)', fontsize=12)
        ax1.set_ylabel('Max Force / Body Weight', fontsize=12)
        ax1.set_title('Normalized Max Force vs. Climbing Grade', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1 = fig.add_subplot(121)
        ax1.text(0.5, 0.5, "Not enough data points for regression analysis\n(need at least 2)",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Normalized Max Force vs. Climbing Grade', fontsize=14)
        ax1.axis('off')
    
    # PLOT 2: Histogram with normal curve and current test marker (right subplot)
    if len(df) >= 3:  # Need at least 3 points for histogram
        ax2 = fig.add_subplot(122)  # Right subplot
        
        y = df['norm_max_force']
        
        counts, bins, _ = ax2.hist(y, bins=min(20, len(df)),
                                   color='skyblue', alpha=0.7, edgecolor='black')

        # Only add normal curve if we have enough data points
        if len(df) >= 5:
            bin_width = bins[1] - bins[0]
            mu, sigma = y.mean(), y.std()
            x = np.linspace(min(bins), max(bins), 100)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
            ax2.plot(x, pdf * len(y) * bin_width, color='darkblue', linewidth=2,
                     label=f'Normal distribution\n(mean={mu:.2f}, std={sigma:.2f})')

        # Add vertical line for current test
        ax2.axvline(curr['norm_max_force'], color='crimson', linestyle='-', linewidth=2,
                    label=f'Current test: {curr["norm_max_force"]:.2f}')
        
        ax2.set_xlabel('Max Force / Body Weight', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Normalized Max Force', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2 = fig.add_subplot(122)
        ax2.text(0.5, 0.5, "Not enough data points for histogram\n(need at least 3)",
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Distribution of Normalized Max Force', fontsize=14)
        ax2.axis('off')
    
    # Add a title for the entire figure
    # fig.suptitle(f'Normalized Max Force Analysis - Test #{current_test_id}', fontsize=16)
    
    # Adjust spacing between subplots
    # fig.subplots_adjust(top=0.88)  # Make room for the suptitle
    fig.subplots_adjust(
        left=0.05, right=0.98,  # margins on the left & right
        wspace=0.3,  # space between the two axes
        top=0.90,  # top margin (so titles don’t clip)
        bottom=0.15  # bottom margin for x-labels
    )
    fig.tight_layout()
    
    return fig

# if __name__ == '__main__':
#     try:
#         # Create database managers
#         climber_manager = ClimberDatabaseManager()
#         test_manager = ClimbingTestManager()
#
#         # Use test ID 55 which appears to have all required data
#         current_test_id = 28
#
#         # Run the plotting function
#         fig = plot_normalized_max_force(
#             climber_manager,
#             test_manager,
#             admin_id=1,
#             current_test_id=current_test_id
#         )
#
#         # Show the figure if it was created
#         if fig:
#             plt.show()
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # Make sure to close database connections
#         try:
#             climber_manager.close()
#             test_manager.close_connection()
#         except:
#             pass