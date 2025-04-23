import os
from datetime import datetime

import numpy as np
import pandas as pd
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, \
    QMessageBox, QScrollArea, QHBoxLayout, QSizePolicy, QGridLayout, QFileDialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui.results_page.pdf_exporter import generate_pdf_report, parameters_explanation_dict, filter_parameters_explanation
from gui.results_page.rep_report_window import RepReportWindow


class TestReportWindow(QMainWindow):
    def __init__(self, participant_info, db_data, parent=None):
        """
        Creates a window displaying the test report summary with participant info,
        test metrics, and an embedded graph.

        Parameters:
            participant_info (dict): Participant data (e.g., name, email, etc.).
            db_data (dict): Information about the selected test saved in the database.
        """
        super().__init__(parent)
        self.setWindowTitle("Test Report Summary")
        self.resize(1600, 800)
        
        # Parse test_results from db_data
        self.test_metrics = None
        if db_data.get('test_results') and db_data['test_results'] != "null":
            try:
                self.test_metrics = json.loads(db_data['test_results'])
            except json.JSONDecodeError as e:
                print(f"Error parsing test_results JSON: {e}")
                try:
                    # Fallback for old format data
                    self.test_metrics = eval(db_data['test_results'], {"np": np})
                except Exception as e2:
                    print(f"Failed to parse test_results with eval: {e2}")
        
        # Parse nirs_results from db_data
        self.nirs_results = None
        if db_data.get('nirs_results') and db_data['nirs_results'] != "null":
            try:
                # Print the raw value to debug
                print(f"Raw nirs_results value: {repr(db_data['nirs_results'])}")
                self.nirs_results = json.loads(db_data['nirs_results'])
            except json.JSONDecodeError as e:
                print(f"Error parsing nirs_results JSON: {e}")
                try:
                    # Fallback for old format data
                    self.nirs_results = eval(db_data['nirs_results'], {"np": np})
                except Exception as e2:
                    print(f"Failed to parse nirs_results with eval: {e2}")
                    # Ensure we have a valid value even if parsing fails
                    self.nirs_results = None
        
        self.participant_info = participant_info
        self.db_data = db_data
        
        # Create the matplotlib figure
        print('data_type:', db_data['data_type'])
        self.fig = None
        try:
            if db_data['data_type'] == "force":
                force_file = db_data['force_file']
                self.fig = self.create_force_figure(force_file=force_file)
            elif db_data['data_type'] == "nirs":
                nirs_file = db_data['nirs_file']
                self.fig = self.create_nirs_figure(nirs_file=nirs_file)
                print("todo: nirs")
            elif db_data['data_type'] == "force_nirs":
                force_file = db_data['force_file']
                nirs_file = db_data['nirs_file']
                self.fig = self.create_combined_figure(
                    force_file=force_file,
                    nirs_file=nirs_file
                )
        except Exception as e:
            print(f"Error creating figure: {e}")
        
        self.setup_ui(self.fig)

    def setup_ui(self, fig):
        """
        Sets up the scrollable layout with:
          - Title
          - Basic Test Info (2-column)
          - Participant Info (2-column)
          - Test Metrics (2-column)
          - Graph
          - Bottom Buttons
        """
        # Create a scroll area so content can be scrolled if needed
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)
        container_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Big Title at the top
        title_label = QLabel(f"All-Out Report for {self.participant_info.get('name', 'Unknown')}")
        title_font = title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title_label)

        # 2) Basic Test Info (two columns of label–value)
        basic_info_pairs = self.build_basic_info_pairs()
        basic_info_group = self.create_two_column_group("Basic Test Information", basic_info_pairs)
        # Make title bigger
        basic_info_group.setStyleSheet("""
            QGroupBox::title {
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        # If the test contains NIRS data, add a NIRS Results section.
        if self.db_data["data_type"] in ["nirs", "force_nirs"]:
            nirs_results_pairs = self.build_nirs_results_pairs()
            print("DEBUG: nirs_results_pairs =", nirs_results_pairs)
            if nirs_results_pairs:
                nirs_results_group = self.create_two_column_group("NIRS Results", nirs_results_pairs)
                nirs_results_group.setStyleSheet("""
                    QGroupBox::title {
                        font-size: 14pt;
                        font-weight: bold;
                    }
                """)

                info_layout = QHBoxLayout()
                info_layout.addWidget(basic_info_group, stretch=1)
                info_layout.addWidget(nirs_results_group, stretch=1)
                container_layout.addLayout(info_layout)
                # container_layout.addWidget(nirs_results_group)
        else:
            container_layout.addWidget(basic_info_group)

        # 3) Participant Info (two columns)
        # participant_pairs = list(self.participant_info.items())  # [(key, value), ...]
        participant_pairs = self.build_participant_info_pairs()
        # participant_group = self.create_two_column_group("Participant Info", participant_pairs)
        # Convert them to label–value
        # For example, participant_pairs = [("name", "Anna"), ("surname", "Sebestikova"), ...]
        participant_group = self.create_two_column_group("Participant Info", participant_pairs)
        # Make title bigger
        participant_group.setStyleSheet("""
            QGroupBox::title {
                font-size: 14pt;
                font-weight: bold;
            }
        """)

        metrics_pairs = self.build_test_metrics_pairs()
        metrics_group = self.create_two_column_group("Test Metrics", metrics_pairs)
        # Make title bigger
        metrics_group.setStyleSheet("""
            QGroupBox::title {
                font-size: 14pt;
                font-weight: bold;
            }
        """)

        # Put participant info & metrics side-by-side
        info_metrics_layout = QHBoxLayout()
        info_metrics_layout.addWidget(participant_group, stretch=1)
        info_metrics_layout.addWidget(metrics_group, stretch=1)
        container_layout.addLayout(info_metrics_layout)

        # 5) Graph Section
        if fig is not None:
            graph_group = QGroupBox("Force-Time Graph")
            graph_layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(800, 400)  # prevent squishing
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            graph_layout.addWidget(canvas)
            graph_group.setLayout(graph_layout)
            container_layout.addWidget(graph_group)

        # 6) Bottom buttons row: left side "Export Report" and "Show Repetitions"; right side "Close"
        bottom_layout = QHBoxLayout()
        export_button = QPushButton("Export Report")
        export_button.clicked.connect(self.export_report)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(export_button)
        if self.db_data['test_type'] in ['ao', 'iit', 'iirt']:
            show_reps_button = QPushButton("Show Repetitions")
            show_reps_button.clicked.connect(self.show_repetitions)
            bottom_layout.addWidget(show_reps_button)
        # bottom_layout.addStretch()  # Pushes the following widget to the right.
        bottom_layout.addWidget(close_button)
        container_layout.addLayout(bottom_layout)

        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

    # import matplotlib.pyplot as plt
    # import peakutils

    def build_test_metrics_pairs(self):
        """
        Build a list of (label, value) pairs for test metrics using a predefined mapping.
        This function uses the keys from self.test_metrics and a mapping dictionary to create
        labels as they should appear in the GUI.
        """
        display_names = {
            "max_strength": "Maximal Force - MVC (kg)",
            "avg_end_force": "Average End-Force (kg)",
            "time_between_max_end_ms": "Average Time btw Max- and End-Force (ms)",
            "force_drop_pct": "Average Force Drop (%)",
            "avg_rep_force": "Average Rep. Force (kg)",
            "critical_force": "Critical Force - CF (kg)",
            "reps_to_cf": "Repetitions to CF",
            "cf_mvc_pct": "CF/MVC (%)",
            "work": "Average Work (kg/s)",
            "sum_work": "Sum Work (kg/s)",
            "avg_work_above_cf": "Average Work above CF (kg/s)",
            "sum_work_above_cf": "Sum Work above CF (kg/s)",
            "avg_pulling_time_ms": "Average Pulling Time (ms)",
            "rfd_overall": "Rate of Force Development - RFD (ms)",
            "rfd_first3": "RFD first three repetitions (ms)",
            "rfd_first6": "RFD first six repetitions (ms)",
            "rfd_last3": "RFD last three repetitions (ms)",
            "rfd_norm_overall": "RFD normalized to force (ms/kg)",
            "rfd_norm_first3": "RFD norm. first three rep. (ms/kg)",
            "rfd_norm_first6": "RFD norm. first six rep. (ms/kg)",
            "rfd_norm_last3": "RFD norm. last three rep. (ms/kg)"
        }
        pairs = []
        # Loop through each key in the test metrics dictionary.
        for key, value in self.test_metrics.items():
            # Use the display name if available; otherwise, use a default conversion.
            label = display_names.get(key, key.replace('_', ' ').capitalize())
            pairs.append((label, str(value)))
        return pairs

    def build_nirs_results_pairs(self):
        """
        Build a list of (label, value) pairs for NIRS evaluation results.
        Assumes that self.test_metrics (obtained from db_data['test_results'])
        contains an entry 'nirs_results' that is a dictionary with keys such as
        'baseline_mean' and 'time_to_recovery'.
        """
        pairs = []
        # You might have stored it as a dictionary inside test_metrics.
        print('nirs_results:', self.nirs_results)
        if self.nirs_results:
            mapping = {
                "baseline_mean": "Baseline Mean (%)",
                "time_to_recovery": "Time to Recovery (s)"
            }
            for key, value in self.nirs_results.items():
                label = mapping.get(key, key.replace('_', ' ').capitalize())
                pairs.append((label, str(value)))
        return pairs


    @staticmethod
    def get_test_type_display(test_type):
        """
        Returns a human-readable display name for the test type.
        """
        test_type_mapping = {
            "ao": "All-Out Test",
            "mvc": "Maximum Voluntary Contraction",
            "iit": "Intermittent Incremental Test",
            "iirt": "Intermittent Isometric Resistance Test",
            "ec": "Endurance Capacity",
            "sit": "Sprint Interval Test"
        }
        return test_type_mapping.get(test_type, test_type)

    @staticmethod
    def get_data_type_display(data_type):
        """
        Returns a human-readable display name for the data type.
        """
        data_type_mapping = {
            "force": "Force only",
            "nirs": "NIRS only",
            "force_nirs": "Force & NIRS"
        }
        return data_type_mapping.get(data_type, data_type)

    def build_participant_info_pairs(self):
        """
        Build a list of (label, value) pairs for the participant info using a predefined mapping.
        Compatible with data from ClimberDatabaseManager.get_user_data()
        """
        user_data_fields = {
            "name": "Name",
            "surname": "Surname",
            "email": "Email",
            "gender": "Gender",
            "dominant_arm": "Dominant Arm",
            "weight": "Weight (kg)",
            "height": "Height (cm)",
            "age": "Age (years)",
            "ircra": "IRCRA",
            "years_climbing": "Years of Climbing",
            "climbing_freq": "Climbing Frequency/week",
            "climbing_hours": "Climbing Hours/week",
            "sport_other": "Other sports",
            "sport_freq": "Sport Frequency/week",
            "sport_activity_hours": "Sport Activity (hours/week)"
        }
        pairs = []
        # Loop through the mapping keys and build the pairs.
        for key, label in user_data_fields.items():
            value = self.participant_info.get(key, "-")
            pairs.append((label, str(value)))
        return pairs

    def build_basic_info_pairs(self):
        """
        Returns a list of (label, value) pairs for the basic test info section,
        each to be displayed in two columns.
        Compatible with data from CombinedDataCommunicator and db_data from ClimbingTestManager
        """
        # Gather data from db_data
        test_type = self.db_data.get("test_type", "-")
        data_type = self.db_data.get("data_type", "-")
        arm_tested = self.db_data.get("arm_tested", "-")
        number_of_reps = self.db_data.get("number_of_reps", "-")
        ts = self.db_data.get("timestamp", "-")

        # Format arm_tested for display
        arm_text = "Dominant" if arm_tested == "D" else "Non-dominant" if arm_tested == "N" else arm_tested

        # Format timestamp
        date_str, time_str = "", ""
        if ts:
            try:
                tfloat = float(ts)
                dt = datetime.fromtimestamp(tfloat)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                pass

        pairs = [
            ("Test Name", self.get_test_type_display(test_type)),
            ("Data Type", self.get_data_type_display(data_type)),
            ("Arm Tested", arm_text),
            ("Number of Repetitions", number_of_reps),
            ("Date", date_str),
            ("Time", time_str)
        ]
        return pairs

    @staticmethod
    def create_two_column_group(title, pairs):
        """
        Creates a QGroupBox with a 2-column layout of label–value pairs.
        For example, if pairs has 6 items, 3 go in the left column, 3 in the right column.
        """
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(50)  # spacing between columns
        group.setLayout(grid)

        # We'll split pairs in half: first half in columns 0/1, second half in columns 2/3
        half = (len(pairs) + 1) // 2  # integer ceiling
        for i in range(half):
            label_text, value_text = pairs[i]
            label_widget = QLabel(str(label_text) + ":")
            value_widget = QLabel(str(value_text))
            # Align left
            label_widget.setAlignment(Qt.AlignLeft)
            value_widget.setAlignment(Qt.AlignLeft)

            grid.addWidget(label_widget, i, 0, alignment=Qt.AlignLeft)
            grid.addWidget(value_widget, i, 1, alignment=Qt.AlignLeft)

        # second half
        for j in range(half, len(pairs)):
            row = j - half
            label_text, value_text = pairs[j]
            label_widget = QLabel(str(label_text) + ":")
            value_widget = QLabel(str(value_text))
            label_widget.setAlignment(Qt.AlignLeft)
            value_widget.setAlignment(Qt.AlignLeft)

            grid.addWidget(label_widget, row, 2, alignment=Qt.AlignLeft)
            grid.addWidget(value_widget, row, 3, alignment=Qt.AlignLeft)

        return group

    def create_force_figure(self, force_file):
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
        force_array = self.smooth_data(force_array, window_size=11)

        critical_force = self.test_metrics.get("critical_force")
        max_strength = self.test_metrics.get("max_strength")
        # w_prime = self.test_metrics.get("sum_work_above_cf")

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

    def create_nirs_figure(self, nirs_file):
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
        nirs_array = self.smooth_data(nirs_array, window_size=11)

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
    # @staticmethod
    # def smooth_data(data, window_size=11):
    #     """Smooths the data using a simple moving average."""
    #     if window_size < 2:
    #         return data
    #     kernel = np.ones(window_size) / window_size
    #     return np.convolve(data, kernel, mode='same')

    @staticmethod
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

    def create_combined_figure(self, force_file, nirs_file):
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
        force_array = self.smooth_data(force_array, window_size=11)

        # --- Identify test start and end times from force data (absolute timestamps) ---
        # max_force = force_array.max()
        max_force = self.test_metrics.get("max_strength")
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
        nirs_array = self.smooth_data(nirs_array, window_size=25)

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

        critical_force = self.test_metrics.get("critical_force")
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

    def export_report(self):
        """
        Exports the complete PDF report including:
          - Basic info, Participant info, Test Metrics, Force-Time Graph,
          - Repetition Metrics table, Repetition Graph, and Parameters Explanation.
        """
        # Gather data from helper functions
        basic_info = self.build_basic_info_pairs()
        participant_info = self.build_participant_info_pairs()
        test_metrics = self.build_test_metrics_pairs()
        if self.db_data['data_type'] != 'force':
            nirs_results = self.build_nirs_results_pairs()
        else:
            nirs_results = None
        try:
            force_df = pd.read_feather(self.db_data['force_file'])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read force file: {e}")
            return
        if self.db_data['test_type'] in ['ao', 'iit', 'iirt']:
            # Retrieve rep_results from db_data and convert to a table
            rep_results_str = self.db_data.get('rep_results', "")
            if rep_results_str:
                try:
                    rep_results = eval(rep_results_str, {"np": np})
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not parse rep metrics: {e}")
                    return
            else:
                rep_results = []
            if rep_results:
                df_rep = pd.DataFrame(rep_results)
                rep_table = [df_rep.columns.tolist()] + df_rep.values.tolist()
                # Transform header row to include line breaks for better fit
                transformed_header = [
                    "Rep\nno.",
                    "Max Force\n(kg)",
                    "End Force\n(kg)",
                    "Force Drop\n(%)",
                    "Avg. Force\n(kg)",
                    "Pull Time\n(ms)",
                    "Max-End\nTime (s)",
                    "RFD\n(ms)",
                    "RFD norm\n(ms/kg)",
                    "W\n(kg/s)",
                    "W'\n(kg/s)",
                ]
                rep_table[0] = transformed_header
                rep_window_temp = RepReportWindow(rep_results=rep_results,
                                                  force_df=force_df,
                                                  test_id=self.db_data['id'],
                                                  parent=self)
                rep_graph_filepath = rep_window_temp.rep_graph_filepath
            else:
                rep_table = None
                rep_graph_filepath = None
        else:
            rep_table = None
            rep_graph_filepath = None

        # Filter the explanation text based on the test metrics
        filtered_parameters_text = filter_parameters_explanation(self.test_metrics, parameters_explanation_dict)

        # Save the Force-Time Graph.
        if self.fig:
            force_graph_filepath = f"{self.db_data['id']}_graph.png"
            self.fig.savefig(force_graph_filepath, format='png')
        else:
            force_graph_filepath = None

        pdf_filename = f"test_{self.db_data['test_type']}_{self.db_data['data_type']}_{self.db_data['id']}.pdf"
        # Choose save path using QFileDialog
        pdf_path, _ = QFileDialog.getSaveFileName(self, "Save Report", pdf_filename, "PDF Files (*.pdf)")
        if not pdf_path:
            return

        try:
            generate_pdf_report(
                pdf_path=pdf_path,
                title_text=f"{self.db_data['test_type'].upper()} Report for {self.participant_info.get('name', 'Unknown')}",
                basic_info=basic_info,
                participant_info=participant_info,
                test_results=test_metrics,
                nirs_results=nirs_results,
                graph_image_path=force_graph_filepath,           # Force-Time Graph
                rep_results=rep_table,          # Repetition Metrics table
                rep_graph_image_path=rep_graph_filepath,   # Repetition Graph
                parameters_explanation=filtered_parameters_text
            )
            QMessageBox.information(self, "Export Report", "PDF report generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred: {e}")
            print(e)

    def show_repetitions(self):
        """
        Open a new window that displays repetition-by-repetition metrics and graphs.
        """
        if not hasattr(self, 'db_data') or not self.db_data:
            QMessageBox.warning(self, "No Data", "No repetition data available.")
            return

        # Parse rep_results if it's a JSON string
        rep_results_db = self.db_data.get('rep_results')
        if not rep_results_db or rep_results_db == "null":
            QMessageBox.warning(self, "No Data", "No repetition data available.")
            return
        print('rep_results_db:', rep_results_db)

        try:
            rep_results = json.loads(rep_results_db)
        except json.JSONDecodeError:
            # Fallback for old format
            rep_results = eval(rep_results_db)

        if not rep_results:
            QMessageBox.warning(self, "No Data", "No repetition data available.")
            return

        # Load force data if available
        force_file = self.db_data.get('force_file')
        force_df = None
        if force_file and os.path.exists(force_file):
            try:
                force_df = pd.read_feather(force_file)
            except Exception as e:
                print(f"Error loading force data: {e}")

        # # Retrieve rep_results from your db_data. Here we assume they were saved as a string.
        # # For security, consider using json instead of eval.
        # rep_results_str = self.db_data['rep_results']
        # print('Rep metrics str:', rep_results_str)
        # if rep_results_str:
        #     try:
        #         rep_results = eval(rep_results_str, {"np": np})
        #     except Exception as e:
        #         QMessageBox.warning(self, "Error", f"Could not parse rep metrics: {e}")
        #         return
        # else:
        #     QMessageBox.information(self, "No Data", "No repetition metrics were computed.")
        #     return
        #
        # # Load the force file into a DataFrame for rep graph generation.
        # force_file = self.db_data['force_file']
        # if force_file is None:
        #     QMessageBox.warning(self, "Error", "Force file path not available.")
        #     return
        #
        # try:
        #     force_df = pd.read_feather(force_file)
        # except Exception as e:
        #     QMessageBox.warning(self, "Error", f"Could not read force file: {e}")
        #     return

        # Instantiate and show the rep report window.
        self.rep_window = RepReportWindow(rep_results=rep_results,
                                          force_df=force_df,
                                          test_id=self.db_data['id'],
                                          parent=self)
        self.rep_window.show()

    def closeEvent(self, event):
        """
        Delete the temporary force graph file and also ensure that if a rep report window is open,
        it is closed (its own closeEvent will delete its temporary rep graph file).
        """
        graph_filepath = f"{self.db_data['id']}_graph.png"
        if os.path.exists(graph_filepath):
            try:
                os.remove(graph_filepath)
            except Exception as e:
                print("Error deleting graph file:", e)
        rep_graph_filepath = f"{self.db_data['id']}_rep_graph.png"
        if os.path.exists(rep_graph_filepath):
            try:
                os.remove(rep_graph_filepath)
            except Exception as e:
                print("Error deleting rep graph file:", e)
        if hasattr(self, 'rep_window') and self.rep_window is not None:
            self.rep_window.close()  # This will trigger rep_window's closeEvent.
        super().closeEvent(event)
