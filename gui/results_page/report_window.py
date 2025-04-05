from datetime import datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, \
    QMessageBox, QScrollArea, QHBoxLayout, QSizePolicy, QGridLayout, QFileDialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui.results_page.pdf_exporter import generate_pdf_report
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
        self.resize(1800, 800)
        test_results = db_data['test_results']
        print("Test metrics:", test_results)
        self.test_metrics = eval(test_results, {"np": np})
        self.participant_info = participant_info
        self.db_data = db_data
        # Create the matplotlib figure
        if db_data['data_type'] == "force":
            force_file = db_data['force_file']
            self.fig = self.create_force_figure(
                force_file=force_file
            )
        else:
            self.fig = None
            print("todo: nirs")
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
        # group_font = basic_info_group.font()
        # group_font.setPointSize(14)
        # group_font.setBold(True)
        # basic_info_group.setFont(group_font)
        container_layout.addWidget(basic_info_group)

        # 3) Participant Info (two columns)
        # participant_pairs = list(self.participant_info.items())  # [(key, value), ...]
        participant_pairs = self.build_participant_info_pairs()
        participant_group = self.create_two_column_group("Participant Info", participant_pairs)
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
        show_reps_button = QPushButton("Show Repetitions")
        show_reps_button.clicked.connect(self.show_repetitions)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(export_button)
        bottom_layout.addWidget(show_reps_button)
        bottom_layout.addStretch()  # Pushes the following widget to the right.
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

    def build_participant_info_pairs(self):
        """
        Build a list of (label, value) pairs for the participant info using a predefined mapping.
        """
        user_data_fields = {
            "name": "Name:",
            "surname": "Surname:",
            "email": "Email:",
            "gender": "Gender:",
            "dominant_arm": "Dominant Arm:",
            "weight": "Weight (kg):",
            "height": "Height (cm):",
            "age": "Age (years):",
            "french_scale": "French Scale Level:",
            "years_climbing": "Years of Climbing:",
            "climbing_freq": "Climbing Frequency/week:",
            "climbing_hours": "Climbing Hours/week:",
            "sport_other": "Other sports:",
            "sport_freq": "Sport Frequency/week:",
            "sport_activity_hours": "Sport Activity (hours/week):"
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
        """
        # Gather data from db_data
        test_name = self.db_data.get("test_type", "-")
        data_type = self.db_data.get("data_type", "-")
        arm_tested = self.db_data.get("arm_tested", "-")
        ts = self.db_data.get("timestamp")
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
            ("Test Name", test_name),
            ("Data Type", data_type),
            ("Arm Tested", arm_tested),
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
        force_df['timestamp'] = force_df['timestamp'].astype(float)
        start_time_force = force_df['timestamp'].iloc[0]
        time_array = force_df['timestamp'] - start_time_force
        force_array = force_df['value'].values

        critical_force = self.test_metrics.get("critical_force")
        max_strength = self.test_metrics.get("max_strength")
        w_prime = self.test_metrics.get("w_prime")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_array, force_array, label='Duration of test', color='darkblue')

        # Plot critical force as a horizontal line
        if critical_force is not None:
            ax.axhline(critical_force, color='crimson',
                       label=f'Critical force: {critical_force:.3f}')

        # Find the index of maximum strength for labeling (if it exists)
        if max_strength is not None:
            max_index = force_array.argmax()
            ax.plot(time_array[max_index], max_strength, 'r.',
                    label=f'Maximum strength: {max_strength:.3f}')
            # Optionally annotate the exact value near the point
            ax.text(time_array[max_index], max_strength,
                    f'{max_strength:.2f}', fontsize=10, ha='left', va='bottom')

        # Shade area above critical force for w_prime
        # only if critical_force is valid
        if (critical_force is not None) and (w_prime is not None):
            ax.fill_between(
                time_array, force_array, critical_force,
                where=(force_array > critical_force),
                color='lightblue', alpha=0.8,
                label=f'w prime: {w_prime:.2f} [kg/s]'
            )

        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('Force [kg]', fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)

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
            # import pandas as pd
            df_rep = pd.DataFrame(rep_results)
            rep_table = [df_rep.columns.tolist()] + df_rep.values.tolist()
            # Transform header row to include line breaks for better fit
            transformed_header = [
                "Repetition\nnumber",
                "MVC\n(kg)",
                "Endforce\n(kg)",
                "Force Drop\n(%)",
                "Avg. Force\n(kg)",
                "W\n(kg/s)",
                "W'\n(kg/s)",
                "Pull Time\n(ms)",
                "RFD\n(ms)",
                "RFD norm\n(ms/kg)"
            ]
            rep_table[0] = transformed_header
        else:
            rep_table = None

        # Prepare Force-Time Graph image (from self.fig)
        import io
        buf = io.BytesIO()
        if self.fig:
            self.fig.savefig(buf, format='png')
            buf.seek(0)
        else:
            buf = None

        # Prepare Repetition Graph image using a temporary RepReportWindow
        # from gui.results_page.rep_report_window import RepReportWindow
        try:
            force_df = pd.read_feather(self.db_data['force_file'])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read force file: {e}")
            return

        if rep_results:
            rep_window_temp = RepReportWindow(rep_results, force_df, sampling_rate=100, parent=self)
            rep_graph_fig = rep_window_temp.rep_figure
            rep_buf = io.BytesIO()
            if rep_graph_fig:
                rep_graph_fig.savefig(rep_buf, format='png')
                rep_buf.seek(0)
        else:
            rep_buf = None

        # Parameters explanation text (adjust as needed)
        parameters_explanation = (
            "Maximal Force - MVC (Kg): The highest force output achieved during a maximal isometric contraction.\n"
            "Average End-Force (Kg): The mean force measured toward the end of a sustained contraction.\n"
            "Critical Force - CF (Kg): The asymptotic force that can be maintained without fatigue.\n"
            # ... additional explanations
        )
        print(self.db_data)
        pdf_filename = f"test_{self.db_data['test_type']}_{self.db_data['data_type']}_{self.db_data['id']}.pdf"
        # Choose save path using QFileDialog
        pdf_path, _ = QFileDialog.getSaveFileName(self, "Save Report", pdf_filename, "PDF Files (*.pdf)")
        if not pdf_path:
            return

        try:
            # from gui.results_page.pdf_exporter import generate_pdf_report
            generate_pdf_report(
                pdf_path,
                title_text=f"All-Out Report for {self.participant_info.get('name', 'Unknown')}",
                basic_info=basic_info,
                participant_info=participant_info,
                test_metrics=test_metrics,
                graph_image_path=buf,           # Force-Time Graph
                rep_metrics=rep_table,          # Repetition Metrics table
                rep_graph_image_path=rep_buf,   # Repetition Graph
                parameters_explanation=parameters_explanation
            )
            QMessageBox.information(self, "Export Report", "PDF report generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred: {e}")

    def show_repetitions(self):
        """
        Open a new window that displays repetition-by-repetition metrics and graphs.
        """
        # Retrieve rep_results from your db_data. Here we assume they were saved as a string.
        # For security, consider using json instead of eval.
        from PySide6.QtWidgets import QMessageBox
        rep_results_str = self.db_data['rep_results']
        print('Rep metrics str:', rep_results_str)
        if rep_results_str:
            try:
                # rep_results = eval(rep_results_str, {"__builtins__": None}, {})
                rep_results = eval(rep_results_str, {"np": np})
            except Exception as e:
                # from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Could not parse rep metrics: {e}")
                return
        else:
            # from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No repetition metrics were computed.")
            return

        # Load the force file into a DataFrame for rep graph generation.
        force_file = self.db_data['force_file']
        if force_file is None:
            # from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Force file path not available.")
            return

        try:
            force_df = pd.read_feather(force_file)
        except Exception as e:
            # from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Could not read force file: {e}")
            return

        # Instantiate and show the rep report window.
        self.rep_window = RepReportWindow(rep_results, force_df, sampling_rate=100, parent=self)
        self.rep_window.show()

    '''
    def generate_final_graph_force(self, force_file):
        """
        Generates a final static graph that plots force data.

        Parameters:
            force_file (str): Filename of the NIRS data h5.
        """
        # Read Force data.
        # force_df = pd.read_csv(force_file)
        # force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
        # force_timestamps = force_df['timestamp'].values
        # force_values = force_df['value'].values

        force_df = pd.read_hdf(force_file, key="data")
        force_df['timestamp'] = force_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        force_timestamps = force_df['timestamp'].values
        force_values = force_df['value'].values

        # Create plot for Force only.
        fig, ax = plt.subplots()
        ax.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (kg)", color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True)
        fig.tight_layout()
        plt.title("Final Force Data")
        ax.legend(loc="upper right")
        plt.show()
        return fig
        # plt.show()

    def generate_final_graph_nirs(self, nirs_file):
        """
        Generates a final static graph that plots NIRS data.

        Parameters:
            nirs_file (str): Filename of the NIRS data h5.
        """
        # # Read NIRS data.
        # nirs_df = pd.read_csv(nirs_file)
        # nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
        # nirs_timestamps = nirs_df['timestamp'].values
        # nirs_values = nirs_df['value'].values

        # Read NIRS data.
        nirs_df = pd.read_hdf(nirs_file, key="data")
        nirs_df['timestamp'] = nirs_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        nirs_timestamps = nirs_df['timestamp'].values
        nirs_values = nirs_df['value'].values

        # Create plot for NIRS only.
        fig, ax = plt.subplots()
        ax.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIRS (%)", color='r')
        ax.tick_params(axis='y', labelcolor='r')
        ax.grid(True)
        fig.tight_layout()
        plt.title("Final NIRS Data")
        ax.legend(loc="upper right")
        return fig
        # plt.show()

    def generate_final_combined_graph(self, force_file, nirs_file):
        """
        Generates a final static graph that plots both Force and NIRS data on a single figure with two y-axes.

        Parameters:
            force_file (str): Filename of the Force data h5.
            nirs_file (str): Filename of the NIRS data h5.
        """
        # # Read Force data.
        # force_df = pd.read_csv(force_file)
        # force_df['timestamp'] = force_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
        # force_timestamps = force_df['timestamp'].values
        # force_values = force_df['value'].values
        #
        # # Read NIRS data.
        # nirs_df = pd.read_csv(nirs_file)
        # nirs_df['timestamp'] = nirs_df['force_timestamp'].str.replace("force_", "", regex=False).astype(float)
        # nirs_timestamps = nirs_df['timestamp'].values
        # nirs_values = nirs_df['value'].values

        # Read Force data.
        force_df = pd.read_hdf(force_file, key="data")
        force_df['timestamp'] = force_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        force_timestamps = force_df['timestamp'].values
        force_values = force_df['value'].values

        # Read NIRS data.
        nirs_df = pd.read_hdf(nirs_file, key="data")
        nirs_df['timestamp'] = nirs_df['timestamp'].str.replace("timestamp_", "", regex=False).astype(float)
        nirs_timestamps = nirs_df['timestamp'].values
        nirs_values = nirs_df['value'].values

        # Create a combined plot.
        fig, ax1 = plt.subplots()
        ax1.plot(force_timestamps, force_values, 'b-', label="Force [kg]")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Force (kg)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(nirs_timestamps, nirs_values, 'r-', label="NIRS (%)")
        ax2.set_ylabel("NIRS (%)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Gather legend handles & labels from both axes and combine them:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.tight_layout()
        plt.title("Final Combined Sensor Data")
        return fig
    '''

