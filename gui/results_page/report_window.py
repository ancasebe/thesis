from datetime import datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, \
    QMessageBox, QScrollArea, QHBoxLayout, QSizePolicy, QGridLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class TestReportWindow(QMainWindow):
    def __init__(self, participant_info, db_data, parent=None):
        """
        Creates a window displaying the test report summary with participant info,
        test metrics, and an embedded graph.

        Parameters:
            participant_info (dict): Participant data (e.g., name, email, etc.).
            test_metrics (dict): Test evaluation metrics (e.g., max_strength, critical_force, w_prime).
            figure (matplotlib.figure.Figure): The figure object containing the test graph.
        """
        super().__init__(parent)
        self.setWindowTitle("Test Report Summary")
        self.resize(1000, 800)
        # self.test_metrics = db_data['test_results']
        self.test_metrics = eval(db_data['test_results'], {"np": np})
        self.participant_info = participant_info
        self.db_data = db_data
        # self.nirs_file = nirs_file
        # self.data_type = data_type
        # self.test_type = test_type
        # Create the matplotlib figure
        if db_data['data_type'] == "force":
            force_file = db_data['file_paths']
            fig = self.create_force_figure(
                force_file=force_file
            )
        else:
            fig = None
            print("todo: nirs")
        self.setup_ui(fig)

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
        participant_pairs = list(self.participant_info.items())  # [(key, value), ...]
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
        # group_font = participant_group.font()
        # group_font.setPointSize(14)
        # group_font.setBold(True)
        # participant_group.setFont(group_font)

        # 4) Test Metrics (two columns)
        metrics_pairs = []
        for k, v in self.test_metrics.items():
            display_key = k.replace('_', ' ').capitalize()
            metrics_pairs.append((display_key, str(v)))
        metrics_group = self.create_two_column_group("Test Metrics", metrics_pairs)
        # Make title bigger
        metrics_group.setStyleSheet("""
            QGroupBox::title {
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        # group_font = metrics_group.font()
        # group_font.setPointSize(14)
        # group_font.setBold(True)
        # metrics_group.setFont(group_font)

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
                label=f'w prime: {w_prime:.3f} [kg/s]'
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
        Placeholder: Implement report export functionality.
        """
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Export Report", "Export report functionality not implemented.")

    def show_repetitions(self):
        """
        Placeholder: Implement functionality to show repetitions details.
        """
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Show Repetitions", "Show repetitions functionality not implemented.")
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
