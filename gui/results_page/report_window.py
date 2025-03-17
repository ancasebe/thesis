import pandas as pd
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class TestReportWindow(QMainWindow):
    def __init__(self, participant_info, test_metrics, data_type, force_file=None, nirs_file=None, parent=None):
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
        self.resize(800, 600)

        # Main container widget and layout.
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # --- Participant Information Section ---
        participant_group = QGroupBox("Participant Info")
        participant_layout = QFormLayout()
        for key, value in participant_info.items():
            participant_layout.addRow(QLabel(f"{key}:"), QLabel(str(value)))
        participant_group.setLayout(participant_layout)
        main_layout.addWidget(participant_group)

        # --- Test Metrics Section ---
        metrics_group = QGroupBox("Test Metrics")
        metrics_layout = QFormLayout()
        for key, value in test_metrics.items():
            display_key = key.replace('_', ' ').capitalize()
            metrics_layout.addRow(QLabel(f"{display_key}:"), QLabel(str(value)))
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # --- Graph Section ---
        # if figure is not None:
        #     graph_group = QGroupBox("Test Graph")
        #     graph_layout = QVBoxLayout()
        #     canvas = FigureCanvas(figure)
        #     graph_layout.addWidget(canvas)
        #     graph_group.setLayout(graph_layout)
        #     main_layout.addWidget(graph_group)
        self.figure = self._generate_figure(data_type, force_file, nirs_file)

        if self.figure is not None:
            self.canvas = FigureCanvas(self.figure)
            self.canvas.draw()
            graph_group = QGroupBox("Test Graph")
            graph_layout = QVBoxLayout()
            graph_layout.addWidget(self.canvas)
            graph_group.setLayout(graph_layout)
            main_layout.addWidget(graph_group)

        # --- Close Button ---
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)

        self.setCentralWidget(main_widget)

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
