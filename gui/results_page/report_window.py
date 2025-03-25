import pandas as pd
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, \
    QMessageBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class TestReportWindow(QMainWindow):
    def __init__(self, participant_info, test_metrics, data_type, test_type, force_file=None, nirs_file=None, parent=None):
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

        self.test_metrics = test_metrics

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

        # --- Graph Section ---
        # This is where we create the figure from your single force file
        figure_group = QGroupBox("Force-Time Graph")
        figure_layout = QVBoxLayout()

        # Retrieve time & force arrays from wherever you saved them
        # For example, from test_metrics or from an external function
        # Suppose we do:
        # time_array = test_metrics.get("time_array", [])
        # force_array = test_metrics.get("force_array", [])
        # critical_force = test_metrics.get("critical_force", None)
        # max_strength = test_metrics.get("max_strength", None)
        # w_prime = test_metrics.get("w_prime", None)

        # Create the matplotlib figure
        fig = self.create_force_figure(
            force_file=force_file,
        )

        # Embed the figure in a FigureCanvas
        canvas = FigureCanvas(fig)
        figure_layout.addWidget(canvas)
        figure_group.setLayout(figure_layout)
        main_layout.addWidget(figure_group)

        # if data_type == "force":
        #     self.figure = self.generate_final_graph_force(force_file)
        # elif data_type == "nirs":
        #     self.figure = self.generate_final_graph_nirs(nirs_file)
        # elif data_type == "force_nirs":
        #     self.figure = self.generate_final_combined_graph(force_file, nirs_file)
        # else:
        #     QMessageBox.warning(self, "Error", "Unknown test type; cannot generate graph.")
        #     raise ValueError("Unknown test type; cannot generate graph.")
        #
        # if self.figure is not None:
        #     self.canvas = FigureCanvas(self.figure)
        #     self.canvas.draw()
        #     graph_group = QGroupBox("Test Graph")
        #     graph_layout = QVBoxLayout()
        #     graph_layout.addWidget(self.canvas)
        #     graph_group.setLayout(graph_layout)
        #     main_layout.addWidget(graph_group)
        #TODO: nirs

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

        # --- Close Button ---
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)

        self.setCentralWidget(main_widget)

    import matplotlib.pyplot as plt
    import peakutils

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

        return fig

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
