"""
Module to display the repetition-by-repetition report.

This window shows rep-by-rep metrics in a table and includes a button to display
the rep graphs. It is styled similarly to the main TestReportWindow.
"""

import pandas as pd
from PySide6.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QPushButton, QScrollArea, QHBoxLayout, QSizePolicy, QLabel
)
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from gui.test_page.evaluations.rep_metrics import RepMetrics  # make sure this is in your PYTHONPATH


class RepReportWindow(QMainWindow):
    def __init__(self, rep_results, force_df, sampling_rate=100, parent=None):
        """
        Initialize the repetition report window.

        Parameters:
            rep_results (list of dict): List of rep-by-rep metric dictionaries.
            force_df (pd.DataFrame): The original force data (used to generate rep graphs).
            sampling_rate (int): Sampling rate in Hz.
            parent (QWidget): Parent widget.
        """
        super().__init__(parent)
        # self.setWindowTitle("Repetition-by-Repetition Report")
        # self.resize(900, 600)
        # self.rep_results = rep_results
        # self.force_df = force_df
        # self.sampling_rate = sampling_rate
        # self.setup_ui()
        self.setWindowTitle("Repetition-by-Repetition Report")
        self.resize(900, 600)
        self.rep_results = rep_results
        self.force_df = force_df
        self.sampling_rate = sampling_rate

        # We'll create our repetition figure in the constructor (or in setup_ui).
        self.rep_figure = self.create_rep_figure()

        self.setup_ui()

    def setup_ui(self):
        """
        Set up the UI: a scrollable area with a title, a table displaying rep metrics,
        and an embedded Matplotlib figure showing rep-by-rep plots.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 1) Title
        title_group = QGroupBox("Repetition-by-Repetition Metrics")
        title_layout = QVBoxLayout(title_group)
        title_label = QLabel("Repetition-by-Repetition Report")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold;")
        title_layout.addWidget(title_label)
        layout.addWidget(title_group)

        # 2) Table of rep metrics
        table = QTableWidget()
        df = pd.DataFrame(self.rep_results)
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)

        # 3) Embedded Matplotlib figure
        if self.rep_figure is not None:
            canvas = FigureCanvas(self.rep_figure)
            # Make sure the canvas can expand nicely
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(canvas)

        # 4) Bottom buttons layout
        button_layout = QHBoxLayout()
        # Optionally remove or rename the Show Graphs button if you no longer need it.
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

    # def setup_ui(self):
    #     """
    #     Set up the UI: a scrollable area with a title, a table displaying rep metrics,
    #     and a bottom row with buttons.
    #     """
    #     scroll_area = QScrollArea()
    #     scroll_area.setWidgetResizable(True)
    #     container = QWidget()
    #     layout = QVBoxLayout(container)
    #     layout.setContentsMargins(20, 20, 20, 20)
    #     layout.setSpacing(15)
    #
    #     # Title label
    #     title = QGroupBox("Repetition-by-Repetition Metrics")
    #     title_layout = QVBoxLayout(title)
    #     title_label = QTableWidgetItem("Repetition-by-Repetition Report")
    #     # Instead of using QTableWidgetItem, we can simply use a QLabel if desired.
    #     # For simplicity, we add the title as a styled QLabel.
    #     from PySide6.QtWidgets import QLabel
    #     title_label = QLabel("Repetition-by-Repetition Report")
    #     title_label.setAlignment(Qt.AlignCenter)
    #     title_label.setStyleSheet("font-size: 20pt; font-weight: bold;")
    #     title_layout.addWidget(title_label)
    #     layout.addWidget(title)
    #
    #     # Create the table widget
    #     table = QTableWidget()
    #     df = pd.DataFrame(self.rep_results)
    #     table.setRowCount(df.shape[0])
    #     table.setColumnCount(df.shape[1])
    #     table.setHorizontalHeaderLabels(df.columns)
    #     for i in range(df.shape[0]):
    #         for j in range(df.shape[1]):
    #             item = QTableWidgetItem(str(df.iat[i, j]))
    #             item.setTextAlignment(Qt.AlignCenter)
    #             table.setItem(i, j, item)
    #     table.resizeColumnsToContents()
    #     layout.addWidget(table)
    #
    #     # Bottom buttons layout
    #     button_layout = QHBoxLayout()
    #     show_graphs_button = QPushButton("Show Rep Graphs")
    #     show_graphs_button.clicked.connect(self.show_rep_graphs)
    #     close_button = QPushButton("Close")
    #     close_button.clicked.connect(self.close)
    #     button_layout.addWidget(show_graphs_button)
    #     button_layout.addStretch()
    #     button_layout.addWidget(close_button)
    #     layout.addLayout(button_layout)
    #
    #     scroll_area.setWidget(container)
    #     self.setCentralWidget(scroll_area)

    def create_rep_figure(self):
        """
        Creates a Matplotlib figure with one subplot per repetition.

        Each subplot:
          - Plots the force vs. local time for that rep
          - Shows a red dot at max force
          - Draws a dashed horizontal line at average force
          - Has a small title "Rep X"
        The figure has one shared X label "Time [s]" and one shared Y label "Force [kg]".
        """
        # 1) We need to identify the rep intervals from the data
        rep_eval = RepMetrics(self.force_df, sampling_rate=self.sampling_rate)
        reps = rep_eval.reps
        n_reps = len(reps)
        if n_reps == 0:
            return None  # No reps to plot

        # 2) Prepare time and force arrays
        time_axis = self.force_df['timestamp'].values
        force_values = self.force_df['value'].values

        # 3) Decide how many columns/rows for subplots
        cols = 3
        rows = (n_reps + cols - 1) // cols  # integer ceil

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        # 4) Plot each rep in its own subplot
        for i, (start_idx, end_idx) in enumerate(reps):
            ax = axes[i]
            # Extract the portion of the data for this repetition
            rep_time = time_axis[start_idx:end_idx+1]
            rep_force = force_values[start_idx:end_idx+1]

            # Convert time so that each rep starts at t=0
            local_time = rep_time - rep_time[0]

            # Plot the rep
            ax.plot(local_time, rep_force, label=f"Rep {i+1}", color='blue')

            # Red dot at max force
            max_i = rep_force.argmax()
            ax.plot(local_time[max_i], rep_force[max_i], 'ro')

            # Dashed line at average force
            avg_force = rep_force.mean()
            ax.axhline(avg_force, color='gray', linestyle='--', linewidth=1)

            # Title above each subplot
            ax.set_title(f"Rep {i+1}")

        # 5) Hide unused subplots if n_reps < rows*cols
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        # 6) Add a single X label and Y label for the entire figure
        fig.supxlabel("Time [s]")
        fig.supylabel("Force [kg]")

        fig.tight_layout()
        return fig

    def show_rep_graphs(self):
        """
        Generate and show the repetition-by-repetition graphs using RepMetrics.
        """
        # Instantiate the rep evaluator again (or re-use if you stored it)

        rep_eval = RepMetrics(self.force_df, sampling_rate=self.sampling_rate)
        rep_eval.plot_rep_graphs()

