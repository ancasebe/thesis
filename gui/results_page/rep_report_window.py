"""
Module to display the repetition-by-repetition report.

This window shows rep-by-rep metrics in a table and includes a button to display
the rep graphs. It is styled similarly to the main TestReportWindow.
"""
import os
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QPushButton, QScrollArea, QHBoxLayout, QSizePolicy, QLabel
)

from gui.test_page.evaluations.force_evaluation import compute_end_force
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from gui.test_page.evaluations.rep_metrics import RepMetrics  # make sure this is in your PYTHONPATH


class RepReportWindow(QMainWindow):
    def __init__(self, rep_results, force_df, test_id, parent=None):
        """
        Initialize the repetition report window.

        Parameters:
            rep_results (list of dict): List of rep-by-rep metric dictionaries.
            force_df (pd.DataFrame): The original force data (used to generate rep graphs).
            test_id (int): ID of a test.
            parent (QWidget): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Repetition-by-Repetition Report")
        self.resize(1600, 800)
        self.rep_results = rep_results
        self.force_df = force_df

        # We'll create our repetition figure in the constructor (or in setup_ui).
        self.rep_figure = self.create_rep_figure()

        # Save the rep figure to a file if it exists.
        self.rep_graph_filepath = None
        if self.rep_figure is not None:
            # Use uuid to create a unique filename.
            self.rep_graph_filepath = f"{test_id}_rep_graph.png"
            self.rep_figure.savefig(self.rep_graph_filepath, format="png")

        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the UI in one scrollable area that contains:
          - A title label
          - A fixed-height table displaying repetition-by-repetition metrics
          - An embedded Matplotlib figure showing the rep graphs
          - A bottom row of buttons
        """
        # Create one scroll area that will hold all the content.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # The container widget holds all UI elements in a vertical layout.
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(15)

        # 1) Title label at the top.
        title_label = QLabel("Repetition-by-Repetition Report")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold;")
        container_layout.addWidget(title_label)

        # 2) Create the table for rep metrics.
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
        # Set the table's height to exactly fit its rows and header.
        header_height = table.horizontalHeader().height()
        row_height = table.verticalHeader().length()  # total height of all rows
        table.setFixedHeight(header_height + row_height + 2)
        container_layout.addWidget(table)

        # # 3) Embedded Matplotlib figure for the repetition graphs.
        # if self.rep_figure is not None:
        #     canvas = FigureCanvas(self.rep_figure)
        #     canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #     container_layout.addWidget(canvas)

        # 3) Instead of a canvas, show the saved rep graph image.
        if self.rep_graph_filepath and os.path.exists(self.rep_graph_filepath):
            image_label = QLabel()
            pixmap = QPixmap(self.rep_graph_filepath)
            image_label.setPixmap(pixmap)
            # Optionally, adjust scaling behavior
            image_label.setScaledContents(True)
            container_layout.addWidget(image_label)

        # 4) Bottom buttons.
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        container_layout.addLayout(button_layout)

        # Set the container as the widget for the scroll area.
        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

    def create_rep_figure(self):
        """
        Creates a Matplotlib figure with one subplot per repetition.

        Each subplot:
          - Plots the force vs. local time for that rep.
          - Displays a red dot at the maximum force.
          - Draws a dashed horizontal line for the average force.
          - Plots the end force as a green cross.
          - Marks the 20% and 80% force thresholds with cyan diamond markers.
          - Has a small title (e.g. "Rep 1").
        Global X and Y labels ("Time [s]" and "Force [kg]") are added only once.

        Returns:
            fig: A Matplotlib figure.
        """
        # Identify rep intervals using RepMetrics.
        rep_eval = RepMetrics(force_df=self.force_df,
                              threshold_ratio=0.1,
                              min_rep_sec=5,
                              max_rep_sec=10)
        reps = rep_eval.reps
        n_reps = len(reps)
        if n_reps == 0:
            return None  # Nothing to plot

        # Prepare the overall time and force arrays (from the global DataFrame).
        time_axis = self.force_df['timestamp'].values
        force_values = self.force_df['value'].values

        # Define the grid: fixed number of columns (e.g., 3) and as many rows as needed.
        cols = 6
        rows = (n_reps + cols - 1) // cols

        # Create subplots with an increased figure size.
        fig, axes = plt.subplots(rows, cols, figsize=(14, 2 * rows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (start_idx, end_idx) in enumerate(reps):
            ax = axes[i]
            # Slice the repetition data and re-index it for local indexing.
            rep_df = self.force_df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
            rep_time = rep_df['timestamp'].values
            rep_force = rep_df['value'].values

            # Compute local time so that each rep starts at 0.
            local_time = rep_time - rep_time[0]
            ax.plot(local_time, rep_force, color='darkblue')

            # Mark the maximum force (red dot).
            max_i = rep_force.argmax()
            ax.plot(local_time[max_i], rep_force[max_i], 'r.')

            # Draw a dashed horizontal line at the average force.
            avg_force = rep_force.mean()
            ax.axhline(avg_force, color='crimson', linestyle='--', linewidth=1)
            try:
                # Compute the end force and its local index.
                end_force, local_end_idx = compute_end_force(rep_df)
                # Calculate the local time corresponding to the end force sample.
                end_local_time = rep_df['timestamp'].iloc[local_end_idx] - rep_df['timestamp'].iloc[0]
                ax.plot(end_local_time, end_force, marker='x', markersize=10, color='green',
                        label="End Force" if i == 0 else "")

                # Compute 20% and 80% of the maximum force.
                threshold20 = 0.2 * rep_force[max_i]
                threshold80 = 0.8 * rep_force[max_i]

                # Find the first index where force reaches or exceeds 20% threshold.
                idx_20 = None
                for j, val in enumerate(rep_force):
                    if val >= threshold20:
                        idx_20 = j
                        break
                if idx_20 is not None:
                    local_time_20 = rep_df['timestamp'].iloc[idx_20] - rep_df['timestamp'].iloc[0]
                    ax.plot(local_time_20, threshold20, marker='3', markersize=8, color='grey',
                            label="20% Threshold" if i == 0 else "")

                # Find the first index (after idx_20) where force reaches or exceeds 80% threshold.
                idx_80 = None
                for j, val in enumerate(rep_force):
                    if idx_20 is not None and j >= idx_20 and val >= threshold80:
                        idx_80 = j
                        break
                if idx_80 is not None:
                    local_time_80 = rep_df['timestamp'].iloc[idx_80] - rep_df['timestamp'].iloc[0]
                    ax.plot(local_time_80, threshold80, marker='3', markersize=8, color='grey',
                            label="80% Threshold" if i == 0 else "")
            except Exception as e:
                print(f"Error plotting other parameters: {e}")
                return None, None

            # Set subplot title.
            ax.set_title(f"Rep {i + 1}", fontsize=10)
            ax.title.set_position([0.5, 1.05])

        # Hide any unused subplots.
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Adjust subplot margins to provide more white space around the plots.
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.98, hspace=0.2, wspace=0.2)

        # Add global X and Y labels.
        fig.supxlabel("Time [s]", fontsize=12, y=0.04)
        fig.supylabel("Force [kg]", fontsize=12, x=0.04)

        return fig
