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
from gui.results_page.graphs_generator import create_combined_figure, create_force_figure, create_nirs_figure, plot_normalized_max_force
from gui.statistics.ircra_predictor import IRCRAPredictor


class TestReportWindow(QMainWindow):
    def __init__(self, participant_info, db_data, admin_id, climber_db_manager, 
                 test_db_manager, parent=None):
        """
        Creates a window displaying the test report summary with participant info,
        test metrics, and an embedded graph.

        Parameters:
            participant_info (dict): Participant data (e.g., name, email, etc.).
            db_data (dict): Information about the selected test saved in the database.
            admin_id (int): Logged-in admin ID.
            climber_db_manager (ClimberDatabaseManager): Instance of ClimberDatabaseManager.
            test_db_manager (ClimbingTestManager): Instance of ClimbingTestManager.
        """
        super().__init__(parent)
        self.setWindowTitle("Test Report Summary")
        self.resize(1600, 800)

        # Store references to managers and admin_id
        self.climber_manager = climber_db_manager
        self.test_manager = test_db_manager
        self.admin_id = admin_id

        # Parse test_results from db_data
        self.test_metrics = None
        if db_data.get('test_results') and db_data['test_results'] != "null":
            try:
                # Check if test_results is already a dictionary
                if isinstance(db_data['test_results'], dict):
                    self.test_metrics = db_data['test_results']
                else:
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
                if isinstance(db_data['nirs_results'], dict):
                    self.nirs_results = db_data['nirs_results']
                else:
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
        self.ircra_prediction = None
        
        # Create the matplotlib figure
        self.fig = None
        try:
            if db_data['data_type'] == "force":
                force_file = db_data['force_file']
                self.fig = create_force_figure(force_file=force_file, test_metrics=self.test_metrics)
            elif db_data['data_type'] == "nirs":
                nirs_file = db_data['nirs_file']
                self.fig = create_nirs_figure(nirs_file=nirs_file)
            elif db_data['data_type'] == "force_nirs":
                force_file = db_data['force_file']
                nirs_file = db_data['nirs_file']
                self.fig = create_combined_figure(
                    force_file=force_file,
                    nirs_file=nirs_file,
                    test_metrics=self.test_metrics
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
        if self.db_data["data_type"] in ["nirs", "force_nirs"] and self.nirs_results is not None:
            nirs_results_pairs = self.build_nirs_results_pairs()
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
        participant_pairs = self.build_participant_info_pairs()
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
        participant_layout = QVBoxLayout()
        participant_layout.addWidget(participant_group)

        # Add the prediction box only for 'ao' tests
        if self.db_data.get('test_type') == 'ao':
            prediction_group = self.create_performance_prediction_group()
            if prediction_group:
                participant_layout.addWidget(prediction_group)

        # Add participant section and metrics to the info_metrics layout
        info_metrics_layout.addLayout(participant_layout, stretch=1)
        info_metrics_layout.addWidget(metrics_group, stretch=1)
        container_layout.addLayout(info_metrics_layout)
        # participant_section_layout.addWidget(self.create_performance_prediction_group())

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
            
        # 6) Normalized Max Force Plots (if managers are available)
        if self.climber_manager and self.test_manager and self.admin_id:
            norm_force_group = self.create_normalized_max_force_group()

            if norm_force_group:
                container_layout.addWidget(norm_force_group)

        # 7) Bottom buttons row: left side "Export Report" and "Show Repetitions"; right side "Close"
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
        bottom_layout.addWidget(close_button)
        container_layout.addLayout(bottom_layout)

        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

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
            "iit": "Intermittent Isometric Test",
            "iirt": "Intermittent Isometric Resistance Test",
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

    def create_performance_prediction_group(self):
        """Create a group for performance prediction"""
        # Create group box
        group_box = QGroupBox("Performance Prediction")
        group_box.setStyleSheet("""
                QGroupBox::title {
                font-size: 14pt;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(group_box)
        layout.setContentsMargins(10, 20, 10, 10)

        # First, check if this is an AO test - if not, don't show prediction
        test_type = self.db_data.get('test_type', '').lower()
        if test_type != 'ao':
            info_label = QLabel("Performance prediction is only available for All-Out tests.")
            info_label.setWordWrap(True)
            info_label.setStyleSheet("font-size: 13px; color: #888888; margin: 5px;")
            layout.addWidget(info_label)
            return group_box

        try:
            # Get the test ID from the database data
            test_id = self.db_data.get('id')
            if test_id:
                # Initialize the predictor
                predictor = IRCRAPredictor(
                    # model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
                )

                # Check if models are loaded (trained)
                if predictor.pca_data is not None and predictor.svr_model is not None:
                    # Predict IRCRA grade
                    predicted_ircra = predictor.predict_ircra(test_id=test_id, model_type='svr')

                    # Create prediction label
                    prediction_label = QLabel(
                        f"With your predispositions, you could be able to climb IRCRA grade: {predicted_ircra}")
                    prediction_label.setWordWrap(True)
                    prediction_label.setStyleSheet("font-size: 14px; margin: 5px;")
                    layout.addWidget(prediction_label)

                    # Add additional analysis
                    current_ircra = int(self.participant_info.get('ircra'))
                    if current_ircra and isinstance(current_ircra, (int, float)) and isinstance(predicted_ircra,
                                                                                                (int, float)):
                        difference = predicted_ircra - current_ircra

                        if abs(difference) <= 1:
                            analysis_text = "Your current performance level aligns perfectly with your physical metrics."
                        elif difference > 1:
                            analysis_text = f"Your test results suggest potential for higher performance. You might be able to climb {abs(difference)} grade{'s' if abs(difference) > 1 else ''} harder with optimal technique and training."
                        else:
                            analysis_text = f"Your climbing experience and technique likely compensate for physical metrics, allowing you to climb {abs(difference)} grade{'s' if abs(difference) > 1 else ''} harder than predicted."

                        analysis_label = QLabel(analysis_text)
                        analysis_label.setWordWrap(True)
                        analysis_label.setStyleSheet("font-size: 13px; margin: 5px;")
                        layout.addWidget(analysis_label)
                        self.ircra_prediction = {'test_type': self.db_data.get('test_type'),
                                                 'predicted_ircra': predicted_ircra,
                                                 'analysis_text': analysis_text,
                                                 'current_ircra': current_ircra,
                                                 'difference': difference}


                else:
                    # Models not loaded message
                    error_label = QLabel(
                        "Performance prediction not available - prediction models not found or not trained.")
                    error_label.setWordWrap(True)
                    error_label.setStyleSheet("font-size: 13px; color: #888888; margin: 5px;")
                    layout.addWidget(error_label)
            else:
                # No test ID message
                error_label = QLabel("Performance prediction not available for this test.")
                error_label.setWordWrap(True)
                error_label.setStyleSheet("font-size: 13px; color: #888888; margin: 5px;")
                layout.addWidget(error_label)

        except Exception as e:
            # Error message
            error_label = QLabel(f"Unable to generate performance prediction: {str(e)}")
            error_label.setWordWrap(True)
            error_label.setStyleSheet("font-size: 13px; color: #888888; margin: 5px;")
            layout.addWidget(error_label)

        return group_box

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
        arm_text = "Dominant" if arm_tested == "d"\
            else "Non-dominant" if arm_tested == "nd" else arm_tested

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
            rep_results_db = self.db_data.get('rep_results', "")
            try:
                if isinstance(rep_results_db, list):
                    rep_results = rep_results_db
                else:
                    rep_results = json.loads(rep_results_db)
            except json.JSONDecodeError:
                # Fallback for old format
                rep_results = eval(rep_results_db)

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

        # Save the Normalized Max Force Graph if available
        norm_force_graph_filepath = None
        if hasattr(self, 'norm_force_fig') and self.norm_force_fig:
            norm_force_graph_filepath = f"{self.db_data['id']}_norm_force_graph.png"
            self.norm_force_fig.savefig(norm_force_graph_filepath, format='png')

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
                norm_force_graph_image_path=norm_force_graph_filepath,  # Normalized Max Force Graph
                ircra_prediction=self.ircra_prediction,  # IRCRA prediction data
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

        try:
            if isinstance(rep_results_db, list):
                rep_results = rep_results_db
            else:
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
            
        norm_force_graph_filepath = f"{self.db_data['id']}_norm_force_graph.png"
        if os.path.exists(norm_force_graph_filepath):
            try:
                os.remove(norm_force_graph_filepath)
            except Exception as e:
                print("Error deleting normalized max force graph file:", e)
            
        rep_graph_filepath = f"{self.db_data['id']}_rep_graph.png"
        if os.path.exists(rep_graph_filepath):
            try:
                os.remove(rep_graph_filepath)
            except Exception as e:
                print("Error deleting rep graph file:", e)
            
        if hasattr(self, 'rep_window') and self.rep_window is not None:
            self.rep_window.close()  # This will trigger rep_window's closeEvent.
        
        super().closeEvent(event)

    def create_normalized_max_force_group(self):
        """
        Creates a QGroupBox containing the normalized max force plots
        using the plot_normalized_max_force function from graphs_generator.py

        Returns:
            QGroupBox containing the plots or None if plots couldn't be created
        """
        try:
            # Call the plotting function to get the figure
            fig = plot_normalized_max_force(
                self.climber_manager,
                self.test_manager,
                self.admin_id,
                current_test_id=self.db_data['id']
            )

            # If the function didn't return a figure, return None
            if fig is None:
                print("Could not create normalized max force plots: insufficient data")
                return None

            # Create an instance of the figure canvas
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(800, 400)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Create a group box and add the canvas
            group_box = QGroupBox("Normalized Max Force Analysis")
            group_box.setStyleSheet("""
                QGroupBox::title {
                    font-size: 14pt;
                    font-weight: bold;
                }
            """)
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            group_box.setLayout(layout)

            # Store the figure for PDF export
            self.norm_force_fig = fig

            return group_box

        except Exception as e:
            print(f"Error creating normalized max force plots: {e}")
            import traceback
            traceback.print_exc()
            return None
