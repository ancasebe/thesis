"""
Data communication and visualization module for the Climbing Testing Application.

This module defines the CombinedDataCommunicator class which handles real-time
data acquisition, visualization, and storage during climbing tests. It manages
force sensor and NIRS data streams, controls test states and timing, and finalizes
test data for storage in the database.

Key functionalities:
- Real-time data acquisition from force sensors and NIRS devices
- Interactive visualization of collected data streams
- Management of test timing, states, and protocols
- Data processing and preparation for storage
- Test finalization and data export

The CombinedDataCommunicator serves as the core interface during test execution,
providing visual feedback to the tester and processing data for later analysis.
"""
import time
import numpy as np
import pyqtgraph as pg
import json
from PySide6.QtWidgets import QMainWindow, QWidget, QMessageBox, QLabel, QDialog, QApplication, QVBoxLayout
from PySide6.QtCore import Signal, QTimer, Qt

from gui.test_page.data_gen import DataGenerator
from gui.test_page.evaluations.nirs_evaluation import NIRSEvaluation
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics, find_test_interval
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.results_page.report_window import TestReportWindow
from gui.test_page.feather_logger import FeatherBinaryLogger


class CombinedDataCommunicator(QMainWindow):
    """
    A class for visualizing, logging, and processing sensor data during climbing tests.
    
    This class provides real-time visualization of force and NIRS data, manages
    test protocols (countdown, pull/rest cycles), logs collected data, and evaluates
    test performance. It serves as the main interface during test execution.
    
    Attributes:
        newData: Signal emitted when new data is received from sensors
        dg: Data generator for sensor data acquisition
        admin_id: ID of the administrator running the test
        climber_id: ID of the climber being tested
        arm_tested: Arm being tested (converted to "d" or "nd")
        data_type: Type of data being collected ("force", "nirs", or "force_nirs")
        test_type: Type of test being conducted ("mvc", "ao", "iit", "sit")
        window_size: Time window for data visualization (in seconds)
        nirs_buffer: Buffer for smoothing NIRS data
        force_data: Dictionary containing force timestamps and values
        nirs_data: Dictionary containing NIRS timestamps and values
        force_file: Logger for force data
        nirs_file: Logger for NIRS data
        timestamp: Start time of the test
        acquisition_started: Flag indicating if data acquisition has started
        finalized: Flag indicating if the test has been finalized
        state: Current test state ("pre", "pull", "rest", "run", etc.)
        remaining: Remaining time in current state
    """

    # Create signal for new data
    newData = Signal(str)

    def __init__(self, data_generator: DataGenerator, admin_id, climber_id, arm_tested, window_size=60, auto_start=True,
                 data_type="force", test_type="ao", parent=None):
        """
        Initialize the data communicator with test parameters and setup UI components.
        
        Args:
            data_generator: DataGenerator instance for data acquisition
            admin_id: ID of the administrator running the test
            climber_id: ID of the climber being tested
            arm_tested: Which arm is being tested ("Dominant" or "Non-dominant")
            window_size: Time window for visualization in seconds (default: 60)
            auto_start: Whether to start acquisition automatically (default: True)
            data_type: Type of data to collect ("force", "nirs", "force_nirs")
            test_type: Type of test to conduct ("mvc", "ao", "iit", "sit")
            parent: Parent widget (default: None)
        """
        super().__init__(parent)
        self.dg = data_generator
        self.admin_id = admin_id
        self.climber_id = climber_id
        if arm_tested == "Dominant":
            self.arm_tested = "d"
        elif arm_tested == "Non-dominant":
            self.arm_tested = "nd"
        else:
            self.arm_tested = "-"
        self.data_type = data_type
        self.test_type = test_type
        self.window_size = window_size
        
        # Add NIRS smoothing parameters
        self.nirs_buffer = []  # Buffer to hold recent NIRS values for smoothing

        # Connect the signal to the handler
        self.newData.connect(self.handle_new_data)

        # storage and file loggers
        self.force_data = {"timestamps": [], "values": []}
        self.nirs_data  = {"timestamps": [], "values": []}
        self.force_file = None
        self.nirs_file  = None
        self.timestamp = None
        self.acquisition_started = False
        self.finalized  = False

        self._make_counter_window()

        if self.data_type == "force":
            self.pre_delay = 5  # seconds before first beep
        else:
            self.pre_delay = 15
        self.pull_dur = 7  # default for ao
        self.rest_dur = 3
        self.state = None  # will be "pre", "pull", "rest", or "run" (for sit/mvc)
        self.remaining = None
        # timers
        self.countdown_timer = QTimer(self, interval=1000)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)
        self.segment_timer = QTimer(self)  # single-shot per segment

        self._setup_ui()
        
        # Connect callbacks when starting acquisition
        if auto_start:
            self.start_acquisition()

    def _setup_ui(self):
        """
        Set up the UI components including plots and visualization settings.
        
        Creates the main plot widget with appropriate axes, ranges, and legends
        for force and/or NIRS data visualization. For force_nirs mode, configures
        a dual-axis plot with synchronized views.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        self.plot = pg.PlotWidget()
        self.plot.enableAutoRange(False)
        layout.addWidget(self.plot)
        self.plot.setXRange(-self.window_size, 0)
        # fix left Y axis for Force
        if self.data_type in ("force", "force_nirs"):
            self.plot.setYRange(0, 80, padding=0.05)

        legend = self.plot.addLegend()
        legend.setParentItem(self.plot.getViewBox())
        legend.anchor((0.5, 0), (0.5, 0))

        if self.data_type in ("force", "force_nirs"):
            self.plot.getAxis('left').setLabel("Force (kg)")
            self.force_curve = self.plot.plot([], [], pen='y')
            self.force_text = pg.TextItem("", anchor=(0, 1))
            self.plot.addItem(self.force_text)
            self.force_text.setText("Force: 0.0 kg")
            self.force_text.setPos(-self.window_size, 80)  # top-left of your force axis

        if self.data_type in ("nirs", "force_nirs"):
            self.plot.showAxis('right')
            self.plot.getAxis('right').setLabel("NIRS (%)")
            self.nirs_view = pg.ViewBox()
            self.nirs_view.enableAutoRange(False)
            self.plot.scene().addItem(self.nirs_view)
            self.plot.getAxis('right').linkToView(self.nirs_view)
            self.nirs_view.setYRange(0, 100, padding=0.05)
            self.nirs_view.setXRange(-self.window_size, 0, padding=0)
            self.nirs_curve = pg.PlotCurveItem(pen='r')
            self.nirs_view.addItem(self.nirs_curve)
            self.nirs_text = pg.TextItem("", anchor=(1, 1))
            self.nirs_view.addItem(self.nirs_text)
            self.nirs_text.setText("NIRS: 0.0 %")
            self.nirs_text.setPos(0, 100)

            # Connect the updateViews function to the plot's X range change signal
            self.plot.getViewBox().sigXRangeChanged.connect(self._update_views)
            # Initialize the views to match initially
            self._update_views()

    def _make_counter_window(self):
        """
        Create a floating timer window to display test state and countdown.
        
        Creates a dialog with two labels:
        - A state label showing the current test phase (preparing, pull, rest)
        - A large countdown/timer label showing seconds remaining or elapsed
        """
        self.counter_win = QDialog(self)
        self.counter_win.setWindowTitle("Timer")
        self.state_lbl = QLabel("Preparing...", self.counter_win)
        self.state_lbl.setAlignment(Qt.AlignCenter)
        state_font = self.state_lbl.font()
        state_font.setPointSize(14)
        state_font.setBold(True)
        self.state_lbl.setFont(state_font)
        
        self.lbl = QLabel("0", self.counter_win)
        self.lbl.setAlignment(Qt.AlignCenter)
        font = self.lbl.font()
        font.setPointSize(32)
        self.lbl.setFont(font)
        
        layout = QVBoxLayout(self.counter_win)
        layout.addWidget(self.state_lbl)
        layout.addWidget(self.lbl)
        self.counter_win.setLayout(layout)
        self.counter_win.setFixedSize(250, 150)

    def _configure_durations(self):
        """
        Configure test durations based on the selected test type.
        
        Sets the pull and rest durations for different test protocols:
        - AO: 7 seconds pull, 3 seconds rest
        - IIT: 6 seconds pull, 4 seconds rest
        - MVC/SIT: Continuous pull without defined durations
        """
        if self.test_type == "ao":
            self.pull_dur, self.rest_dur = 7, 3
        elif self.test_type == "iit":
            self.pull_dur, self.rest_dur = 6, 4
        elif self.test_type == "mvc":
            self.pull_dur, self.rest_dur = None, None
        elif self.test_type == "sit":
            self.pull_dur, self.rest_dur = None, None

    def _on_countdown_tick(self):
        """
        Handle timer ticks for countdown and test state management.
        
        This method updates the countdown display, manages transitions between
        test states (pre-test, pull, rest), and emits audio cues at state transitions.
        It handles different test protocols including intermittent tests (AO, IIT) 
        and continuous tests (MVC, SIT).
        """
        self.remaining -= 1
        self.lbl.setText(str(self.remaining))
        if self.remaining <= 0:
            self.countdown_timer.stop()
            # time for a beep and next action
            QApplication.beep()
            if self.state == "pre":
                # test is now truly started
                if self.test_type in ("ao", "iit"):
                    self.state = "pull"
                    self.remaining = self.pull_dur
                    self.state_lbl.setText("Pull")
                    # start segment countdown
                    self.countdown_timer.start()
                elif self.test_type == "mvc":
                    # Change to countdown for 5 seconds instead of using singleShot
                    self.state = "mvc_countdown"
                    self.remaining = 5
                    self.state_lbl.setText("Pull")
                    self.countdown_timer.start()
                elif self.test_type == "sit":
                    # just run an elapsed timer
                    self.state = "run"
                    self.remaining = 0
                    self.state_lbl.setText("Pull")
                    # restart for elapsed time display
                    self.countdown_timer.start()
            elif self.state == "mvc_countdown":
                # After the 5-second MVC countdown, switch to run state
                self.state = "run"
                self.remaining = 0
                self.state_lbl.setText("Pull")
                # restart for elapsed time display
                self.countdown_timer.start()
            elif self.state in ("pull", "rest"):
                # toggle pull/rest
                if self.state == "pull":
                    self.state = "rest"
                    self.remaining = self.rest_dur
                    self.state_lbl.setText("Rest")
                else:
                    self.state = "pull"
                    self.remaining = self.pull_dur
                    self.state_lbl.setText("Pull")
                # beep already emitted, restart countdown
                self.countdown_timer.start()
            elif self.state == "run":
                # in sit or mvc elapsed mode: just keep counting up
                self.remaining += 1
                self.countdown_timer.start()
        # else: simply updating the label each second

    def _update_views(self):
        """
        Synchronize the NIRS view with the main plot view.
        
        Ensures that when the main plot's X-axis changes, the secondary NIRS view
        is updated to maintain the same X-axis range for synchronized visualization.
        """
        self.nirs_view.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self.nirs_view.linkedViewChanged(self.plot.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        """
        Initialize data acquisition, file logging, and start the test protocol.
        
        This method:
        1. Clears previous data and initializes file loggers
        2. Sets up callback functions for data streams
        3. Configures test durations based on test type
        4. Starts the pre-test countdown
        5. Launches sensor data acquisition
        """
        if not self.acquisition_started:
            # Clear data when starting acquisition
            self.dg.clear_data()

            self.timestamp = time.time()
            if self.data_type in ("force", "force_nirs"):
                self.force_file = FeatherBinaryLogger(prefix=f"{self.test_type}_force", timestamp=self.timestamp)
            if self.data_type in ("nirs", "force_nirs"):
                self.nirs_file  = FeatherBinaryLogger(prefix=f"{self.test_type}_nirs",  timestamp=self.timestamp)

            # Set up callback functions that emit signals
            force_callback = lambda s: self.newData.emit(s) if self.data_type in ("force", "force_nirs") else None
            nirs_callback = lambda s: self.newData.emit(s) if self.data_type in ("nirs", "force_nirs") else None
            
            # Set callbacks for data generator
            self.dg.set_callbacks(
                force_callback=force_callback,
                nirs_callback=nirs_callback
            )
            
            # Resume forwarding data to our UI
            self.dg.resume_data_forwarding()

            # Then kick off our test‐timer sequence:
            self._configure_durations()
            self.state = "pre"
            self.remaining = self.pre_delay
            self.state_lbl.setText("Test starts in:")
            self.counter_win.show()
            self.lbl.setText(str(self.remaining))
            QApplication.beep()  # optional "ready" click
            self.countdown_timer.start()
        
        # launch sensors if they're not already running
        self.dg.start(nirs=(self.data_type != 'force'), force=(self.data_type != 'nirs'))
        self.acquisition_started = True

    def handle_new_data(self, data_str):
        """
        Process incoming sensor data, store it, and update visualizations.
        
        Args:
            data_str: String containing sensor data in format "sensorID_timestamp_value"
            
        Parses the data string, identifies the sensor type (force=1, NIRS=2),
        stores the data in appropriate containers, logs it to files, and
        triggers plot updates.
        """
        parts = data_str.split('_')
        if len(parts) < 3:
            return
        
        try:
            sid = parts[0]
            ts = float(parts[1])
            val = float(parts[2])
        except ValueError:
            return
        
        if sid == '1':  # force
            self.force_data['timestamps'].append(ts)
            self.force_data['values'].append(val)
            if self.force_file:
                self.force_file.log(ts, val)
            # Update plot for force data
            if self.data_type in ("force", "force_nirs"):
                self.update_plot()
        elif sid == '2':  # NIRS SMO2 data (only process sensor ID 2)
            smoothed_val = self.smooth_nirs_data(float(val))
            self.nirs_data['timestamps'].append(ts)
            self.nirs_data['values'].append(smoothed_val)
            # Log only SMO2 values
            if self.nirs_file:
                self.nirs_file.log(ts, val)
            # Update plot for NIRS SMO2 data
            if self.data_type in ("nirs", "force_nirs"):
                self.update_plot()

    def update_plot(self):
        """
        Update the visualization plots with current data.
        
        Updates force and/or NIRS plot curves using the collected data,
        adjusting the x-axis to be relative to the most recent timestamp.
        Limits the displayed data to the configured window size and
        updates the current value text displays.
        """
        if self.force_data['timestamps']:
            t0 = self.force_data['timestamps'][-1]
            x = np.array(self.force_data['timestamps']) - t0
            y = np.array(self.force_data['values'])
            mask = (x >= -self.window_size)
            self.force_curve.setData(x[mask], y[mask])
            # position in top-left of left axis
            self.force_text.setText(f"Force: {y[-1]:.1f} kg")
            self.force_text.setPos(-self.window_size, 80)

        if self.nirs_data['timestamps']:
            t0 = self.nirs_data['timestamps'][-1]
            x = np.array(self.nirs_data['timestamps']) - t0
            y = np.array(self.nirs_data['values'])
            mask = (x >= -self.window_size)
            self.nirs_curve.setData(x[mask], y[mask])
            # position in top-right of right axis
            self.nirs_text.setText(f"NIRS: {y[-1]:.1f} %")
            self.nirs_text.setPos(0, 100)

        # Update the NIRS view to match the main plot's view
        if self.data_type in ("nirs", "force_nirs"):
            self._update_views()

    def stop_acquisition(self):
        """
        Stop data acquisition, timers, and finalize the test.
        
        Pauses data forwarding, stops countdown timers, hides the counter window,
        stops force sensors, flushes log files, and triggers test finalization
        to process and save the results.
        """
        # Pause data forwarding to UI
        self.dg.pause_data_forwarding()

        self.countdown_timer.stop()
        self.segment_timer.stop()
        self.counter_win.hide()
        
        # Stop sensors
        self.dg.stop(nirs=False, force=True)
        
        # Flush logs
        if self.force_file:
            self.force_file.flush()
        if self.nirs_file:
            self.nirs_file.flush()

        self.finalize_acquisition()
        
        # Clear data
        self.dg.clear_data()

    def finalize_acquisition(self):
        """
        Process test data, evaluate performance, and save results to the database.
        
        This method:
        1. Evaluates test metrics using appropriate evaluators
        2. Prepares results for database storage
        3. Saves test results to the database
        4. Creates and displays a test report window
        
        Handles different evaluations based on the data type (force only, NIRS only,
        or combined force and NIRS).
        """
        if not self.finalized:
            if self.data_type == "force":
                force_evaluator = ForceMetrics(file_path=self.force_file.actual_filename,
                                               test_type=self.test_type,
                                               threshold_ratio=0.1)
                test_results, rep_results = force_evaluator.evaluate()
                number_of_reps = len(rep_results)
                nirs_results = None

            elif self.data_type == "nirs":
                nirs_results = {'Only NIRS': 'TODO'}
                test_results = rep_results = number_of_reps = None
            elif self.data_type == "force_nirs":
                force_evaluator = ForceMetrics(file_path=self.force_file.actual_filename,
                                               test_type=self.test_type)
                test_results, rep_results = force_evaluator.evaluate()
                start_time, test_start_abs, test_end_abs = find_test_interval(self.force_file.actual_filename)
                test_start_rel = test_start_abs - start_time
                test_end_rel = test_end_abs - start_time
                nirs_eval = NIRSEvaluation(self.nirs_file.actual_filename, smoothing_window=25,
                                       baseline_threshold=0.1, recovery_tolerance=1.0)
                try:
                    nirs_results = nirs_eval.evaluate(start_time, test_start_rel, test_end_rel)

                except TypeError as e:
                    print(f"Error in NIRS evaluation: {e}")
                    # Provide fallback results if NIRS evaluation fails
                    nirs_results = {}
                number_of_reps = len(rep_results)

            else:
                QMessageBox.warning(self, "Error", "Unknown data type; cannot generate report")
                raise ValueError("Unknown data type; cannot generate report.")

            test_db_manager = ClimbingTestManager()
            force_file = self.force_file.filename if self.force_file else ''
            nirs_file = self.nirs_file.filename if self.nirs_file else ''
            
            # Convert results to JSON strings instead of using str()
            test_results_json = json.dumps(test_results) if test_results is not None else "null"
            nirs_results_json = json.dumps(nirs_results) if nirs_results is not None else "null"
            rep_results_json = json.dumps(rep_results) if rep_results is not None else "null"
            
            db_data_save = {
                'arm_tested': self.arm_tested,
                'data_type': self.data_type,
                'test_type': self.test_type,
                'timestamp': self.timestamp,
                'number_of_reps': number_of_reps,
                'force_file': force_file,
                'nirs_file': nirs_file,
                'test_results': test_results_json,
                'nirs_results': nirs_results_json,
                'rep_results': rep_results_json
            }
            # print("Data to DB: ", db_data_save)
            test_id = test_db_manager.add_test_result(admin_id=self.admin_id,
                                                      participant_id=self.climber_id,
                                                      db_data=db_data_save)

            QMessageBox.information(self, "Test saving", "Test was saved successfully.")
            self.finalized = True

            climber_db_manager = ClimberDatabaseManager()
            climber_data = climber_db_manager.get_user_data(self.admin_id, self.climber_id)

            if not climber_data:
                climber_data = {
                    "name": "Unknown",
                    "surname": "Unknown",
                    "email": "N/A",
                    "gender": "N/A",
                    "dominant_arm": "N/A"
                }
            db_data_save['id'] = test_id
            if self.force_file.actual_filename:
                db_data_save['force_file'] = self.force_file.actual_filename
            if self.nirs_file.actual_filename:
                db_data_save['nirs_file'] = self.nirs_file.actual_filename
            report_window = TestReportWindow(
                participant_info=climber_data,
                db_data=db_data_save,
                admin_id=self.admin_id,
                climber_db_manager=climber_db_manager,
                test_db_manager=test_db_manager,
                parent=self
            )
            report_window.show()
            test_db_manager.close_connection()
            climber_db_manager.close()

    def closeEvent(self, event):
        """
        Handle the window close event with proper cleanup.
        
        Args:
            event: The close event to be handled
            
        Ensures that acquisition is stopped and data forwarding is paused
        before accepting the close event.
        """
        # Safely stop acquisition
        if not self.finalized:
            self.stop_acquisition()
        else:
            # If already finalized, just make sure to pause callbacks
            self.dg.pause_data_forwarding()
        
        event.accept()

    def smooth_nirs_data(self, value, nirs_smoothing_window=7):
        """
        Apply moving average smoothing to NIRS data in real-time.
        
        Args:
            value: The new NIRS value to add to the buffer
            nirs_smoothing_window: Size of the moving average window (default: 7)
            
        Returns:
            float: The smoothed NIRS value based on recent values
            
        Implements a simple moving average filter to reduce noise in NIRS data
        while preserving important trends.
        """
        # Add the new value to the buffer
        self.nirs_buffer.append(value)

        # Keep only the most recent values based on the window size
        if len(self.nirs_buffer) > nirs_smoothing_window:
            self.nirs_buffer.pop(0)

        # Calculate the smoothed value (simple moving average)
        if len(self.nirs_buffer) > 0:
            return sum(self.nirs_buffer) / len(self.nirs_buffer)
        else:
            return value