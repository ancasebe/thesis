import os
import queue
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtCore import Signal, QThread

from gui.test_page.data_gen import DataGenerator
from gui.test_page.evaluations.nirs_evaluation import NIRSEvaluation
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics, find_test_interval
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.results_page.report_window import TestReportWindow
from gui.test_page.feather_logger import FeatherBinaryLogger


class CombinedDataCommunicator(QMainWindow):
    """
    CombinedDataCommunicator visualizes and logs sensor data via direct callbacks and Qt signals.

    It updates plots and logs data (using file loggers) that is being pushed by the
    DataGenerator. All sensor communicators (Serial and Bluetooth) are assumed to be started
    externally (via DataGenerator). New readings come in as 'sensorID_timestamp_value' strings.
    """
    # Create signal for new data
    newData = Signal(str)

    def __init__(self, data_generator: DataGenerator, admin_id, climber_id, arm_tested, window_size=60, auto_start=True,
                 data_type="force", test_type="ao", parent=None):
        super().__init__(parent)
        self.dg = data_generator
        self.admin_id = admin_id
        self.climber_id = climber_id
        if arm_tested == "Dominant":
            self.arm_tested = "D"
        elif arm_tested == "Non-dominant":
            self.arm_tested = "N"
        else:
            self.arm_tested = "-"
        self.data_type = data_type
        self.test_type = test_type
        self.window_size = window_size

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

        self._setup_ui()
        
        # Connect callbacks when starting acquisition
        if auto_start:
            self.start_acquisition()

    def _setup_ui(self):
        """Sets up the UI components and dual-axis plot layout."""
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

            # self.force_text = pg.TextItem(anchor=(0, 1))
            # self.plot.addItem(self.force_text)
            # legend.addItem(self.force_curve, "Force [kg]")

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
            # self.nirs_text = pg.TextItem(anchor=(1, 1))  # top-right corner
            # self.nirs_view.addItem(self.nirs_text)
            # legend.addItem(self.nirs_curve, "NIRS (%)")


    def _update_views(self):
        """Synchronize the geometry of the secondary NIRS view with the main view."""
        self.nirs_view.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self.nirs_view.linkedViewChanged(self.plot.getViewBox(), self.nirs_view.XAxis)

    def start_acquisition(self):
        """
        Initializes file loggers, resets internal data containers, and connects
        callbacks to start receiving data
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
            
            # launch sensors if they're not already running
            self.dg.start(nirs=(self.data_type != 'force'), force=(self.data_type != 'nirs'))
            self.acquisition_started = True

    def handle_new_data(self, data_str):
        """
        Slot that handles new data coming from the worker thread.
        The data string is expected in the format "sensorID_timestamp_value".
        Update internal data arrays and call updatePlot.
        """
        # parts = data_str.split("_")
        # print('Data to plot:', data_str)
        # if len(parts) < 3:
        #     return
        # try:
        #     sensor = int(parts[0])
        #     timestamp = float(parts[1])
        #     value = float(parts[2])
        # except ValueError:
        #     return

        sid, ts, val = data_str.split('_')
        ts, val = float(ts), float(val)
        if sid == '1':  # force
            print('plotting Force:', ts, val)
            self.force_data['timestamps'].append(ts)
            self.force_data['values'].append(val)
            if self.force_file:
                self.force_file.log(ts, val)
        else:           # NIRS channels: '2','3','4'
            print('plotting NIRS:', ts, val)
            self.nirs_data['timestamps'].append(ts)
            self.nirs_data['values'].append(val)
            if self.nirs_file:
                self.nirs_file.log(ts, val)
        self.update_plot()

    def update_plot(self):
        """
        Update the plot curves using the data that has been collected.
        We adjust the x-axis to be relative to the most recent timestamp.
        """
        if self.force_data['timestamps']:
            t0 = self.force_data['timestamps'][-1]
            x = np.array(self.force_data['timestamps']) - t0
            y = np.array(self.force_data['values'])
            mask = (x >= -self.window_size)
            self.force_curve.setData(x[mask], y[mask])
            # position in top-left of left axis
            self.force_text.setText(f"Force: {y[-1]:.1f} kg")
            # yr_left = self.plot.getViewBox().viewRange()[1][1]
            self.force_text.setPos(-self.window_size, 80)

        if self.nirs_data['timestamps']:
            t0 = self.nirs_data['timestamps'][-1]
            x = np.array(self.nirs_data['timestamps']) - t0
            y = np.array(self.nirs_data['values'])
            mask = (x >= -self.window_size)
            self.nirs_curve.setData(x[mask], y[mask])
            # position in top-right of right axis
            self.nirs_text.setText(f"NIRS: {y[-1]:.1f} %")
            # yr_right = self.nirs_view.viewRange()[1][1]
            self.nirs_text.setPos(0, 100)

    def stop_acquisition(self):
        """
        Stops the worker thread, logs any remaining data, stops sensor connections,
        and finalizes the test evaluation.
        """
        # Pause data forwarding to UI
        self.dg.pause_data_forwarding()
        
        # Stop sensors
        self.dg.stop(nirs=True, force=True)
        
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
        Finalizes the test by computing evaluation metrics, saving results to the database,
        and opening a report window. This method is called once acquisition is stopped.
        """
        if not self.finalized:
            if self.data_type == "force":
                force_evaluator = ForceMetrics(file_path=self.force_file.filename,
                                               test_type=self.test_type,
                                               threshold_ratio=0.1)
                test_results, rep_results = force_evaluator.evaluate()
                number_of_reps = len(rep_results)
                nirs_results = None
                print('number of reps:', number_of_reps)
                print("Force Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            elif self.data_type == "nirs":
                nirs_results = {'Only NIRS': 'TODO'}
                test_results = rep_results = number_of_reps = None
            elif self.data_type == "force_nirs":
                force_evaluator = ForceMetrics(file_path=self.force_file.filename,
                                               test_type=self.test_type)
                test_results, rep_results = force_evaluator.evaluate()
                start_time, test_start_abs, test_end_abs = find_test_interval(self.force_file.filename)
                test_start_rel = test_start_abs - start_time
                test_end_rel = test_end_abs - start_time
                nirs_eval = NIRSEvaluation(self.nirs_file.filename, smoothing_window=25,
                                           baseline_threshold=0.1, recovery_tolerance=1.0)
                print(start_time, test_start_rel, test_end_rel)
                nirs_results = nirs_eval.evaluate(start_time, test_start_rel, test_end_rel)
                print("NIRS Evaluation Results:")
                print(nirs_results)
                number_of_reps = len(rep_results)
                print('number of reps:', number_of_reps)
                print("Force Evaluation Results:")
                print(test_results)
                print("Repetition-by-Repetition Metrics:", rep_results)
            else:
                QMessageBox.warning(self, "Error", "Unknown data type; cannot generate report")
                print(self.data_type)
                raise ValueError("Unknown data type; cannot generate report.")

            db_manager = ClimbingTestManager()
            force_file = self.force_file.filename if self.force_file else ''
            nirs_file = self.nirs_file.filename if self.nirs_file else ''
            db_data_save = {
                'arm_tested': self.arm_tested,
                'data_type': self.data_type,
                'test_type': self.test_type,
                'timestamp': self.timestamp,
                'number_of_reps': number_of_reps,
                'force_file': force_file,
                'nirs_file': nirs_file,
                'test_results': str(test_results),
                'nirs_results': str(nirs_results),
                'rep_results': str(rep_results)
            }
            print("data to db: ", db_data_save)
            test_id = db_manager.add_test_result(admin_id=str(self.admin_id),
                                                 participant_id=str(self.climber_id),
                                                 db_data=db_data_save)
            db_manager.close_connection()
            QMessageBox.information(self, "Test saving", "Test was saved successfully.")
            self.finalized = True

            climber_db = ClimberDatabaseManager()
            climber_data = climber_db.get_user_data(self.admin_id, self.climber_id)
            climber_db.close()
            if not climber_data:
                climber_data = {
                    "name": "Unknown",
                    "surname": "Unknown",
                    "email": "N/A",
                    "gender": "N/A",
                    "dominant_arm": "N/A"
                }
            db_data_save['id'] = test_id
            self.report_window = TestReportWindow(
                participant_info=climber_data,
                db_data=db_data_save,
                parent=self
            )
            self.report_window.show()

    def closeEvent(self, event):
        """
        Handles the window close event by stopping timers, finalizing acquisition,
        and closing any open log files.
        """
        # Safely stop acquisition
        if not self.finalized:
            self.stop_acquisition()
        else:
            # If already finalized, just make sure to pause callbacks
            self.dg.pause_data_forwarding()
        
        event.accept()