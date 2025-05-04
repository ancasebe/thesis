# statistics_page.py
import os
import joblib
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QMessageBox, QGroupBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal

from gui.statistics.ircra_prediction_model import PredictionIRCRA

# Define model directory and file paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file paths
PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_pipeline.joblib')
SVR_MODEL_PATH = os.path.join(MODEL_DIR, 'svr_model.joblib')
PCR_PLOT_PATH = os.path.join(MODEL_DIR, 'pcr_prediction.png')
SVR_PLOT_PATH = os.path.join(MODEL_DIR, 'svr_prediction.png')


class ModelTrainingThread(QThread):
    """Thread for training IRCRA prediction models without freezing the UI"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    training_complete = Signal(dict)  # Modified to pass plot paths
    training_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.mse_pcr = None
        self.mse_svr = None
        # self.admin_id = admin_id

    def run(self):
        try:
            # Update progress and status
            self.progress_updated.emit(5)
            self.status_updated.emit("Initializing model training...")

            # Create the prediction model
            model = PredictionIRCRA()
            
            # More detailed progress reporting
            self.progress_updated.emit(10)
            self.status_updated.emit("Creating model instance...")
            
            # Load and prepare data - ADD TRACEBACK FOR ERRORS
            try:
                self.progress_updated.emit(20)
                self.status_updated.emit("Loading test data...")
                df = model.load_data()
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                self.training_failed.emit(f"Error loading data: {str(e)}\n\n{error_details}")
                return
            
            # Check if thread was requested to stop
            if self.isInterruptionRequested():
                return

            # Prepare features
            self.progress_updated.emit(40)
            self.status_updated.emit("Preparing features...")
            df_selected, X, y = model.prepare_features(df)
            
            # Create correlation matrix
            self.status_updated.emit("Generating correlation matrix...")
            corr_matrix_path = model.corr_matrix(df_selected)

            if self.isInterruptionRequested():
                return

            # Perform PCA analysis
            self.progress_updated.emit(60)
            self.status_updated.emit("Performing PCA analysis...")
            pca_df = model.pca_df(X_raw=X, y=y)
            X_pca, y_pca = model.prepare_pca_features(pca_df)

            if self.isInterruptionRequested():
                return

            # Train SVR model
            self.progress_updated.emit(80)
            self.status_updated.emit("Training SVR model...")
            self.mse_pcr, self.mse_svr, svr_plot_path, pcr_plot_path = model.train_best_svr(X_pca, y_pca)

            # Store the best model accuracy (100 - validation_mse is a rough approximation of accuracy %)
            # if isinstance(best_regr, pd.Series) and 'validation_mse' in best_regr:
            # Calculate approximate accuracy - this is a simplification and might need adjustment
            # for your specific context
            # self.best_accuracy = 100 - best_regr['validation_mse']
            
            # Get plot paths for later display
            plot_paths = {
                'corr_matrix': corr_matrix_path,
                'pca_variance': os.path.join(model.plots_dir, 'pca_explained_variance.png'),
                'pca_scatter': os.path.join(model.plots_dir, 'pca_scatter.png'),
                'svr_accuracy': svr_plot_path,
                'pcr_accuracy': pcr_plot_path,
                'svr_scatter': os.path.join(model.plots_dir, 'svr_sel_scatter.png'),
                'pcr_scatter': os.path.join(model.plots_dir, 'pcr_sel_scatter.png')
            }

            self.progress_updated.emit(100)
            self.status_updated.emit("Training complete!")

            # Signal completion with plot paths
            self.training_complete.emit(plot_paths)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in training: {str(e)}\n{error_trace}")
            self.training_failed.emit(f"Error: {str(e)}")


class StatisticsPage(QWidget):
    """Page for training and viewing IRCRA prediction models"""

    def __init__(self, admin_id):
        super().__init__()
        self.admin_id = admin_id
        self.training_thread = None
        self.current_plot_path = None
        self.setup_ui()
        self.check_models_exist()

    def closeEvent(self, event):
        """Handle proper cleanup when the widget is closed"""
        self.stop_training_thread()
        event.accept()

    def stop_training_thread(self):
        """Safely stop the training thread if it's running"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption()
            self.training_thread.wait(2000)  # Wait up to 2 seconds

            if self.training_thread.isRunning():
                self.training_thread.terminate()
                self.training_thread.wait()

    def setup_ui(self):
        """Set up the user interface"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title_label = QLabel("Climbing Performance Prediction")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")

        # Info text
        info_text = """
        <p>This tool trains prediction models for IRCRA climbing grades based on test results 
        from your athlete database. The models use Principal Component Analysis (PCA) to identify
        patterns in test metrics that correlate with climbing ability.</p>
        """
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 14px; margin: 10px;")

        # Controls section
        controls_group = QGroupBox("Model Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #cccccc;
                margin-top: 16px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        # Status label
        self.model_status_label = QLabel("Model status: Not trained")
        controls_layout.addWidget(self.model_status_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_model_training)

        self.view_pca_button = QPushButton("View PCA Analysis")
        self.view_pca_button.clicked.connect(lambda: self.view_model_plot('pca'))
        self.view_pca_button.setEnabled(False)
        
        self.view_pcr_button = QPushButton("View PCR Plot")
        self.view_pcr_button.clicked.connect(lambda: self.view_model_plot('pcr'))
        self.view_pcr_button.setEnabled(False)

        self.view_svr_button = QPushButton("View SVR Plot")
        self.view_svr_button.clicked.connect(lambda: self.view_model_plot('svr'))
        self.view_svr_button.setEnabled(False)

        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.view_pca_button)
        button_layout.addWidget(self.view_pcr_button)
        button_layout.addWidget(self.view_svr_button)
        controls_layout.addLayout(button_layout)
    
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        controls_layout.addWidget(self.status_label)

        # Plot section
        self.plot_group = QGroupBox("Model Accuracy Visualization")
        self.plot_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #cccccc;
                margin-top: 16px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        plot_layout = QVBoxLayout()
        self.plot_group.setLayout(plot_layout)

        # Plot display area
        self.plot_label = QLabel("No plot available. Train the model first.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumHeight(400)
        self.plot_label.setStyleSheet("font-size: 14px; color: #666666;")
        plot_layout.addWidget(self.plot_label)

        # Add components to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(info_label)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.plot_group)

        # Initially hide the plot section
        self.plot_group.setVisible(False)

    def check_models_exist(self):
        """Check if trained models exist and update UI accordingly"""
        pca_exists = os.path.exists(PCA_MODEL_PATH)
        svr_exists = os.path.exists(SVR_MODEL_PATH)
        models_exist = pca_exists and svr_exists
        
        # Check for plot files
        pcr_plot_exists = hasattr(self, 'plot_paths') and 'pcr_accuracy' in self.plot_paths
        svr_plot_exists = hasattr(self, 'plot_paths') and 'svr_accuracy' in self.plot_paths
        pca_plot_exists = hasattr(self, 'plot_paths') and 'pca_variance' in self.plot_paths
        
        # If we don't have paths in the instance, fall back to global paths
        if not pcr_plot_exists:
            pcr_plot_exists = os.path.exists(PCR_PLOT_PATH)
        if not svr_plot_exists:
            svr_plot_exists = os.path.exists(SVR_PLOT_PATH)
        
        # Update buttons based on what exists
        self.view_pcr_button.setEnabled(models_exist and pcr_plot_exists)
        self.view_svr_button.setEnabled(models_exist and svr_plot_exists)
        
        # Enable PCA button if it exists
        if hasattr(self, 'view_pca_button'):
            self.view_pca_button.setEnabled(models_exist and pca_plot_exists)
        
        # Update model status text with accuracy if available
        if models_exist:
            status_text = "Model status: Trained"
            if (hasattr(self, 'mse_svr') and hasattr(self, 'mse_pcr') and
                    (self.mse_pcr is not None and self.mse_svr is not None)):
                status_text += f" (SVR accuracy: {self.mse_svr:.2f})"
                status_text += f" (PCR accuracy: {self.mse_pcr:.2f})"
            self.model_status_label.setText(status_text)
        else:
            self.model_status_label.setText("Model status: Not trained")

    def start_model_training(self):
        """Start the model training process"""
        # Confirm before starting training
        reply = QMessageBox.question(
            self,
            "Train Model",
            "Training may take several minutes. Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # Stop any existing thread
        self.stop_training_thread()

        # Reset UI
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Preparing...")
        self.status_label.setVisible(True)

        # Disable controls
        self.train_button.setEnabled(False)
        self.view_pcr_button.setEnabled(False)
        self.view_svr_button.setEnabled(False)

        # Create and start thread
        self.training_thread = ModelTrainingThread()

        # Connect signals
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.status_updated.connect(self.update_status)
        self.training_thread.training_complete.connect(self.training_completed)
        self.training_thread.training_failed.connect(self.training_failed)

        # Start the thread
        self.training_thread.start()

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update the status message"""
        self.status_label.setText(message)

    def training_completed(self, plot_paths=None):
        """
        Handle completion of model training thread.
        
        Args:
            plot_paths: Dictionary containing paths to generated plots
        """
        # Update UI
        self.progress_bar.setValue(100)
        self.status_label.setText("Training completed successfully!")
        
        # Store the plot paths for later use
        self.plot_paths = plot_paths if plot_paths else {}
        
        # Update global file paths based on the new plots
        global PCR_PLOT_PATH, SVR_PLOT_PATH
        if plot_paths and 'pcr_accuracy' in plot_paths:
            PCR_PLOT_PATH = plot_paths['pcr_accuracy']
        if plot_paths and 'svr_accuracy' in plot_paths:
            SVR_PLOT_PATH = plot_paths['svr_accuracy']
        
        # Re-enable the train button
        self.train_button.setEnabled(True)
        self.train_button.setText("Train Models")
        
        # Store the best model accuracy if provided
        if hasattr(self.training_thread, 'mse_pcr'):
            self.mse_pcr = self.training_thread.mse_pcr

        if hasattr(self.training_thread, 'mse_svr'):
            self.mse_svr = self.training_thread.mse_svr
        
        # Update model status and button states
        self.check_models_exist()
        
        # Show success message
        QMessageBox.information(
            self,
            "Training Complete",
            "The IRCRA prediction model has been successfully trained."
        )

    def training_failed(self, error_message):
        """Handle model training failure"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        print(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red;")

        # Re-enable controls
        self.train_button.setEnabled(True)
        self.check_models_exist()  # Reset button states based on available models

        # Show error message
        QMessageBox.critical(
            self,
            "Training Failed",
            f"An error occurred during model training:\n{error_message}"
        )

    def view_model_plot(self, model_type='pcr'):
        """Display the selected model plot"""
        # Determine which plot to show
        if model_type == 'pca':
            # Show PCA explained variance plot
            if hasattr(self, 'plot_paths') and 'pca_variance' in self.plot_paths:
                plot_path = self.plot_paths['pca_variance']
            else:
                # You might want to define a global path for this
                plot_path = os.path.join(os.path.dirname(__file__), 'plots', 'pca_explained_variance.png')
            plot_title = "PCA Explained Variance"
        elif model_type == 'pcr':
            # First try instance plot paths, then fall back to global path
            if hasattr(self, 'plot_paths') and 'pcr_accuracy' in self.plot_paths:
                plot_path = self.plot_paths['pcr_accuracy']
            else:
                plot_path = PCR_PLOT_PATH
            plot_title = "PCR Model Accuracy"
        else:  # svr
            if hasattr(self, 'plot_paths') and 'svr_accuracy' in self.plot_paths:
                plot_path = self.plot_paths['svr_accuracy']
            else:
                plot_path = SVR_PLOT_PATH
            plot_title = "SVR Model Accuracy"

        # Check if plot exists
        if not os.path.exists(plot_path):
            QMessageBox.warning(
                self,
                "Plot Not Found",
                f"The {model_type.upper()} plot is not available. Train the model first."
            )
            return

        # Update plot section title
        self.plot_group.setTitle(f"Model Visualization: {plot_title}")

        # Show the plot section
        self.plot_group.setVisible(True)

        # Load and display the plot
        pixmap = QPixmap(plot_path)
        if not pixmap.isNull():
            self.plot_label.setPixmap(pixmap.scaled(
                self.plot_label.width(),
                self.plot_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        else:
            self.plot_label.setText(f"Error loading the {model_type.upper()} plot.")

        # Store current plot path for resize events
        self.current_plot_path = plot_path

    def resizeEvent(self, event):
        """Handle resize events to scale the plot properly"""
        super().resizeEvent(event)

        # Rescale the plot if visible
        if self.plot_group.isVisible() and self.current_plot_path and os.path.exists(self.current_plot_path):
            pixmap = QPixmap(self.current_plot_path)
            if not pixmap.isNull():
                self.plot_label.setPixmap(pixmap.scaled(
                    self.plot_label.width(),
                    self.plot_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))