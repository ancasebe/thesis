import os
import joblib
import numpy as np
import pandas as pd
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QMessageBox, QGroupBox, QComboBox, QDialog, QFileDialog
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal

from gui.statistics.ircra_prediction_model import PredictionIRCRA

# Define model directory and file paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model file paths
PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_pipeline.joblib')
SVR_MODEL_PATH = os.path.join(MODEL_DIR, 'svr_model.joblib')
SVR_PLOT_PATH = os.path.join(MODEL_DIR, 'svr_prediction.png')


class PlotViewerDialog(QDialog):
    """Dialog for displaying high-resolution plots with export functionality"""

    def __init__(self, plot_path, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)  # Larger default size for better resolution

        # Store the original plot path for export
        self.plot_path = plot_path

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create image label with scroll area to handle large images
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(850, 600)  # Ensure minimum size for display

        # Set scroll policy to accommodate large images
        self.image_label.setScaledContents(False)

        # Load high-resolution image with better quality settings
        self.load_image(plot_path)

        # Add to layout with good margins
        main_layout.addWidget(self.image_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Add export button
        export_button = QPushButton("Export Plot")
        export_button.setToolTip("Save this plot to a location of your choice")
        export_button.clicked.connect(self.export_plot)
        button_layout.addWidget(export_button)

        # Add spacer to push buttons apart
        button_layout.addStretch()

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

    def load_image(self, plot_path):
        """Load the image with high quality settings"""
        if not os.path.exists(plot_path):
            self.image_label.setText(f"Error: Image not found at {plot_path}")
            return

        # Load the image directly with maximum quality
        original_pixmap = QPixmap(plot_path)

        if original_pixmap.isNull():
            self.image_label.setText(f"Error loading image from {plot_path}")
            return

        # Set the pixmap directly (will be properly scaled in resizeEvent)
        self.image_label.setPixmap(original_pixmap)

        # Store original dimensions for scaling
        self.original_width = original_pixmap.width()
        self.original_height = original_pixmap.height()

    def export_plot(self):
        """Export the plot to a user-selected location"""
        # Get suggested filename from the original path
        suggested_filename = os.path.basename(self.plot_path)

        # Open file dialog for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            suggested_filename,
            "Images (*.png *.jpg *.jpeg *.tiff);;All Files (*)"
        )

        # If user canceled, return
        if not file_path:
            return

        try:
            # Ensure the source file exists
            if not os.path.exists(self.plot_path):
                raise FileNotFoundError(f"Source file not found: {self.plot_path}")

            # Copy the original file to the new location to preserve full quality
            import shutil
            shutil.copy2(self.plot_path, file_path)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Plot exported successfully to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export plot: {str(e)}"
            )

    def resizeEvent(self, event):
        """Handle resize events to show high-quality image with proper scaling"""
        super().resizeEvent(event)

        # If we have a pixmap, scale it to fit the current window size
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            # Calculate available space
            available_width = self.image_label.width()
            available_height = self.image_label.height()

            # Calculate optimal display size while preserving aspect ratio
            if hasattr(self, 'original_width') and hasattr(self, 'original_height'):
                aspect_ratio = self.original_width / self.original_height

                # Determine if we're constrained by width or height
                if available_width / aspect_ratio <= available_height:
                    # Width-constrained
                    display_width = available_width
                    display_height = available_width / aspect_ratio
                else:
                    # Height-constrained
                    display_height = available_height
                    display_width = available_height * aspect_ratio

                # Load the original image and scale with high quality
                original_pixmap = QPixmap(self.plot_path)
                if not original_pixmap.isNull():
                    scaled_pixmap = original_pixmap.scaled(
                        int(display_width),
                        int(display_height),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation  # High quality transformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)

class ModelTrainingThread(QThread):
    """Thread for training IRCRA prediction models without freezing the UI"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    training_complete = Signal(dict)  # Modified to pass plot paths
    training_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.mse_svr = None

    def run(self):
        try:
            # Update progress and status
            self.progress_updated.emit(5)
            self.status_updated.emit("Initializing model training...")

            # Create the prediction model
            model = PredictionIRCRA()

            # Make sure the model uses the correct directories
            model.model_dir = MODEL_DIR
            model.plots_dir = PLOTS_DIR

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
            print('X', X)

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
            _, self.mse_svr, svr_plot_path, _ = model.train_best_svr(X_pca, y_pca)

            # Get plot paths for later display
            plot_paths = {
                'corr_matrix': corr_matrix_path,
                'pca_variance': os.path.join(model.plots_dir, 'pca_explained_variance.png'),
                'pca_scatter': os.path.join(model.plots_dir, 'pca_scatter.png'),
                'svr_accuracy': svr_plot_path,
                'svr_scatter': os.path.join(model.plots_dir, 'svr_sel_scatter.png'),
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
        self.mse_svr = None

        # Load saved model metadata if available
        self.load_model_metadata()

        # Find all available plots in the plots directory
        self.find_available_plots()

        self.setup_ui()
        self.check_models_exist()

    def load_model_metadata(self):
        """Load saved model metadata from file"""
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r') as f:
                    metadata = json.load(f)

                # Load accuracy information
                if 'mse_svr' in metadata:
                    self.mse_svr = metadata['mse_svr']

                # Load plot paths if available
                if 'plot_paths' in metadata:
                    self.plot_paths = metadata['plot_paths']

                print(f"Loaded model metadata: SVR accuracy={self.mse_svr}")
            except Exception as e:
                print(f"Error loading model metadata: {e}")

    def save_model_metadata(self):
        """Save model metadata to file"""
        metadata = {
            'mse_svr': self.mse_svr,
        }

        # Save plot paths
        if hasattr(self, 'plot_paths'):
            metadata['plot_paths'] = self.plot_paths

        try:
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("Model metadata saved successfully")
        except Exception as e:
            print(f"Error saving model metadata: {e}")

    def find_available_plots(self):
        """Find all available plots in the plots directory"""
        # Initialize empty plot paths dictionary if not loaded from metadata
        if not hasattr(self, 'plot_paths'):
            self.plot_paths = {}

        # Look for standard plot files in the plots directory
        if os.path.exists(PLOTS_DIR):
            # Mapping of file names to plot keys
            plot_file_mapping = {
                'pca_explained_variance.png': 'pca_variance',
                'pca_scatter.png': 'pca_scatter',
                'correlation_matrix.png': 'corr_matrix',
                'svr_accuracy.png': 'svr_accuracy',
                'svr_sel_scatter.png': 'svr_scatter',
            }

            # Check for each standard plot file
            for file_name, plot_key in plot_file_mapping.items():
                file_path = os.path.join(PLOTS_DIR, file_name)
                if os.path.exists(file_path):
                    self.plot_paths[plot_key] = file_path

        # Also check for plots in the model directory
        if os.path.exists(MODEL_DIR):
            if os.path.exists(SVR_PLOT_PATH):
                self.plot_paths['svr_accuracy'] = SVR_PLOT_PATH

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
        <p>This tool trains a Support Vector Regression (SVR) model to predict IRCRA climbing grades 
        based on test results from your athlete database. The model uses Principal Component Analysis (PCA)
        to reduce dimensionality and identify key performance factors that correlate with climbing ability.</p>
        
        <p>The trained model can help identify strengths and weaknesses in an athlete's performance profile
        and suggest areas for targeted training improvements.</p>
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
                margin-top: 20px;
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

        # Train button and plot selector in separate layouts
        train_button_layout = QHBoxLayout()
        self.train_button = QPushButton("Train New Model")
        self.train_button.clicked.connect(self.start_model_training)
        train_button_layout.addWidget(self.train_button)
        train_button_layout.addStretch()
        controls_layout.addLayout(train_button_layout)

        # Plot selection combo box
        plot_selection_layout = QHBoxLayout()
        plot_selection_layout.addWidget(QLabel("Select plot to view:"))

        self.plot_combo = QComboBox()
        self.plot_combo.setMinimumWidth(250)
        plot_selection_layout.addWidget(self.plot_combo)

        self.view_plot_button = QPushButton("View Plot")
        self.view_plot_button.clicked.connect(self.show_selected_plot)
        self.view_plot_button.setEnabled(False)
        plot_selection_layout.addWidget(self.view_plot_button)

        controls_layout.addLayout(plot_selection_layout)

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

        # Add components to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(info_label)
        main_layout.addWidget(controls_group)

        # Initialize plot descriptions
        self.plot_descriptions = {
            'pca_variance': 'PCA Explained Variance',
            'pca_scatter': 'PCA Component Scatter Plot',
            'svr_accuracy': 'SVR Model Prediction Accuracy',
            'svr_scatter': 'SVR Prediction Scatter Plot',
            'corr_matrix': 'Feature Correlation Matrix'
        }

    def update_plot_combo(self):
        """Update the plot combo box with available plots"""
        self.plot_combo.clear()
        self.plot_combo.addItem("Select a plot...", "")

        # Add plots from plot_paths attribute
        if hasattr(self, 'plot_paths') and self.plot_paths:
            has_plots = False
            for key, path in self.plot_paths.items():
                if os.path.exists(path):  # Only add if file exists
                    display_name = self.plot_descriptions.get(key, key.replace('_', ' ').title())
                    self.plot_combo.addItem(display_name, key)
                    has_plots = True

            self.view_plot_button.setEnabled(has_plots)

    def show_selected_plot(self):
        """Show the selected plot in a separate dialog window"""
        selected_data = self.plot_combo.currentData()
        if not selected_data:
            return

        # Determine the plot path
        if hasattr(self, 'plot_paths') and selected_data in self.plot_paths:
            plot_path = self.plot_paths[selected_data]
        else:
            QMessageBox.warning(self, "Plot Not Found", "The selected plot file could not be found.")
            return

        # Check if plot file exists
        if not os.path.exists(plot_path):
            QMessageBox.warning(
                self,
                "Plot Not Found",
                f"The selected plot file could not be found at {plot_path}."
            )
            return

        # Get the display name for the title
        title = self.plot_descriptions.get(selected_data, selected_data.replace('_', ' ').title())

        # Create and show the plot viewer dialog
        dialog = PlotViewerDialog(plot_path, title, self)
        dialog.exec()

    def check_models_exist(self):
        """Check if trained models exist and update UI accordingly"""
        pca_exists = os.path.exists(PCA_MODEL_PATH)
        svr_exists = os.path.exists(SVR_MODEL_PATH)
        models_exist = pca_exists and svr_exists

        # Update plot combo if needed
        self.update_plot_combo()

        # Update model status text with accuracy if available
        if models_exist:
            status_text = "Model status: Trained"
            if hasattr(self, 'mse_svr') and self.mse_svr is not None:
                status_text += f" (SVR accuracy: {self.mse_svr:.2f})"
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
        self.view_plot_button.setEnabled(False)

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
        global SVR_PLOT_PATH
        if plot_paths and 'svr_accuracy' in plot_paths:
            SVR_PLOT_PATH = plot_paths['svr_accuracy']

        # Update the plot combo box
        self.update_plot_combo()

        # Re-enable the train button
        self.train_button.setEnabled(True)
        self.train_button.setText("Train Model")

        # Store the model accuracy from the training thread
        if hasattr(self.training_thread, 'mse_svr'):
            self.mse_svr = self.training_thread.mse_svr

        # Save the metadata to file for future use
        self.save_model_metadata()

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