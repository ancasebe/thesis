# statistics_page.py

import os
import pickle
import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QMessageBox, QGroupBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal

# Import the prediction model modules
from gui.statistics.pca_model import MLDataProcessor
from gui.statistics.ircra_prediction_model import plot_regression_results


# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# PCA model path
PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_model.pkl')
# PCR model path
PCR_MODEL_PATH = os.path.join(MODEL_DIR, 'pcr_model.pkl')
# Model info path
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.pkl')
# Plot path
PLOT_PATH = os.path.join(MODEL_DIR, 'pcr_prediction.png')


class ModelTrainingThread(QThread):
    """Thread for training the IRCRA prediction model without freezing the UI"""
    
    # Signals
    progress_updated = Signal(int)
    status_updated = Signal(str)
    training_complete = Signal(dict)
    training_failed = Signal(str)
    
    def __init__(self, admin_id, climber_db_path, tests_db_path):
        super().__init__()
        self.admin_id = admin_id
        self.climber_db_path = climber_db_path
        self.tests_db_path = tests_db_path
    
    def run(self):
        try:
            self.status_updated.emit("Initializing model training...")
            self.progress_updated.emit(10)
            
            # 1. Initialize the data processor
            self.status_updated.emit("Loading data processor...")
            processor = MLDataProcessor(
                climber_db=self.climber_db_path,
                tests_db=self.tests_db_path,
                admin_id=self.admin_id
            )
            self.progress_updated.emit(20)
            
            # 2. Load the data
            self.status_updated.emit("Loading and processing test data...")
            df = processor.load_data(test_type='ao')
            self.progress_updated.emit(40)
            
            # 3. Prepare features and fit PCA
            self.status_updated.emit("Preparing features and fitting PCA...")
            X_raw, y = processor.prepare_features(df)
            pca_result = processor.fit_pca(X_raw, n_components=5, save_path=PCA_MODEL_PATH)
            self.progress_updated.emit(60)
            
            # 4. Prepare for PCR model
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            
            # Transform features
            X_scaled = pca_result['scaler'].transform(X_raw)
            X_pcs = pca_result['pca'].transform(X_scaled)
            
            # Train/test split
            self.status_updated.emit("Splitting data into training and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_pcs, y, test_size=0.3, random_state=42
            )
            self.progress_updated.emit(70)
            
            # 5. Train PCR model
            self.status_updated.emit("Training PCR model...")
            linreg = LinearRegression()
            linreg.fit(X_train, y_train)
            self.progress_updated.emit(80)
            
            # 6. Evaluate model
            self.status_updated.emit("Evaluating model performance...")
            y_pred_lin = linreg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_lin)
            r2 = r2_score(y_test, y_pred_lin)
            
            # Calculate Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((y_test - y_pred_lin) / y_test)) * 100
            
            # Save the PCR model
            with open(PCR_MODEL_PATH, 'wb') as f:
                pickle.dump(linreg, f)
            
            # 7. Generate and save the plot
            self.status_updated.emit("Generating performance visualization...")
            plot_regression_results(
                X_test, y_test, y_pred_lin,
                "Prediction of IRCRA level using PCR",
                PLOT_PATH
            )
            
            # 8. Save model information
            model_info = {
                'training_date': datetime.datetime.now(),
                'mse': mse,
                'r2': r2,
                'mape': mape,
                'samples_count': len(df),
                'features_count': X_raw.shape[1],
                'components_used': X_pcs.shape[1]
            }
            
            with open(MODEL_INFO_PATH, 'wb') as f:
                pickle.dump(model_info, f)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Model training complete!")
            self.training_complete.emit(model_info)
            
        except Exception as e:
            self.training_failed.emit(f"Error during model training: {str(e)}")


class StatisticsPage(QWidget):
    """
    The Statistics page shows performance analytics and provides
    functionality to train and view the IRCRA prediction model.
    """
    
    def __init__(self, admin_id):
        super().__init__()
        self.admin_id = admin_id
        self.training_thread = None
        self.setup_ui()
        self.load_model_info()
    
    def setup_ui(self):
        """Set up the user interface for the Statistics page"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # 1. Title and application info
        title_label = QLabel("Climbing Performance Statistics")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        
        info_text = """
        <p>This page provides statistical analysis tools for climbing performance data. 
        It allows you to generate prediction models for IRCRA climbing grades based on 
        test results across your athlete database.</p>
        
        <p>The IRCRA grade prediction model uses Principal Component Regression (PCR) to 
        find patterns in test metrics that correlate with climbing ability.</p>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft)
        info_label.setStyleSheet("font-size: 14px; margin: 10px;")
        
        # 2. Model Information section
        model_group = QGroupBox("IRCRA Prediction Model")
        model_group.setStyleSheet("""
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
        
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        # Model status information
        self.model_info_label = QLabel("Loading model information...")
        self.model_info_label.setStyleSheet("font-size: 14px; margin: 5px;")
        model_layout.addWidget(self.model_info_label)
        
        # Training controls
        training_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train New Model")
        self.train_button.clicked.connect(self.start_model_training)
        
        self.view_plot_button = QPushButton("View Accuracy Plot")
        self.view_plot_button.clicked.connect(self.view_accuracy_plot)
        self.view_plot_button.setEnabled(os.path.exists(PLOT_PATH))
        
        training_layout.addWidget(self.train_button)
        training_layout.addWidget(self.view_plot_button)
        model_layout.addLayout(training_layout)
        
        # Progress bar for training
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        model_layout.addWidget(self.progress_bar)
        
        # Status label for training
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        model_layout.addWidget(self.status_label)
        
        # 3. Accuracy Plot section (initially empty)
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
        
        # Plot container
        self.plot_label = QLabel("No accuracy plot available. Train a model first.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumHeight(400)
        self.plot_label.setStyleSheet("font-size: 14px; color: #666666;")
        plot_layout.addWidget(self.plot_label)
        
        # Add all components to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(info_label)
        main_layout.addWidget(model_group)
        main_layout.addWidget(self.plot_group)
        
        # Initially hide the plot section
        self.plot_group.setVisible(False)
    
    def load_model_info(self):
        """Load and display information about the current model"""
        if os.path.exists(MODEL_INFO_PATH):
            try:
                with open(MODEL_INFO_PATH, 'rb') as f:
                    model_info = pickle.load(f)
                
                # Format the date
                date_str = model_info['training_date'].strftime('%Y-%m-%d at %H:%M:%S')
                
                # Create info text
                info_text = f"""
                <b>Model Status:</b> Trained and ready
                <b>Last Training:</b> {date_str}
                <b>Mean Squared Error:</b> {model_info['mse']:.4f}
                <b>R² Score:</b> {model_info['r2']:.4f}
                <b>Mean Absolute % Error:</b> {model_info['mape']:.2f}%
                <b>Samples Used:</b> {model_info['samples_count']}
                <b>Features Count:</b> {model_info['features_count']}
                <b>PCA Components:</b> {model_info['components_used']}
                """
                
                self.model_info_label.setText(info_text)
                
                # Enable view plot button if the plot exists
                self.view_plot_button.setEnabled(os.path.exists(PLOT_PATH))
                
            except Exception as e:
                self.model_info_label.setText(f"Error loading model info: {str(e)}")
        else:
            self.model_info_label.setText("No trained model found. Please train a new model.")
            self.view_plot_button.setEnabled(False)
    
    def start_model_training(self):
        """Start the model training process in a separate thread"""
        # Get database paths
        climber_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'databases', 'climber_database.db')
        tests_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'databases', 'tests_database.db')
        
        # Check if the database files exist
        if not os.path.exists(climber_db_path) or not os.path.exists(tests_db_path):
            QMessageBox.warning(self, "Database Not Found", 
                               "Could not locate the database files needed for training.")
            return
        
        # Confirm training
        reply = QMessageBox.question(self, "Train New Model",
                                    "Training a new model may take several minutes. Proceed?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
        
        # Disable controls
        self.train_button.setEnabled(False)
        self.view_plot_button.setEnabled(False)
        
        # Show progress bar and status
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Initializing...")
        self.status_label.setVisible(True)
        
        # Start training in a separate thread
        self.training_thread = ModelTrainingThread(
            admin_id=self.admin_id,
            climber_db_path=climber_db_path,
            tests_db_path=tests_db_path
        )
        
        # Connect signals
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.status_updated.connect(self.update_status)
        self.training_thread.training_complete.connect(self.training_completed)
        self.training_thread.training_failed.connect(self.training_failed)
        
        # Start the thread
        self.training_thread.start()
    
    def update_progress(self, value):
        """Update the progress bar value"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update the status label text"""
        self.status_label.setText(message)
    
    def training_completed(self, model_info):
        """Handle the completion of model training"""
        # Re-enable controls
        self.train_button.setEnabled(True)
        self.view_plot_button.setEnabled(True)
        
        # Update UI
        self.progress_bar.setValue(100)
        self.status_label.setText("Model training complete!")
        
        # Update model info display
        self.load_model_info()
        
        # Show success message
        QMessageBox.information(self, "Training Complete", 
                               "The IRCRA prediction model has been successfully trained.")
        
        # Hide progress after delay
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        QTimer.singleShot(2000, lambda: self.status_label.setVisible(False))
    
    def training_failed(self, error_message):
        """Handle errors during model training"""
        # Re-enable controls
        self.train_button.setEnabled(True)
        self.view_plot_button.setEnabled(os.path.exists(PLOT_PATH))
        
        # Update UI
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red;")
        
        # Show error message
        QMessageBox.critical(self, "Training Failed", 
                            f"An error occurred during model training:\n{error_message}")
    
    def view_accuracy_plot(self):
        """Display the accuracy plot in the UI"""
        if not os.path.exists(PLOT_PATH):
            QMessageBox.warning(self, "Plot Not Found", 
                               "The accuracy plot is not available. Train a model first.")
            return
        
        # Show the plot section
        self.plot_group.setVisible(True)
        
        # Load and display the plot image
        pixmap = QPixmap(PLOT_PATH)
        if not pixmap.isNull():
            # Scale the pixmap to fit the container while maintaining aspect ratio
            self.plot_label.setPixmap(pixmap.scaled(
                self.plot_label.width(), self.plot_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.plot_label.setText("Error loading the accuracy plot.")