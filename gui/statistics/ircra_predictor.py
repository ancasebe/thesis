
"""
IRCRA prediction module for the Climbing Testing Application.

This module defines the IRCRAPredictor class which uses machine learning models
to predict a climber's IRCRA (International Rock Climbing Research Association)
grade based on their test performance data.

Key functionalities:
- Load and use trained machine learning models
- Process and transform test data for prediction
- Generate IRCRA grade predictions using multiple models
- Extract relevant features from test data
- Provide confidence measures for predictions

The IRCRAPredictor serves as the inference engine for the application's performance
prediction capabilities, using models trained in the statistical module to estimate
a climber's performance potential.
"""

import os
import json
import ast
import pandas as pd
import joblib

from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager


class IRCRAPredictor:
    """
    Class for loading trained models and making IRCRA predictions for climbers.
    """

    def __init__(self, model_dir=None, climber_db=None, tests_db=None):
        """
        Initialize the predictor with model paths and database connections.
        
        Args:
            model_dir (str): Directory where models are stored
            climber_db (str): Path to climber database
            tests_db (str): Path to tests database
        """
        if model_dir is None:
            # Use the models directory within statistics folder
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        else:
            self.model_dir = model_dir
            
        # Set default database paths if not provided
        if climber_db is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.climber_db = os.path.join(script_dir, '..', 'databases', 'climber_database.db')
        else:
            self.climber_db = climber_db
            
        if tests_db is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.tests_db = os.path.join(script_dir, '..', 'databases', 'tests_database.db')
        else:
            self.tests_db = tests_db
            
        # Initialize database managers
        self.climber_manager = ClimberDatabaseManager(db_name=self.climber_db)
        self.test_manager = ClimbingTestManager(db_name=self.tests_db)
        
        # Load models
        self.pca_data = None
        self.svr_model = None
        self.linreg_model = None
        self.load_models()
    
    def load_models(self):
        """Load the trained PCA, SVR, and Linear Regression models."""
        try:
            # Load PCA model
            pca_path = os.path.join(self.model_dir, 'pca_pipeline.joblib')

            if os.path.exists(pca_path):
                self.pca_data = joblib.load(pca_path)
            else:
                print(f"Warning: PCA model not found at {pca_path}")
                
            # Load SVR model
            svr_path = os.path.join(self.model_dir, 'svr_model.joblib')

            if os.path.exists(svr_path):
                self.svr_model = joblib.load(svr_path)
            else:
                print(f"Warning: SVR model not found at {svr_path}")
                
            # Load Linear Regression model (PCR)
            # linreg_path = os.path.join(self.model_dir, 'pcr_model.joblib')
            # if not os.path.exists(linreg_path):
            # linreg_path = os.path.join(self.model_dir, 'linreg_model.joblib')
            #     # if not os.path.exists(linreg_path):
            #     #     # Try alternative extension
            #     #     linreg_path = os.path.join(self.model_dir, 'linreg_model.pkl')
            #
            # if os.path.exists(linreg_path):
            #     self.linreg_model = joblib.load(linreg_path)
            #     print(f"Linear Regression model loaded from {linreg_path}")
            # else:
            #     print(f"Warning: Linear Regression model not found at {linreg_path}")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def load_test_data(self, test_id):
        """
        Loads test data and climber information for a specific test ID.
        
        Args:
            test_id (int): ID of the test to load
            
        Returns:
            dict: Combined data from test and climber records
        """
        # Get test data
        test_data = self.test_manager.get_test_data(test_id)
        
        if not test_data:
            raise ValueError(f"No test found with ID: {test_id}")
        
        # Get climber data
        participant_id = test_data.get('participant_id')
        admin_id = test_data.get('admin_id', 1)  # Default to admin_id 1 if not specified
        
        climber_data = self.climber_manager.get_user_data(admin_id, participant_id)
        
        if not climber_data:
            raise ValueError(f"No climber found with participant ID: {participant_id}")
        
        # Combine data
        combined_data = {**climber_data, **test_data}
        
        # Normalize gender and dominant_arm values to match training data format
        if combined_data.get('gender') == 'Female':
            combined_data['gender'] = 1
        elif combined_data.get('gender') == 'Male':
            combined_data['gender'] = 0
        else:
            combined_data['gender'] = 2

        if combined_data.get('dominant_arm') == 'Left':
            combined_data['dominant_arm'] = 1
        elif combined_data.get('dominant_arm') == 'Right':
            combined_data['dominant_arm'] = 0
        else:
            combined_data['dominant_arm'] = None
            
        return combined_data

    def prepare_features(self, data_dict, include_metrics=None, include_demographics=None):
        """
        Prepare feature matrix from test data.
        
        Args:
            data_dict (dict): Data dictionary with test and climber info
            include_metrics (list): Metrics to include
            include_demographics (list): Demographics to include
            
        Returns:
            DataFrame: Feature matrix X
        """
        if include_demographics is None:
            include_demographics = [
                'gender', 'age', 'years_climbing', 'bouldering', 'climbing_indoor',
                'dominant_arm', 'weight', 'height', 'climbing_freq'
            ]
        if include_metrics is None:
            include_metrics = ['max_strength', 'sum_work', 'sum_work_above_cf',
                               'critical_force', 'rfd_norm_overall']
        
        # Create DataFrame with a single row
        df = pd.DataFrame([data_dict])
        
        # Parse test_results if needed
        test_results = data_dict.get('test_results', {})
        if isinstance(test_results, str):
            try:
                test_results = json.loads(test_results)
            except json.JSONDecodeError:
                try:
                    test_results = ast.literal_eval(test_results)
                except:
                    test_results = {}
                    
        # Add metrics to DataFrame
        for metric in include_metrics:
            if metric in test_results:
                df[metric] = test_results[metric]
                # Convert to numeric if string
                if isinstance(df[metric].iloc[0], str):
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
            else:
                df[metric] = 0
        
        # Process demographic columns
        include_cols = []
        
        for col in include_demographics:
            if col in df.columns:
                include_cols.append(col)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0
                include_cols.append(col)
                
        for col in include_metrics:
            if col in df.columns:
                include_cols.append(col)
            
        # Select only the needed columns
        X = df[include_cols].copy()
        
        # Fill NaN values with 0
        X = X.fillna(0)
        
        return X
    
    def predict_ircra(self, test_id=None, feature_data=None, model_type='svr'):
        """
        Predicts IRCRA rating for a climber using either a test ID or provided feature data.
        
        Args:
            test_id (int): ID of the test to use
            feature_data (dict): Raw feature data (alternative to test_id)
            model_type (str): Type of model to use ('svr' or 'linear')
            
        Returns:
            float: Predicted IRCRA rating
        """
        # Check if models are loaded
        if self.pca_data is None:
            raise ValueError("PCA model not loaded")
            
        if model_type == 'svr' and self.svr_model is None:
            raise ValueError("SVR model not loaded")

        # Extract the scaler and PCA components
        scaler = self.pca_data.get('scaler')
        pca = self.pca_data.get('pca')
        
        if scaler is None or pca is None:
            raise ValueError("Invalid PCA model data")
        
        # Get data
        if test_id is not None:
            data = self.load_test_data(test_id)
        elif feature_data is not None:
            data = feature_data
        else:
            raise ValueError("Either test_id or feature_data must be provided")
        
        # Prepare features
        X_raw = self.prepare_features(data)
        
        # Transform features using the same pipeline used for training
        X_scaled = scaler.transform(X_raw)
        X_pca = pca.transform(X_scaled)
        
        # Select model and make prediction
        if model_type == 'svr':
            prediction = self.svr_model.predict(X_pca)[0]
        else:  # linear/PCR
            prediction = self.linreg_model.predict(X_pca)[0]
        
        # Return rounded prediction
        return round(prediction)

# # Example usage
if __name__ == "__main__":
    # Initialize with the correct models directory
    predictor = IRCRAPredictor(
        model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    )

    # Example: Predict IRCRA for a specific test
    try:
        test_id = 8  # Replace with actual test ID

        # Try SVR prediction
        if predictor.svr_model is not None:
            svr_prediction = predictor.predict_ircra(test_id=test_id, model_type='svr')
        else:
            print("SVR model not available for prediction")

        # Try Linear prediction
        if predictor.linreg_model is not None:
            linear_prediction = predictor.predict_ircra(test_id=test_id, model_type='linear')
            print(f"Predicted IRCRA (Linear): {linear_prediction}")
        else:
            print("Linear model not available for prediction")

        # Print test data info
        test_data = predictor.load_test_data(test_id)
        print("\nClimber Information:")
        for key in ['name', 'ircra', 'age', 'gender', 'weight', 'height', 'years_climbing']:
            if key in test_data:
                print(f"  {key}: {test_data.get(key)}")

        # Print test metrics
        print("\nTest Metrics:")
        test_results = test_data.get('test_results', {})
        if isinstance(test_results, dict):
            for key, value in test_results.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error predicting IRCRA: {str(e)}")