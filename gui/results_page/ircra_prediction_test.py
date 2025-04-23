# predict_ircra.py
# Predicts IRCRA rating for a climber using trained PCR model

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pca_model import MLDataProcessor
from db_testing.project_files.climber_db_manager import ClimberDatabaseManager
from db_testing.project_files.test_db_manager import ClimbingTestManager

MODEL_DIR = 'models'

def load_test_data(test_id, climber_db=None, tests_db=None):
    """
    Loads test data and climber information for a specific test ID.
    
    Args:
        test_id (int): ID of the test to load
        climber_db (str): Path to the climber database
        tests_db (str): Path to the tests database
        
    Returns:
        dict: Combined data from test and climber records
    """
    # Set default database paths if not provided
    if not climber_db:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        climber_db = os.path.join(script_dir, '..', 'databases', 'climber_database.db')
    
    if not tests_db:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tests_db = os.path.join(script_dir, '..', 'databases', 'tests_database.db')
    
    # Initialize database managers
    test_manager = ClimbingTestManager(db_name=tests_db)
    climber_manager = ClimberDatabaseManager(db_name=climber_db)
    
    # Get test data
    test_data = test_manager.get_test_data(test_id)
    
    if not test_data:
        raise ValueError(f"No test found with ID: {test_id}")
    
    # Get climber data
    participant_id = test_data.get('participant_id')
    admin_id = test_data.get('admin_id', 1)  # Default to admin_id 1 if not specified
    
    climber_data = climber_manager.get_user_data(admin_id, participant_id)
    
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

def prepare_data_for_prediction(data_dict, scaler=None, include_metrics=None, include_demographics=None):
    """
    Prepares data for prediction, ensuring it matches the structure expected by the scaler
    """
    # Default columns if not specified
    if include_demographics is None:
        include_demographics = [
            'gender', 'age', 'years_climbing', 'bouldering', 'climbing_indoor',
            'dominant_arm', 'weight', 'height', 'sport_freq', 'climbing_freq'
        ]
    if include_metrics is None:
        include_metrics = ['max_strength', 'sum_work', 'sum_work_above_cf',
                          'critical_force', 'rfd_norm_overall', 'reps_to_cf',
                          'force_drop_pct', 'rfd_norm_first3', 'rfd_norm_last3']
    
    # Create a DataFrame with a single row
    df = pd.DataFrame([data_dict])
    
    # Extract test_results from the data if it exists
    test_results = data_dict.get('test_results', {})
    if isinstance(test_results, str):
        import json
        try:
            test_results = json.loads(test_results)
        except json.JSONDecodeError:
            # Try using literal_eval as fallback
            import ast
            try:
                test_results = ast.literal_eval(test_results)
            except:
                test_results = {}
    
    # Add test metrics to DataFrame
    for metric in include_metrics:
        if metric in test_results:
            df[metric] = test_results[metric]
        else:
            # Add missing metrics with default value of 0
            df[metric] = 0
    
    # Ensure all demographic columns exist with default values
    for col in include_demographics:
        if col not in df.columns:
            df[col] = 0
    
    # Identify categorical vs numeric columns
    categorical_cols = []
    numeric_cols = []
    
    # Check each demographic column
    for col in include_demographics:
        if col not in df.columns:
            continue
            
        # Check if column contains string/object data or known categorical columns
        if (df[col].dtype == 'object' or 
            col in ['gender', 'dominant_arm']):
            categorical_cols.append(col)
        else:
            # These are numeric demographics (age, weight, etc.)
            numeric_cols.append(col)
            # Ensure they're numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # All metrics should be numeric
    for col in include_metrics:
        if col in df.columns:
            numeric_cols.append(col)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select features (both categorical and numeric)
    X = df[categorical_cols + numeric_cols].copy()
    
    # Make sure gender and other categorical values are properly formatted
    # This ensures consistent one-hot encoding results
    if 'gender' in X.columns:
        # Convert to numeric if needed
        X['gender'] = pd.to_numeric(X['gender'], errors='coerce').fillna(2)
    
    if 'dominant_arm' in X.columns:
        # Convert to numeric if needed
        X['dominant_arm'] = pd.to_numeric(X['dominant_arm'], errors='coerce').fillna(2)
    
    # Instead of using pandas get_dummies, which can create inconsistent columns,
    # we'll manually create the one-hot encoded columns for categorical variables
    # This gives us more control over exactly which columns are created
    
    # Handle gender manually to ensure all expected columns are created
    if 'gender' in X.columns:
        # Create dummy columns for all possible gender values (0, 1, 2)
        gender_val = X['gender'].iloc[0]
        X['gender_0'] = 1 if gender_val == 0 else 0
        X['gender_1'] = 1 if gender_val == 1 else 0
        X['gender_2'] = 1 if gender_val == 2 else 0
        # Drop the original column
        X = X.drop('gender', axis=1)
    else:
        # If gender column doesn't exist, add the encoded columns with 0s
        X['gender_0'] = 0
        X['gender_1'] = 0
        X['gender_2'] = 0
    
    # Handle dominant_arm similarly if needed
    if 'dominant_arm' in X.columns:
        arm_val = X['dominant_arm'].iloc[0]
        X['dominant_arm_0'] = 1 if arm_val == 0 else 0
        X['dominant_arm_1'] = 1 if arm_val == 1 else 0
        X['dominant_arm_2'] = 1 if arm_val == 2 else 0
        X = X.drop('dominant_arm', axis=1)
    
    # For any other categorical columns, use pandas get_dummies
    remaining_cat_cols = [col for col in categorical_cols if col not in ['gender', 'dominant_arm']]
    if remaining_cat_cols:
        X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=False)
    
    # Ensure specific required columns that we know were in the training data
    required_columns = ['climbing_freq', 'sport_freq', 'gender_0', 'gender_1', 'gender_2']
    for col in required_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Fill any remaining NaN values with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(0)
    
    # If scaler is provided, ensure column names match exactly
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_
        
        # Create a dataframe with all expected columns, filled with zeros
        aligned_X = pd.DataFrame(0, index=[0], columns=expected_cols)
        
        # Fill in values from our processed dataframe where columns match
        for col in X.columns:
            if col in expected_cols:
                aligned_X[col] = X[col].values
        
        # Use the aligned dataframe
        X = aligned_X
    
    return X

def load_models(pca_path=None, pcr_path=None):
    """
    Loads the PCA and PCR models from disk.
    
    Args:
        pca_path (str): Path to PCA model file
        pcr_path (str): Path to PCR model file
        
    Returns:
        tuple: (scaler, pca, pcr_model)
    """
    if not pca_path:
        pca_path = os.path.join(MODEL_DIR, 'pca_model.pkl')
    if not pcr_path:
        pcr_path = os.path.join(MODEL_DIR, 'pcr_model.pkl')
    
    # Load PCA components
    with open(pca_path, 'rb') as f:
        pca_data = pickle.load(f)
    scaler = pca_data['scaler']
    pca = pca_data['pca']
    
    # Load PCR model
    with open(pcr_path, 'rb') as f:
        pcr_model = pickle.load(f)
    
    return scaler, pca, pcr_model

def predict_ircra(test_id=None, feature_data=None, climber_db=None, tests_db=None):
    """
    Predicts IRCRA rating for a climber using either a test ID or provided feature data.
    
    Args:
        test_id (int): ID of the test to use
        feature_data (dict): Raw feature data (alternative to test_id)
        climber_db (str): Path to climber database
        tests_db (str): Path to tests database
        
    Returns:
        float: Predicted IRCRA rating
    """
    # Load models first
    scaler, pca, pcr_model = load_models()
    
    # Get data
    if test_id is not None:
        data = load_test_data(test_id, climber_db, tests_db)
    elif feature_data is not None:
        data = feature_data
    else:
        raise ValueError("Either test_id or feature_data must be provided")
    
    # Prepare data with access to the scaler
    X = prepare_data_for_prediction(data, scaler=scaler)
    
    # Transform features
    X_scaled = scaler.transform(X)
    X_pcs = pca.transform(X_scaled)
    
    # Make prediction
    ircra_prediction = pcr_model.predict(X_pcs)[0]
    
    return round(ircra_prediction)

# def main():
#     """
#     Example usage of the prediction function.
#     """
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Predict IRCRA rating for a climber')
#     parser.add_argument('--test_id', type=int, help='Test ID to use for prediction')
#     parser.add_argument('--climber_db', type=str, help='Path to climber database')
#     parser.add_argument('--tests_db', type=str, help='Path to tests database')
#
#     args = parser.parse_args()
#
#     if args.test_id:
#         try:
#             # Load the test data
#             test_data = load_test_data(args.test_id, args.climber_db, args.tests_db)
#             print(f"Loaded test data for test ID {args.test_id}")
#
#             # Predict IRCRA
#             ircra = predict_ircra(args.test_id, None, args.climber_db, args.tests_db)
#             print(f"Predicted IRCRA rating: {ircra:.2f}")
#
#             # Print the data used for prediction
#             print("\nClimber Information:")
#             for key in ['name', 'age', 'gender', 'weight', 'height', 'years_climbing']:
#                 if key in test_data:
#                     print(f"  {key}: {test_data.get(key)}")
#
#             print("\nTest Metrics Used:")
#             if 'test_results' in test_data and isinstance(test_data['test_results'], dict):
#                 for key, value in test_data['test_results'].items():
#                     print(f"  {key}: {value}")
#
#         except Exception as e:
#             print(f"Error predicting IRCRA: {e}")
#     else:
#         print("Please provide a test_id to predict IRCRA rating")
#         print("Example: python predict_ircra.py --test_id 123")

if __name__ == '__main__':
    ircra_rating = predict_ircra(test_id=1)
    print(f"Predicted IRCRA rating: {int(ircra_rating):d}")
    # main()