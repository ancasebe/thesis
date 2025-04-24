import json
import pickle
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager


class MLDataProcessor:
    """
    Processor to load climber and test data from SQLite databases,
    merge and prepare features, and compute PCA for dimensionality reduction.

    Attributes:
        climber_manager: Instance of ClimberDatabaseManager to fetch climber info.
        test_manager: Instance of ClimbingTestManager to fetch test results.
        admin_id: Admin ID to filter data by administrator.
    """

    def __init__(self,
                 climber_db: str = "climbers_database.db",
                 tests_db: str = "tests_database.db",
                 admin_id: int = 1):
        # Initialize database managers
        self.climber_manager = ClimberDatabaseManager(db_name=climber_db)
        self.test_manager = ClimbingTestManager(db_name=tests_db)
        self.admin_id = admin_id

    def load_data(self, test_type: str = "ao") -> pd.DataFrame:
        """
        Load and merge climber demographic data with test metrics.
        """
        # Fetch test entries for this admin - pass admin_id as an integer
        tests = self.test_manager.fetch_results_by_admin(self.admin_id)
        print(f"Number of tests fetched: {len(tests)}")

        records = []
        for entry in tests:
            # Filter by test_type
            if entry.get('test_type') != test_type:
                continue
            
            # Debugging: Show the entry structure
            print(f"Processing entry for test_type '{test_type}'")

            # Get participant_id as an integer
            climber_id = entry.get('participant_id')
            # Ensure climber_id is an integer
            if isinstance(climber_id, str):
                try:
                    climber_id = int(climber_id)
                except ValueError:
                    print(f"Warning: Could not convert participant_id '{climber_id}' to integer. Skipping record.")
                    continue
                
            # Fetch climber info with integer IDs
            user_data = self.climber_manager.get_user_data(self.admin_id, climber_id)
            
            # Check if user_data is None before trying to access it
            if user_data is None:
                print(f"No user data found for climber ID: {climber_id}")
                continue
            
            # Convert gender values to numeric format
            if user_data.get('gender') == 'Female':
                user_data['gender'] = 1
            elif user_data.get('gender') == 'Male':
                user_data['gender'] = 0
            else:
                user_data['gender'] = 2

            if user_data.get('dominant_arm') == 'Left':
                user_data['dominant_arm'] = 1
            elif user_data.get('dominant_arm') == 'Right':
                user_data['dominant_arm'] = 0
            else:
                user_data['dominant_arm'] = None

            # ===== IMPROVED TEST RESULTS HANDLING =====
            # Parse test_results JSON
            raw_metrics = entry.get('test_results')
            print(f"Raw metrics type: {type(raw_metrics)}")
            
            # Different parsing strategies based on the data format
            metrics = {}
            if raw_metrics:
                # Case 1: Already a dictionary
                if isinstance(raw_metrics, dict):
                    metrics = raw_metrics
                    print("Using raw_metrics directly as it's already a dictionary")
                # Case 2: Double-encoded JSON string (happens when a JSON string gets re-encoded)
                elif isinstance(raw_metrics, str):
                    # Try direct JSON parsing
                    try:
                        metrics = json.loads(raw_metrics)
                        print("Successfully parsed test_results with json.loads()")
                    except json.JSONDecodeError:
                        print(f"JSON decode error for: {raw_metrics[:100]}...")
                        # Try alternative parsing methods
                        try:
                            import ast
                            # Try Python's literal_eval for dictionary-like strings
                            if raw_metrics.startswith('{') and raw_metrics.endswith('}'):
                                metrics = ast.literal_eval(raw_metrics)
                                print("Successfully parsed with ast.literal_eval()")
                            else:
                                # Manual parsing for other formats
                                metrics = self._parse_dict_string(raw_metrics)
                                print("Used manual parsing for test_results")
                        except Exception as e:
                            print(f"Error in alternative parsing: {str(e)}")
                            metrics = {}
                else:
                    print(f"Unexpected type for test_results: {type(raw_metrics)}")
                    metrics = {}

            # Ensure all metric values are proper types (not strings)
            for key, value in metrics.items():
                if isinstance(value, str):
                    try:
                        # Convert strings to appropriate numeric types
                        if '.' in value:
                            metrics[key] = float(value)
                        else:
                            metrics[key] = int(value)
                        print(f"Converted metric {key} from string '{value}' to {type(metrics[key])}")
                    except ValueError:
                        # Keep as string if conversion fails
                        print(f"Could not convert metric {key}='{value}' to numeric")
                        pass
            
            # Debug the extracted metrics
            print(f"Final extracted metrics (keys): {list(metrics.keys())}")
            for key, value in metrics.items():
                print(f"  {key}: {value} ({type(value)})")

            # Combine into single record - ensure participant_id is stored as an integer
            record = {**user_data, **metrics, 'participant_id': climber_id}
            records.append(record)

        df = pd.DataFrame(records)

        # Replace 'N/A' values with NaN for all columns
        for col in df.columns:
            df[col] = df[col].replace('N/A', np.nan)
        
        # Debug the final DataFrame
        print("\nFinal DataFrame:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        if not df.empty:
            print(f"  Column types:")
            for col in df.columns:
                print(f"    {col}: {df[col].dtype}")
        
        return df

    @staticmethod
    def prepare_features(df: pd.DataFrame,
                         include_metrics: list = None,
                         include_demographics: list = None) -> (pd.DataFrame, pd.Series):
        """
        Prepare feature matrix X and target vector y from merged DataFrame.

        Args:
            df: Merged DataFrame from load_data().
            include_metrics: List of metric keys to include (e.g., ["max_strength", "sum_work_above_cf", "critical_force"]).
            include_demographics: List of demographic columns to include (e.g., ["gender", "age", "years_climbing"]).

        Returns:
            X: DataFrame of predictor variables (one-hot encoding applied where needed).
            y: Series of target variable (ircra rating).
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

        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Target variable - convert to numeric, keeping only valid values
        if 'ircra' in df_copy.columns:
            df_copy['ircra'] = pd.to_numeric(df_copy['ircra'], errors='coerce')
            # Remove rows where target is NaN
            valid_indices = ~df_copy['ircra'].isna()
            df_copy = df_copy.loc[valid_indices]
            y = df_copy['ircra']
        else:
            print("Warning: 'ircra' column not found in data")
            y = None
            valid_indices = pd.Series(True, index=df_copy.index)

        # Ensure each required column exists with default values where needed
        for col in include_demographics:
            if col not in df_copy.columns:
                print(f"Adding missing demographic column: {col}")
                df_copy[col] = 0

        for col in include_metrics:
            if col not in df_copy.columns:
                print(f"Adding missing metric column: {col}")
                df_copy[col] = 0

        # Identify which columns to treat as categorical vs numeric
        categorical_cols = []
        numeric_cols = []

        # Check each demographic column
        for col in include_demographics:
            if col not in df_copy.columns:
                continue

            # Check if column contains string/object data or known categorical columns
            if (df_copy[col].dtype == 'object' or
                    col in ['gender', 'dominant_arm']):
                categorical_cols.append(col)
                print(f"Treating '{col}' as categorical with values: {df_copy[col].unique()}")
            else:
                # These are numeric demographics (age, weight, etc.)
                numeric_cols.append(col)
                # Ensure they're numeric
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                print(f"Treating '{col}' as numeric")

        # All metrics should be numeric
        for col in include_metrics:
            if col in df_copy.columns:
                numeric_cols.append(col)
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # Select features (both categorical and numeric)
        X = df_copy[categorical_cols + numeric_cols].copy()

        # Make sure gender and other categorical values are properly formatted
        # This ensures consistent one-hot encoding results
        if 'gender' in X.columns:
            # Convert to numeric if needed
            X['gender'] = pd.to_numeric(X['gender'], errors='coerce').fillna(2)

        if 'dominant_arm' in X.columns:
            # Convert to numeric if needed
            X['dominant_arm'] = pd.to_numeric(X['dominant_arm'], errors='coerce').fillna(2)

        # One-hot encode categorical variables - IMPORTANT: using drop_first=False for consistency
        if categorical_cols:
            print(f"One-hot encoding categorical columns: {categorical_cols}")
            # Use drop_first=False to ensure consistent column generation
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        # Ensure critical feature columns exist (those mentioned in error)
        required_columns = ['climbing_freq', 'gender_1', 'sport_freq']
        for col in required_columns:
            if col not in X.columns:
                print(f"Adding missing required column: {col}")
                X[col] = 0

        # Fill any remaining NaN values in numeric columns with median
        for col in X.select_dtypes(include='number').columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"Filled NaN values in '{col}' with median: {median_val}")

        return X, y

    @staticmethod
    def _parse_dict_string(dict_str):
        """
        Manually parse a string that looks like a Python dictionary.
        This is a fallback for when JSON parsing fails.
        """
        metrics = {}
        # Remove brackets and split by commas
        if dict_str.startswith('{'):
            dict_str = dict_str[1:]
        if dict_str.endswith('}'):
            dict_str = dict_str[:-1]

        pairs = dict_str.split(',')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")

            # Try to convert value to numeric if possible
            try:
                if '.' in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value

        return metrics

    def fit_pca(self,
                X: pd.DataFrame,
                y: pd.Series = None,
                threshold: float = 0.95,
                save_path: str = "pca_model.pkl") -> pd.DataFrame:
        """
        Fit PCA on standardized features, save scaler and PCA model, and return transformed DataFrame.

        Args:
            X: DataFrame of features.
            y: Optional Series of target values to prepend to output.
            threshold: Cumulative explained variance threshold to select n_components.
            save_path: File path to pickle the scaler and PCA objects.

        Returns:
            principal_df: DataFrame containing principal components (and y, if provided).
        """
        # First, make a copy to avoid modifying the original
        X = X.copy()

        # Check for NaN values and identify problematic columns
        nan_counts = X.isna().sum()
        nan_count_total = nan_counts.sum()

        if nan_count_total > 0:
            print(f"Warning: Found {nan_count_total} NaN values in the input data.")
            print("NaN counts per column:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaNs ({count / len(X) * 100:.1f}%)")

            # Remove columns that are entirely NaN (or above threshold)
            all_nan_cols = [col for col in X.columns if X[col].isna().sum() == len(X)]
            if all_nan_cols:
                print(f"Dropping columns that are entirely NaN: {all_nan_cols}")
                X = X.drop(columns=all_nan_cols)

            # Handle columns with partial NaN values
            # Option 1: Drop rows with NaN values if not too many
            nan_rows = X.isna().any(axis=1).sum()
            if nan_rows <= X.shape[0] * 0.2:  # If less than 20% of rows have NaNs
                print(f"Dropping {nan_rows} rows with missing values.")
                nan_indices = X.isna().any(axis=1)
                X = X[~nan_indices]
                if y is not None:
                    y = y[~nan_indices].copy()
            else:
                # Option 2: Impute missing values
                print("Imputing missing values for each column.")
                for col in X.columns:
                    # Skip columns that are entirely NaN (should be removed already, but just in case)
                    if X[col].isna().all():
                        print(f"Column {col} is all NaN values. Consider removing this column.")
                        continue

                    # For numeric columns
                    if pd.api.types.is_numeric_dtype(X[col]):
                        # Get non-NaN values
                        non_nan_values = X[col].dropna()
                        if len(non_nan_values) > 0:
                            # Use median for imputation (more robust than mean)
                            fill_value = non_nan_values.median()
                            X[col] = X[col].fillna(fill_value)
                            print(f"  Imputed {col} with median: {fill_value}")
                    # For categorical/object columns
                    else:
                        # Get most common value excluding NaN
                        non_nan_values = X[col].dropna()
                        if len(non_nan_values) > 0:
                            mode_values = non_nan_values.mode()
                            if not mode_values.empty:
                                fill_value = mode_values[0]
                                X[col] = X[col].fillna(fill_value)
                                print(f"  Imputed {col} with mode: {fill_value}")
                            else:
                                # Fallback if mode is empty
                                X[col] = X[col].fillna("Unknown")
                                print(f"  Imputed {col} with 'Unknown'")

        # Final check for any remaining NaNs
        remaining_nans = X.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: Still found {remaining_nans} NaN values after imputation.")
            print("Replacing remaining NaNs with 0 for numeric columns and 'Unknown' for others")

            # Last resort: replace any remaining NaNs with zeros or 'Unknown'
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna('Unknown')

        # Verify data is clean
        final_check = X.isna().sum().sum()
        if final_check > 0:
            print(f"ERROR: Still have {final_check} NaN values after all imputation steps!")
            raise ValueError("Failed to clean all NaN values from input data")

        # Try to detect non-numeric columns that might have been missed
        non_numeric_cols = []
        for col in X.columns:
            try:
                # This will raise exception if column contains non-numeric strings
                X[col].astype(float)
            except:
                non_numeric_cols.append(col)

        if non_numeric_cols:
            print(f"Warning: Found non-numeric columns that need encoding: {non_numeric_cols}")
            # Apply one-hot encoding to these columns
            X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

        # Convert all data to float to ensure compatibility with StandardScaler and PCA
        for col in X.columns:
            X[col] = X[col].astype(float)

        # Standardize the data
        print("Standardizing data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Verify no NaNs in scaled data
        if np.isnan(X_scaled).any():
            print("Warning: NaNs found in scaled data. Replacing with zeros.")
            nan_count = np.isnan(X_scaled).sum()
            print(f"Number of NaNs in scaled data: {nan_count}")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # If we have fewer than 2 samples, PCA cannot be reliably calculated
        if X_scaled.shape[0] < 2:
            print("Warning: Insufficient number of samples for PCA. Returning standardized features.")
            pc_cols = [f'PC{i + 1}' for i in range(X_scaled.shape[1])]
            principal_df = pd.DataFrame(X_scaled, columns=pc_cols)
            if y is not None:
                principal_df.insert(0, 'ircra', y.values)
            # Save only the scaler, PCA will not be calculated
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'scaler': scaler, 'pca': None}, f)
            return principal_df

        # Full PCA to determine component count
        print("Fitting PCA to determine optimal components...")
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = max(1, int(np.argmax(cum_var >= threshold) + 1))
        print(f"Selected {n_components} components explaining {cum_var[n_components - 1] * 100:.1f}% of variance")

        # Fit PCA with optimal components
        pca = PCA(n_components=n_components)
        PCs = pca.fit_transform(X_scaled)

        # Build DataFrame of PCs
        pc_cols = [f'PC{i + 1}' for i in range(n_components)]
        principal_df = pd.DataFrame(PCs, columns=pc_cols)

        # Prepend target if available
        if y is not None:
            principal_df.insert(0, 'ircra', y.values)

        # Save scaler and PCA
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'pca': pca}, f)

        print("PCA completed successfully!")
        return principal_df

# Example usage
if __name__ == '__main__':
    processor = MLDataProcessor(admin_id=1,
                               climber_db='climber_database.db',
                               tests_db='tests_database.db')
    df = processor.load_data()

    # Debug: Print the columns and first few rows
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame shape:", df.shape)
    print("First few rows:")
    print(df.head())

    # Check if 'ircra' is in the columns
    if 'ircra' not in df.columns:
        print("Column 'ircra' is missing. Available columns are:", df.columns.tolist())

        # Create feature matrix without target variable
        X, _ = processor.prepare_features(df)  # Pass none as y
        
        print("Created feature matrix X with shape:", X.shape)
        print("Columns in X:", X.columns.tolist())
    else:
        # Continue with original code if 'ircra' exists
        X, y = processor.prepare_features(df)
        pca_df = processor.fit_pca(X, y, threshold=0.95, save_path='models/pca_model.pkl')

        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        pca_df.to_excel('outputs/pca_data.xlsx', index=False)