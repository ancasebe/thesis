"""
IRCRA prediction model training module for the Climbing Testing Application.

This module defines the PredictionIRCRA class which handles the training and
evaluation of machine learning models for predicting climbing performance levels.
It implements data preprocessing, feature engineering, dimensionality reduction
(PCA), and support vector regression (SVR) for IRCRA grade prediction.

Key functionalities:
- Load and process test and climber data
- Prepare and engineer features for model training
- Perform principal component analysis (PCA)
- Train and optimize support vector regression models
- Generate visualizations of model performance
- Save trained models for later use in prediction

The prediction model training module provides the statistical foundation for
the application's performance prediction capabilities.
"""

import os
import matplotlib
# Set non-interactive backend for thread safety
matplotlib.use('Agg')  # This must be called before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # Library for linear regression model
import joblib

from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager


class PredictionIRCRA:
    """
    A class that handles IRCRA prediction model training and evaluation.
    
    This class loads climber and test data from databases, prepares features,
    performs dimensionality reduction using PCA, and trains regression models
    for predicting climber IRCRA levels.
    
    Attributes:
        climber_manager: Instance of ClimberDatabaseManager to access climber data.
        test_manager: Instance of ClimbingTestManager to access test results.
        admin_id: ID of the administrator whose data is being analyzed.
        model_dir: Directory path where trained models will be saved.
        plots_dir: Directory path where visualizations will be saved.
    """

    def __init__(self,
                 climber_db: str = "climber_database.db",
                 tests_db: str = "tests_database.db",
                 admin_id: int = 1):
        """
        Initialize the PredictionIRCRA class with database connections and paths.
        
        Args:
            climber_db: Path to the climber database file.
            tests_db: Path to the tests database file.
            admin_id: ID of the administrator whose data should be used.
        """
        # Initialize database managers
        self.climber_manager = ClimberDatabaseManager(db_name=climber_db)
        self.test_manager = ClimbingTestManager(db_name=tests_db)
        self.admin_id = admin_id
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create plots directory if it doesn't exist
        self.plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_data(self, test_type: str = "ao") -> pd.DataFrame:
        """
        Load and merge climber demographic data with test metrics.
        
        Args:
            test_type: Type of test to filter by (default: "ao").
            
        Returns:
            DataFrame containing merged climber and test data.
        """
        # Fetch test entries for this admin - pass admin_id as an integer
        tests = self.test_manager.fetch_results_by_admin(self.admin_id)
        print(f"Number of tests fetched: {len(tests)}")

        records = []
        for entry in tests:
            # Filter by test_type
            if entry.get('test_type') != test_type:
                continue

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
            # df[col] = df[col].replace('N/A', np.nan)
            df[col] = df[col].replace('N/A', np.nan).infer_objects(copy=False)

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
            include_metrics: List of metric keys to include (default: max_strength, 
                             sum_work, sum_work_above_cf, critical_force, rfd_norm_overall).
            include_demographics: List of demographic columns to include (default: gender, 
                                  age, years_climbing, bouldering, climbing_indoor, etc.).

        Returns:
            Tuple containing:
            - corr_df: DataFrame for correlation analysis including target and features.
            - X: DataFrame of predictor variables.
            - y: Series of target variable (IRCRA rating).
        """
        # Default columns if not specified
        if include_demographics is None:
            include_demographics = [
                'gender', 'age', 'years_climbing', 'bouldering', 'climbing_indoor',
                'dominant_arm', 'weight', 'height', 'climbing_freq'
            ]
        if include_metrics is None:
            include_metrics = ['max_strength', 'sum_work', 'sum_work_above_cf',
                               'critical_force', 'rfd_norm_overall']

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

        include_cols = []

        for col in include_demographics:
            if col in df_copy.columns:
                include_cols.append(col)
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            else:
                print(f"Adding missing demographic column: {col}")
                df_copy[col] = 0

        for col in include_metrics:
            if col in df_copy.columns:
                include_cols.append(col)
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            else:
                print(f"Adding missing metric column: {col}")
                df_copy[col] = 0

        X = df_copy[include_cols].copy()
        corr_df = df_copy[['ircra'] + include_cols].copy()

        # Drop rows with NaN values
        X = X.dropna()
        # Keep y aligned with X
        if y is not None:
            y = y.loc[X.index]
        print(f"Dropped rows with NaN values. Remaining rows: {len(X)}")

        return corr_df, X, y

    def corr_matrix(self, df):
        """
        Generate and save a correlation matrix visualization.
        
        Args:
            df: DataFrame containing the data for correlation analysis.
            
        Returns:
            Path to the saved correlation matrix image file.
        """
        correlation_matrix = df.corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title('Correlation Matrix')
        
        # Save figure instead of displaying
        save_path = os.path.join(self.plots_dir, 'correlation_matrix.png')
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        return save_path

    def pca_df(self, X_raw, y, save_model=True):
        """
        Perform PCA, generate visualizations, and optionally save the models.
        
        Args:
            X_raw: DataFrame of raw features.
            y: Series containing target variable.
            save_model: Whether to save the PCA model for future use (default: True).
            
        Returns:
            DataFrame containing PCA components and target variable, or None if insufficient data.
        """
        # Check if we have valid data
        if X_raw is None or y is None or len(X_raw) == 0 or len(y) == 0:
            print("Not enough valid data for PCA.")
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Full PCA to determine optimal components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        ideal = int(np.argmax(cum_var >= 0.95) + 1)

        # Plot explained variance
        fig = plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, alpha=0.6, label='Individual')
        plt.step(range(1, len(cum_var) + 1), cum_var, where='mid',
                 label='Cumulative')
        plt.axvline(ideal, linestyle='--', color='red',
                    label=f'{ideal} PCs for 95% var')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        plt.legend()
        plt.title('PCA Explained Variance')
        
        # Save to plots directory
        pca_variance_path = os.path.join(self.plots_dir, 'pca_explained_variance.png')
        fig.savefig(pca_variance_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Fit PCA with optimal components
        pca = PCA(n_components=ideal)
        PCs = pca.fit_transform(X_scaled)
        pc_cols = [f'PC{i + 1}' for i in range(ideal)]
        principal_df = pd.DataFrame(PCs, columns=pc_cols)
        
        # Create final DataFrame with target variable and PC components
        final_df = pd.concat([y.reset_index(drop=True), principal_df.reset_index(drop=True)], axis=1)

        # Scatter of first two PCs
        if ideal >= 2:
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PC1', y='PC2', hue='ircra', data=final_df,
                            palette='coolwarm', s=50, alpha=0.8, edgecolor='k')
            plt.title('PCA Scatter: PC1 vs PC2')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.axhline(0, color='grey', lw=0.5)
            plt.axvline(0, color='grey', lw=0.5)
            plt.grid(True)
            
            # Save scatter plot
            pca_scatter_path = os.path.join(self.plots_dir, 'pca_scatter.png')
            fig.savefig(pca_scatter_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory

        # Save scaler + PCA for reuse
        if save_model:
            save_path = os.path.join(self.model_dir, 'pca_pipeline.joblib')
            joblib.dump({'scaler': scaler, 'pca': pca}, save_path)
            print(f"PCA model saved to: {save_path}")

        print(f"PCA completed with {ideal} components.")
        
        # Return the data and the path to the plot
        return final_df

    @staticmethod
    def prepare_pca_features(pca_df):
        """
        Prepare PCA-transformed features for regression models.
        
        Args:
            pca_df: DataFrame with PCA components and target variable.
            
        Returns:
            Tuple containing:
            - X: DataFrame of PCA components as features.
            - y: Series of target variable (IRCRA rating).
        """
        df_copy = pca_df.copy()

        # Extract target variable
        if 'ircra' in df_copy.columns:
            y = df_copy['ircra']
            # Remove target from features
            X = df_copy.drop('ircra', axis=1)
        else:
            print("Warning: 'ircra' column not found in PCA data")
            y = None
            X = df_copy.copy()

        # Drop rows with NaN values in either X or y
        if y is not None:
            # Find rows where either X or y has NaN
            missing_mask = X.isna().any(axis=1) | y.isna()
            if missing_mask.any():
                # Remove rows with NaN
                X = X.loc[~missing_mask]
                y = y.loc[~missing_mask]
                print(f"Dropped rows with NaN values. Remaining rows: {len(X)}")
        else:
            # If no target variable, just drop NaN rows from X
            X = X.dropna()
            print(f"Dropped rows with NaN values. Remaining rows: {len(X)}")

        return X, y

    def train_best_svr(self, X, y, kernels=None, C_params=None):
        """
        Train SVR models with different configurations and find the best model.
        
        Args:
            X: DataFrame of features.
            y: Series of target values.
            kernels: List of kernel types to try (default: ["poly", "rbf", "sigmoid"]).
            C_params: Array of regularization parameters (default: [0.1, 1, 5, 10, 20, 100]).
            
        Returns:
            Tuple containing:
            - mse_pcr: Mean squared error for PCR model.
            - mse_svr: Mean squared error for SVR model.
            - svr_results_path: Path to the SVR results visualization.
            - pcr_results_path: Path to the PCR results visualization.
        """
        if kernels is None:
            kernels = ["poly", "rbf", "sigmoid"]
        if C_params is None:
            C_params = np.array([0.1, 1, 5, 10, 20, 100])
            
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        df_accuracy = pd.DataFrame({"kernel": [], "C": [], "fit_mse": [], "validation_mse": []})

        for kernel in kernels:
            for C in C_params:
                regre = svm.SVR(kernel=kernel, C=C)
                regre.fit(x_train, y_train)
                y_pred_train = regre.predict(x_train)
                y_pred_test = regre.predict(x_test)
                mse_fit = mean_squared_error(y_pred_train, y_train)
                mse_val = mean_squared_error(y_pred_test, y_test)
                print(f"Kernel: {kernel}. Regularization parameter C: {C:.0e}.")
                print(f"\t - Training error (MSE): {mse_fit:.2f}. Validation error (MSE): {mse_val:.2f}.")
                df_accuracy.loc[df_accuracy.shape[0]] = [kernel, C, mse_fit, mse_val]

        best_regr = df_accuracy[df_accuracy.validation_mse == df_accuracy.validation_mse.min()].iloc[0]

        regre = svm.SVR(kernel=best_regr['kernel'], C=best_regr['C'])
        regre.fit(x_train, y_train)
        y_pred_test_svr = regre.predict(x_test)
        svr_path = os.path.join(self.model_dir, 'svr_model.joblib')
        joblib.dump(regre, svr_path)

        # Create scatter plot
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test_svr, alpha=0.7, color="darkblue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('IRCRA prediction with SVR')
        
        # Save SVR scatter plot
        svr_scatter_path = os.path.join(self.plots_dir, 'svr_sel_scatter.png')
        fig.savefig(svr_scatter_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Plot SVR results
        svr_results_path = self.plot_regression_results(
            x_test, y_test, y_pred_test_svr, 
            "Prediction of IRCRA level using SVR",
            "svr_accuracy.png"
        )

        # Train and evaluate Linear Regression for comparison
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)
        y_pred_test_linreg = linreg.predict(x_test)
        linreg_path = os.path.join(self.model_dir, 'linreg_model.joblib')
        joblib.dump(linreg, linreg_path)
        
        # Plot Linear Regression results
        pcr_results_path = self.plot_regression_results(
            x_test, y_test, y_pred_test_linreg, 
            "Prediction of IRCRA level using PCR",
            "pcr_accuracy.png"
        )

        # Create PCR scatter plot
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test_linreg, alpha=0.7, color="darkblue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('IRCRA prediction with PCR')
        
        # Save PCR scatter plot
        pcr_scatter_path = os.path.join(self.plots_dir, 'pcr_sel_scatter.png')
        fig.savefig(pcr_scatter_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        mse_pcr = mean_squared_error(y_pred_test_linreg, y_test)
        mse_svr = mean_squared_error(y_pred_test_svr, y_test)

        print(
            f"Linear regression error: {mean_squared_error(y_pred_test_linreg, y_test)}, SVR error: {mean_squared_error(y_pred_test_svr, y_test)}.")

        return mse_pcr, mse_svr, svr_results_path, pcr_results_path

    def plot_regression_results(self, x_test, y_test, y_pred_test, title, filename):
        """
        Create and save visualization of regression model results.
        
        Args:
            x_test: DataFrame of test features.
            y_test: Series of actual target values.
            y_pred_test: Array of predicted target values.
            title: Title for the plot.
            filename: Filename to save the plot.
            
        Returns:
            Path to the saved plot file.
        """
        x_dummy = np.linspace(1, x_test.shape[0], x_test.shape[0])

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(title, fontsize=20)
        plt.subplots_adjust(top=0.95, bottom=0.01, left=0.0, right=1.0, hspace=0.01, wspace=0.0)

        axes[0].scatter(x_dummy, (y_pred_test - y_test) / y_test * 100, marker="o", color="white",
                        edgecolor="darkgreen")
        axes[0].set_ylabel(r"$percentage$ $error$ $[\%]$", fontsize=16)
        axes[0].set_xticks([])

        axes[1].scatter(x_dummy, y_test, color="darkblue", label="Real data")
        axes[1].scatter(x_dummy, y_pred_test, color="crimson", label="Estimated data")
        axes[1].set_xlabel(r"$Index$", fontsize=16)
        axes[1].set_ylabel(r"$IRCRA$", fontsize=16)
        axes[1].legend(fontsize=16)

        # Save the figure to the plots directory
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        
        return save_path