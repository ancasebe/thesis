#!/usr/bin/env python3
import time
import sqlite3
import numpy as np
import pandas as pd
from gui.test_page.evaluations.force_evaluation import ForceMetrics  # for evaluating force and rep metrics

# Fix the import paths to match the actual location
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager


# Function to check if a participant exists in the database using JSON extraction
def get_existing_participant(email, name, surname, climber_db_path="climber_database.db"):
     """
     Look up a climber by email, name and surname.
     Returns the existing climber.id or None.
     """
     conn = sqlite3.connect(climber_db_path)
     cur = conn.cursor()
     cur.execute("""
         SELECT id FROM climbers
         WHERE json_extract(basic_info, '$.email') = ? 
         AND json_extract(basic_info, '$.name') = ? 
         AND json_extract(basic_info, '$.surname') = ?
     """, (email, name, surname))
     row = cur.fetchone()
     conn.close()
     return row[0] if row else None


# Helper function to convert NumPy types to native Python types
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


# Function to prepare force data for detection of repetitions
def prepare_force_data(data_column, start_time=None, sample_rate=100):
    """
    Prepares force data for repetition detection by ensuring proper scaling and timestamps.
    
    Args:
        data_column: The force values as a pandas Series or numpy array
        start_time: Start time in epoch seconds, defaults to current time
        sample_rate: Sample rate in Hz
        
    Returns:
        DataFrame with proper 'time' and 'value' columns
    """
    if start_time is None:
        start_time = time.time()
    
    # Convert to numpy array if needed
    if isinstance(data_column, pd.Series):
        force_values = data_column.values
    else:
        force_values = np.array(data_column)
    
    # Normalize/scale the data if needed
    # Check if force values are too small (might need scaling)
    max_force = np.max(force_values)
    if max_force < 1.0:
        force_values = force_values * 100  # Scale up very small values
    
    # Create a time series with the proper sampling rate
    n_samples = len(force_values)
    # time_values = np.linspace(start_time, start_time + 0.01, n_samples)
    time_values = start_time + np.arange(n_samples) / sample_rate

    # Create DataFrame
    df = pd.DataFrame({
        'time': time_values,
        'value': force_values
    })
    
    return df


# ---------------------
# Main function to transfer data from the Excel file into the two databases.
def transfer_from_excel(excel_file_path):
    # Read the Excel file: assume the first sheet contains test measurements
    # and the "categories" sheet holds participant/test meta-data.
    print("Reading Excel file...")
    df_tests = pd.read_excel(excel_file_path, sheet_name=0)
    df_categories = pd.read_excel(excel_file_path, sheet_name="categories")

    # Check that the tests sheet contains a 'time' column
    original_time_column_exists = 'time' in df_tests.columns
    if not original_time_column_exists:
        print("No 'time' column found in the Excel file. Will generate time values.")
    
    # Get the list of test columns (all columns except 'time').
    test_columns = [col for col in df_tests.columns if col.lower() != "time"]

    # For simplicity, we assume that the order of test columns in df_tests corresponds
    # to the rows in the categories sheet.
    if len(test_columns) != len(df_categories):
        print("Warning: the number of test columns does not equal the number of rows in the categories sheet.")

    # Initialize the database managers.
    climber_manager = ClimberDatabaseManager()  # Uses its default database (e.g. climber_database.db)
    test_manager = ClimbingTestManager()          # Uses its default test DB

    # climber_db_path = "../databases/climber_database.db"

    # Create a directory for feather files if it doesn't exist.
    # feather_dir = "./tests"
    # os.makedirs(feather_dir, exist_ok=True)

    # Cache participant IDs by email to avoid duplicate lookups.
    # participant_cache = {}

    # Base time for all tests (to ensure consistent timestamps)
    # base_time = time.time()

    print("Starting transfer for each test column...\n")
    # Iterate over each test column and its corresponding categories row (by order).
    for idx, test_col in enumerate(test_columns):
        base_time = time.time()
        print(f"Processing test: {test_col}")
        # Get the corresponding categories row.
        try:
            cat_row = df_categories.iloc[idx]
        except IndexError:
            print(f"No matching row in 'categories' for test column {test_col}. Skipping.")
            continue

        # Determine gender for the participant.
        if cat_row.get("gender") == 'm':
            gender = 'Male'
        elif cat_row.get("gender") == 'f':
            gender = 'Female'
        else:
            gender = 'Other'

        print(cat_row.get("d_arm"))
        # Determine dominant arm from the 'arm_tested' field.
        if cat_row.get("d_arm") == 'r':
            dominant_arm = 'Right'
        elif cat_row.get("d_arm") == 'l':
            dominant_arm = 'Left'
        else:
            dominant_arm = '-'

        # --- Participant / Climber Data ---
        climber_data = {
            "name": cat_row.get("name"),
            "surname": cat_row.get("surname"),
            "email": cat_row.get("mail"),
            "gender": gender,
            "dominant_arm": dominant_arm,
            "weight": float(cat_row.get("bm")) if pd.notna(cat_row.get("bm")) else None,
            "height": float(cat_row.get("height")) if pd.notna(cat_row.get("height")) else None,
            "age": int(cat_row.get("age")) if pd.notna(cat_row.get("age")) else None,
            "ircra": str(cat_row.get("IRCRA")),
            "climbing_freq": cat_row.get("climbing"),
            "climbing_hours": float(cat_row.get("hours_c")) if pd.notna(cat_row.get("hours_c")) else None,
            "years_climbing": float(cat_row.get("yoc")) if pd.notna(cat_row.get("yoc")) else None,
            "bouldering": cat_row.get("bouldering"),
            "lead_climbing": cat_row.get("rope"),
            "climbing_indoor": cat_row.get("indoor"),
            "climbing_outdoor": cat_row.get("outdoor"),
            "sport_other": cat_row.get("activity"),
            "sport_freq": cat_row.get("other"),
            "sport_activity_hours": float(cat_row.get("hours_o")) if pd.notna(cat_row.get("hours_o")) else None,
            # Use the same base_time as the timestamp.
            "timestamp": base_time
        }

        # Convert any NumPy types to native Python types
        climber_data = convert_numpy_types(climber_data)

        print('Climber data:', climber_data)

        # Manual check for existing climber before attempting registration
        conn = climber_manager.connection
        cursor = conn.cursor()

        # Check if a climber with the same email, name, and surname exists
        cursor.execute("""
                SELECT id FROM climbers 
                WHERE json_extract(basic_info, '$.email') = ? 
                AND json_extract(basic_info, '$.name') = ?
                AND json_extract(basic_info, '$.surname') = ?
            """, (climber_data["email"], climber_data["name"], climber_data["surname"]))

        existing_climber = cursor.fetchone()

        if existing_climber:
            participant_id = existing_climber[0]
            print(f"Using existing climber with ID: {participant_id}")
        else:
            # Register new climber
            participant_id = climber_manager.register_climber(admin_id=1, **climber_data)
            if participant_id is None:
                print("Failed to register climber. Aborting data transfer.")
                climber_manager.close()
                return
            print(f"New climber registered with ID: {participant_id}")

        # --- Create Feather File for Test Data ---
        # For the Feather file, prepare force data with proper time and value columns
        test_data_series = df_tests[test_col]
        
        # Prepare the force data with proper time values and scaling if needed
        df_feather = prepare_force_data(
            test_data_series, 
            start_time=base_time,
            # duration_seconds=60,  # Adjust based on your data
            sample_rate=100
        )
        
        # Visualize data distribution to help diagnose rep detection issues
        force_values = df_feather['value'].values
        min_force = np.min(force_values)
        max_force = np.max(force_values)
        mean_force = np.mean(force_values)
        std_force = np.std(force_values)
        
        print(f"Force stats - Min: {min_force:.2f}, Max: {max_force:.2f}, Mean: {mean_force:.2f}, Std: {std_force:.2f}")
        
        # Build the filename using the test column name and the base timestamp.
        save_timestamp = str(base_time).replace('.', '_')
        # feather_filename = os.path.join(feather_dir, f"ao_{idx}_force_{save_timestamp}.feather")
        feather_filename = f"/Users/annasebestikova/PycharmProjects/thesis/gui/test_page/tests/ao_{idx}_force_{save_timestamp}.feather"

        try:
            df_feather.to_feather(feather_filename)
            print(f"Feather file created: {feather_filename}")
        except Exception as e:
            print(f"Error writing feather file for {test_col}: {e}")
            continue

        # --- Compute Force and Repetition Metrics using ForceMetrics ---
        try:
            fm = ForceMetrics(feather_filename, test_type="ao")
            test_results, rep_results = fm.evaluate()
            number_of_reps = len(rep_results)
            print('Number of Reps:', number_of_reps)
        except Exception as e:
            print("Error evaluating force metrics:", e)
            test_results, rep_results, number_of_reps = {}, [], 0

        # --- Prepare Test Data for tests_database ---
        # Set data_type to "force" and test_type to "ao" as per requirements.
        arm_tested = cat_row.get("arm_tested")
        print('Force file db:', feather_filename)
        test_data = {
            "arm_tested": arm_tested.lower() if arm_tested else "d",  # Default to dominant if not specified
            "data_type": "force",
            "test_type": "ao",
            "timestamp": base_time,  # Use the base_time for the test timestamp.
            "number_of_reps": number_of_reps,
            "force_file": feather_filename,
            "nirs_file": "",  # Not used since only force (sensor 1) data is present.
            "test_results": test_results,
            "nirs_results": {},
            "rep_results": rep_results
        }

        # Convert any NumPy types to native Python types
        # test_data = convert_numpy_types(test_data)

        print(test_data)

        # --- Insert the Test Data into tests_database ---
        try:
            test_manager.add_test_result(admin_id="1", participant_id=str(participant_id), db_data=test_data)
            print(f"Test result added for participant ID {participant_id}.\n")
        except Exception as e:
            print(f"Error adding test result for {test_col}: {e}\n")

    # Close connection to the test database.
    test_manager.close_connection()
    print("Data transfer from Excel complete.")


# ---------------------
# Main entry point.
if __name__ == "__main__":
    # Change the path to your Excel file as needed.
    excel_file = "complete_dataset_new.xlsx"
    transfer_from_excel(excel_file)
