#!/usr/bin/env python3
import sqlite3
import pandas as pd

from gui.test_page.evaluations.nirs_evaluation import NIRSEvaluation
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager
from gui.test_page.evaluations.force_evaluation import ForceMetrics, find_test_interval  # for evaluating force and rep metrics

def get_test_and_participant_info(db_path, test_id):
    """
    Given a database path and a test ID, this function:
      1. Retrieves the row from 'tests' with the given ID.
      2. Extracts the participant_id from that row.
      3. Retrieves the participant row from 'participants' with that participant_id.
      4. Prints both rows along with their column names.

    Parameters:
      db_path (str): Path to the SQLite database file.
      test_id (int): The test ID to look up.
    """
    # Connect to the database and use sqlite3.Row to access columns by name
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Fetch the test row by test_id
    cur.execute("SELECT * FROM tests WHERE ID = ?", (test_id,))
    test_row = cur.fetchone()
    if test_row is None:
        print(f"No test found with ID {test_id}.")
        conn.close()
        return None, None

    # Print test row with column names
    print("Test row:")
    for key in test_row.keys():
        print(f"  {key}: {test_row[key]}")

    # 2. Extract the participant_id. Adjust the key if your column is named differently.
    participant_id = test_row["participant_id"]

    # 3. Fetch the participant row using the participant_id
    cur.execute("SELECT * FROM participants WHERE ID = ?", (participant_id,))
    participant_row = cur.fetchone()
    if participant_row is None:
        print(f"No participant found with ID {participant_id}.")
        conn.close()
        return test_row, None

    # Print participant row with column names
    print("\nParticipant row:")
    for key in participant_row.keys():
        print(f"  {key}: {participant_row[key]}")

    conn.close()
    return test_row, participant_row

def export_force_data(db_path, test_id, filename, sensor):
    """
    Exports measurements for a given sensor to a Feather file.
    For sensor 1 (force): export rows with time between start_time and end_time.
    For sensor 2 (NIRS): export rows with time between (start_time - 30) and (end_time + 30).
    The Feather file is saved under the provided filename.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT start_time, end_time FROM tests WHERE ID = ?", (test_id,))
    row = cur.fetchone()
    if row is None:
        print(f"No test found with ID {test_id}.")
        conn.close()
        return

    start_time, end_time = row

    if sensor == 1:
        query = """
            SELECT time, value
            FROM measurements
            WHERE sensor = 1 AND time >= ? AND time <= ?
        """
        cur.execute(query, (start_time, end_time))
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=["time", "value"])
        df.to_feather(filename)
        print(f"Exported sensor 1 data (force) to {filename}")

    elif sensor == 2:
        sensor2_start = start_time - 30
        sensor2_end = end_time + 30
        query = """
            SELECT time, value
            FROM measurements
            WHERE sensor = 2 AND time >= ? AND time <= ?
        """
        cur.execute(query, (sensor2_start, sensor2_end))
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=["time", "value"])
        df.to_feather(filename)
        print(f"Exported sensor 2 data (NIRS) to {filename}")

    conn.close()


def transfer_data(original_db_path, test_id):
    """
    Transfers test and participant data from the original DB into the climber and test databases.
    • admin_id is set to 1.
    • The test's start_time is used as the timestamp.
    • File paths use the pattern:
         Force: '/Users/annasebestikova/PycharmProjects/thesis/gui/test_page/tests/{test_type}_force_{start_time}.feather'
         NIRS:  '/Users/annasebestikova/PycharmProjects/thesis/gui/test_page/tests/{test_type}_nirs_{start_time}.feather'
    • data_type is "force" if only sensor 1 data exists or "force_nirs" if sensor 2 data is present.
    • arm_tested comes from the test table column 'tested_arm'.
    • test_type is derived from the test table column 'name' by stripping off "test_".
    • After exporting force data files, ForceMetrics is run on the force file
      and the computed test_results and rep_results are stored as JSON in the test DB.
    """
    # 1. Retrieve test and participant info
    test_row, participant_row = get_test_and_participant_info(original_db_path, test_id)
    if not test_row:
        print("Test not found. Aborting data transfer.")
        return
    if not participant_row:
        print("Participant info not found. Aborting data transfer.")
        return

    if participant_row["gender"] == 'm':
        gender = 'Male'
    elif participant_row["gender"] == 'f':
        gender = 'Female'
    else:
        gender = 'Other'

    if participant_row["d_arm"] == 'r':
        dominant_arm = 'Right'
    elif participant_row["d_arm"] == 'l':
        dominant_arm = 'Left'
    else:
        dominant_arm = '-'

    # 2. Prepare climber data for registration
    climber_data = {
        "name": participant_row["name"],
        "surname": participant_row["surname"],
        "email": participant_row["mail"],
        "gender": gender,
        "dominant_arm": dominant_arm,
        "weight": participant_row["bm"],
        "height": participant_row["height"],
        "age": participant_row["age"],
        "ircra": participant_row["ircra"],
        "climbing_freq": participant_row["climbing"],
        "climbing_hours": participant_row["hours_c"],
        "years_climbing": participant_row["yoc"],
        "bouldering": participant_row["bouldering"],
        "lead_climbing": participant_row["rope"],
        "climbing_indoor": participant_row["indoor"],
        "climbing_outdoor": participant_row["outdoor"],
        "sport_other": participant_row["activity"],
        "sport_freq": participant_row["other"],
        "sport_activity_hours": participant_row["hours_o"],
        "timestamp": participant_row["timestamp"]
    }
    print("Climber data:", climber_data)
    
    # 2.1 Connect to the climber database and check for existing climber
    climber_manager = ClimberDatabaseManager()  # Uses default DB name
    
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
    
    # 3. Process test information.
    start_time = test_row["start_time"]
    end_time = test_row["end_time"]
    tested_arm = test_row["tested_arm"]
    test_name = test_row["name"]  # e.g., "test_mvc"
    if test_name.startswith("test_"):
        test_type = test_name[len("test_"):]
    else:
        test_type = test_name

    # 4. Check for sensor 2 data.
    conn = sqlite3.connect(original_db_path)
    cur = conn.cursor()
    sensor2_count = 0
    if start_time is not None and end_time is not None:
        sensor2_start = start_time - 30
        sensor2_end = end_time + 30
        cur.execute(
            "SELECT COUNT(*) FROM measurements WHERE sensor = 2 AND time >= ? AND time <= ?",
            (sensor2_start, sensor2_end)
        )
        sensor2_count = cur.fetchone()[0]
    conn.close()

    if sensor2_count > 0:
        data_type = "force_nirs"
    else:
        data_type = "force"

    # 5. Build file paths using the test_type and formatted timestamp.
    start_time_str = str(start_time)
    save_timestamp = start_time_str.replace('.', '_')
    force_file = f"/Users/annasebestikova/PycharmProjects/thesis/gui/test_page/tests/{test_type}_force_{save_timestamp}.feather"
    nirs_file = ""

    # 6. Export force data to Feather files.
    export_force_data(original_db_path, test_id, force_file, sensor=1)
    if data_type == "force_nirs":
        nirs_file = f"/Users/annasebestikova/PycharmProjects/thesis/gui/test_page/tests/{test_type}_nirs_{save_timestamp}.feather"
        export_force_data(original_db_path, test_id, nirs_file, sensor=2)

    # 7. Compute force metrics and repetition metrics from the force file - store as JSON now
    try:
        fm = ForceMetrics(force_file, test_type=test_type)
        test_results, rep_results = fm.evaluate()
        number_of_reps = len(rep_results)
        print('Number of Reps:', number_of_reps)
    except Exception as e:
        print("Error evaluating force metrics:", e)
        test_results, rep_results, number_of_reps = {}, [], 0

    # 8. Compute NIRS metrics if available - store as JSON now
    nirs_results = {}
    if data_type == "force_nirs":
        try:
            start_time, test_start_abs, test_end_abs = find_test_interval(force_file)
            # Convert test boundaries to relative time:
            test_start_rel = test_start_abs - start_time
            test_end_rel = test_end_abs - start_time
            nirs_eval = NIRSEvaluation(nirs_file, smoothing_window=25, baseline_threshold=0.1,
                                      recovery_tolerance=1.0)
            nirs_results = nirs_eval.evaluate(start_time, test_start_rel, test_end_rel)
            print("NIRS Evaluation Results:", nirs_results)
        except Exception as e:
            print("Error evaluating nirs metrics:", e)
            nirs_results = {}

    # 9. Prepare the test data for insertion.
    test_data = {
        "arm_tested": tested_arm,
        "data_type": data_type,
        "test_type": test_type,
        "timestamp": start_time_str,
        "number_of_reps": number_of_reps,
        "force_file": force_file,
        "nirs_file": nirs_file,
        "test_results": test_results,  # This is now a JSON-serializable dictionary
        "nirs_results": nirs_results,  # This is now a JSON-serializable dictionary
        "rep_results": rep_results     # This is now a JSON-serializable list
    }
    print("Test DB:", test_data)

    # 10. Insert the test result into the tests database.
    test_manager = ClimbingTestManager()  # It determines its own DB path.
    # Make sure to pass participant_id as an integer, not a string
    test_manager.add_test_result(admin_id=1, participant_id=participant_id, db_data=test_data)
    test_manager.close_connection()
    climber_manager.close()
    print("Test data transferred.")
    print("")


if __name__ == "__main__":
    original_db = "tests/data_2.db"  # Your original database file path.
    # A list of test IDs to transfer.
    test_ids = [
        691, 546, 685, 53, 660, 30, 605, 336, 672, 530, 654, 304, 675, 588, 579,
        59, 351, 599, 9, 26, 657, 307, 362, 648, 39, 50, 417, 368, 42, 38
    ]
    for test_id in test_ids:
        transfer_data(original_db, test_id)

    # test_id = 605
    # transfer_data(original_db, test_id)