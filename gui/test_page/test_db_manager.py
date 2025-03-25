# --- Database Manager for Test Results ---
import os
import sqlite3


class ClimbingTestManager:
    """
    Manages the climbing test database, handling adding, updating, and retrieving test results.

    Parameters:
        db_name (str): SQLite database filename.
    """

    def __init__(self, db_name="tests_database.db"):
        script_dir = os.path.dirname(__file__)
        # This is the folder where the current .py file (e.g. test_page.py) resides

        # Move one level up, then into the 'databases' folder:
        db_folder = os.path.join(script_dir, '..', 'databases')

        # Construct the full path to a specific DB file:
        climbing_test_db_path = os.path.join(db_folder, db_name)
        self.connection = sqlite3.connect(climbing_test_db_path)
        self.create_climbing_tests_table()

    def create_climbing_tests_table(self):
        """Creates the climbing_tests table in the database if it does not exist."""
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS climbing_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    arm_tested TEXT,
                    data_type TEXT,
                    test_type TEXT,
                    timestamp TEXT,
                    file_paths TEXT,
                    test_results TEXT
                )
            ''')

    def add_test_result(self, admin_id, participant_id, db_data):
        """
        Adds a new test result to the database.

        Parameters:
            admin_id (str): Administrator ID.
            participant_id (str): Participant ID.
            arm_tested (str): Actually tested arm.
            timestamp (str): Date of the test.
            file_paths (str): CSV filenames for Force and/or NIRS data.
            test_results (str): Individual test results.
        """
        with self.connection:
            self.connection.execute('''
                INSERT INTO climbing_tests (admin_id, participant_id, arm_tested, data_type, test_type, timestamp, file_paths, test_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (admin_id, participant_id, db_data['arm_tested'], db_data['data_type'], db_data['test_type'],
                  db_data['timestamp'], db_data['file_paths'], db_data['test_results']))

            print(admin_id, participant_id, db_data['arm_tested'], db_data['data_type'], db_data['test_type'],
                  db_data['timestamp'], db_data['file_paths'], db_data['test_results'])
            print("Test result saved successfully.")

    def fetch_all_results(self):
        """
        Fetches all test results from the database.

        Returns:
            list: Rows from the climbing_tests table.
        """
        with self.connection:
            cursor = self.connection.execute('SELECT * FROM climbing_tests')
            return cursor.fetchall()

    def fetch_results_by_participant(self, admin_id, participant_id):
        """
        Fetches test results for a specific participant.

        Parameters:
            admin_id (str): Admin ID.
            participant_id (str): Participant ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            cursor = self.connection.execute('SELECT * FROM climbing_tests WHERE admin_id = ? AND participant_id = ?',
                                             (admin_id, participant_id,))
            return cursor.fetchall()

    def fetch_results_by_admin(self, admin_id):
        """
        Fetches test results for a specific participant.

        Parameters:
            admin_id (str): Admin ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            cursor = self.connection.execute('SELECT * FROM climbing_tests WHERE admin_id = ?', (admin_id,))
            return cursor.fetchall()

    def close_connection(self):
        """Closes the database connection."""
        self.connection.close()
