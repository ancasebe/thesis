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
                    force_file TEXT,
                    nirs_file TEXT,
                    test_results TEXT,
                    rep_results TEXT
                )
            ''')

    def add_test_result(self, admin_id, participant_id, db_data):
        """
        Adds a new test result to the database.

        Parameters:
            admin_id (str): Administrator ID.
            participant_id (str): Participant ID.
            db_data (dict): Actually tested arm.
        """
        with self.connection:
            self.connection.execute('''
                INSERT INTO climbing_tests (admin_id, participant_id, arm_tested, data_type, test_type, 
                timestamp, force_file, nirs_file, test_results, rep_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (admin_id, participant_id, db_data['arm_tested'], db_data['data_type'],
                  db_data['test_type'], db_data['timestamp'], db_data['force_file'], db_data['nirs_file'],
                  db_data['test_results'], db_data['rep_results']))

            print("Test result saved successfully.")
            # seems to be working alright

    def fetch_results_by_participant(self, participant_id):
        """
        Fetches test results for a specific participant.

        Parameters:
            participant_id (str): Participant ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            cursor = self.connection.execute(
                'SELECT * FROM climbing_tests WHERE participant_id = ?',
                (participant_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def fetch_results_by_admin(self, admin_id):
        """
        Fetches test results for a specific admin.

        Parameters:
            admin_id (str): Admin ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            # Ensure admin_id is compared as an integer
            if admin_id == "1":
                cursor = self.connection.execute('SELECT * FROM climbing_tests')
            else:
                cursor = self.connection.execute(
                    'SELECT * FROM climbing_tests WHERE admin_id = ?',
                    (admin_id,)
                )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def fetch_result_by_test_id(self, test_id):
        """
        Fetches test results for a specific test id.

        Parameters:
            test_id (str): Test ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            cursor = self.connection.execute('SELECT * FROM climbing_tests WHERE id = ?', (test_id,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_test_data(self, test_id):
        """
        Retrieves all field values for a test record based on test_id.

        Returns:
            dict: A dictionary of the test's field values if found; None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, admin_id, participant_id, arm_tested, data_type, test_type, timestamp, 
            force_file, nirs_file, test_results, rep_results
            FROM climbing_tests
            WHERE id = ?;
        """, (test_id,))
        result = cursor.fetchone()
        if result:
            fields = ["id", "admin_id", "participant_id", "arm_tested", "data_type", "test_type", "timestamp",
                      "force_file", "nirs_file", "test_results", "rep_results"]
            return dict(zip(fields, result))
        return None

    def close_connection(self):
        """Closes the database connection."""
        self.connection.close()
