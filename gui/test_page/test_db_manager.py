"""
This module provides a database manager for climbing test results,
using JSON for efficient storage of complex data structures.
"""
import os
import sqlite3
import json
from gui.test_page.path_utils import resolve_path


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

        # Ensure that the target directory exists; if not, create it.
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)

        # Construct the full path to a specific DB file:
        climbing_test_db_path = os.path.join(db_folder, db_name)
        self.connection = sqlite3.connect(climbing_test_db_path)
        # Enable JSON support
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.create_climbing_tests_table()

    def create_climbing_tests_table(self):
        """Creates the climbing_tests table in the database if it does not exist."""
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS climbing_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER NOT NULL,
                    participant_id INTEGER NOT NULL,
                    timestamp TEXT,
                    test_metadata JSON NOT NULL,
                    test_data JSON NOT NULL
                )
            ''')

    def add_test_result(self, admin_id, participant_id, db_data):
        """
        Adds a new test result to the database and returns the test id.

        Parameters:
            admin_id (int): Administrator ID.
            participant_id (int): Participant ID.
            db_data (dict): Dictionary containing test details.

        Returns:
            int: The ID of the newly inserted test result.
        """
        # Organize data into JSON objects
        test_metadata = {
            'arm_tested': db_data.get('arm_tested'),
            'data_type': db_data.get('data_type'),
            'test_type': db_data.get('test_type'),
            'number_of_reps': db_data.get('number_of_reps')
        }

        test_data = {
            'force_file': db_data.get('force_file'),
            'nirs_file': db_data.get('nirs_file'),
            'test_results': db_data.get('test_results'),
            'nirs_results': db_data.get('nirs_results'),
            'rep_results': db_data.get('rep_results')
        }

        if admin_id is str:
            admin_id = int(admin_id)
        if participant_id is str:
            participant_id = int(participant_id)

        with self.connection:
            cursor = self.connection.execute('''
                INSERT INTO climbing_tests (admin_id, participant_id, timestamp, test_metadata, test_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (admin_id, participant_id, db_data.get('timestamp'),
                  json.dumps(test_metadata), json.dumps(test_data)))


            print("Test result saved successfully.")

            new_id = cursor.lastrowid
            return new_id

    def fetch_results_by_participant(self, participant_id):
        """
        Fetches test results for a specific participant.

        Parameters:
            participant_id (int): Participant ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        # Convert participant_id to integer if it's a string

        if participant_id is str:
            participant_id = int(participant_id)

        with self.connection:
            cursor = self.connection.execute(
                'SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data FROM climbing_tests WHERE participant_id = ?',
                (participant_id,)
            )
            results = []
            for row in cursor.fetchall():
                test_id, admin_id, participant_id, timestamp, test_metadata, test_data = row
                test_metadata_dict = json.loads(test_metadata)
                test_data_dict = json.loads(test_data)

                # Combine all data
                result = {
                    'id': test_id,
                    'admin_id': admin_id,
                    'participant_id': participant_id,
                    'timestamp': timestamp,
                    **test_metadata_dict,
                    **test_data_dict
                }

                # Resolve paths for force_file and nirs_file
                if 'force_file' in result:
                    result['force_file'] = resolve_path(result['force_file'])
                if 'nirs_file' in result:
                    result['nirs_file'] = resolve_path(result['nirs_file'])

                results.append(result)
            return results

    def fetch_results_by_admin(self, admin_id):
        """
        Fetches test results for a specific admin.

        Parameters:
            admin_id (int): Admin ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        with self.connection:
            if admin_id is str:
                admin_id = int(admin_id)

                # Ensure admin_id is compared as an integer
            if admin_id == 1:
                cursor = self.connection.execute(
                    'SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data FROM climbing_tests'
                )
            else:
                cursor = self.connection.execute(
                    'SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data FROM climbing_tests WHERE admin_id = ?',
                    (admin_id,)
                )

            results = []
            for row in cursor.fetchall():
                test_id, admin_id, participant_id, timestamp, test_metadata, test_data = row
                test_metadata_dict = json.loads(test_metadata)
                test_data_dict = json.loads(test_data)

                # Combine all data
                result = {
                    'id': test_id,
                    'admin_id': admin_id,
                    'participant_id': participant_id,
                    'timestamp': timestamp,
                    **test_metadata_dict,
                    **test_data_dict
                }

                # Resolve paths for force_file and nirs_file
                if 'force_file' in result:
                    result['force_file'] = resolve_path(result['force_file'])
                if 'nirs_file' in result:
                    result['nirs_file'] = resolve_path(result['nirs_file'])

                results.append(result)

            return results

    def get_test_data(self, test_id):
        """
        Retrieves all field values for a test record based on test_id.

        Returns:
            dict: A dictionary of the test's field values if found; None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data
            FROM climbing_tests
            WHERE id = ?;
        """, (test_id,))
        result = cursor.fetchone()

        if result:
            test_id, admin_id, participant_id, timestamp, test_metadata, test_data = result
            test_metadata_dict = json.loads(test_metadata)
            test_data_dict = json.loads(test_data)

            # Combine all data
            result =  {
                'id': test_id,
                'admin_id': admin_id,
                'participant_id': participant_id,
                'timestamp': timestamp,
                **test_metadata_dict,
                **test_data_dict
            }

            # Resolve paths for force_file and nirs_file
            if 'force_file' in result:
                result['force_file'] = resolve_path(result['force_file'])
            if 'nirs_file' in result:
                result['nirs_file'] = resolve_path(result['nirs_file'])

            return result

        return None

    def close_connection(self):
        """Closes the database connection."""
        self.connection.close()