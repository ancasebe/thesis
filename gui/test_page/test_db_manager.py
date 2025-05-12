"""
Test database management module for the Climbing Testing Application.

This module defines the ClimbingTestManager class which handles all database
operations related to climbing test data. It manages the storage, retrieval,
and organization of test results, including force and NIRS data.

Key functionalities:
- Create and maintain the climbing tests database schema
- Store test results and associated metadata
- Retrieve test data by various criteria (admin, participant, etc.)
- Handle file path management for raw data files
- Process and format test data for retrieval

The test database manager ensures proper data persistence and retrieval for
all climbing tests conducted within the application.
"""
import os
import sqlite3
import json
import logging
from gui.test_page.path_utils import resolve_path


class ClimbingTestManager:
    """
    Manages the climbing test database, handling adding, updating, and retrieving test results.

    Parameters:
        db_name (str): SQLite database filename.
    """

    def __init__(self, db_name="tests_database.db"):
        try:
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
        except Exception as e:
            logging.error(f"Error initializing ClimbingTestManager: {str(e)}")
            raise

    def create_climbing_tests_table(self):
        """Creates the climbing_tests table in the database if it does not exist."""
        try:
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
        except sqlite3.Error as e:
            logging.error(f"Database error creating table: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error creating table: {str(e)}")
            raise

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
        try:
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

            # Convert string IDs to integers if necessary
            if isinstance(admin_id, str):
                admin_id = int(admin_id)
            if isinstance(participant_id, str):
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
                
        except sqlite3.Error as e:
            logging.error(f"Database error adding test result: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Value error adding test result: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error adding test result: {str(e)}")
            raise

    def fetch_results_by_participant(self, participant_id):
        """
        Fetches test results for a specific participant.

        Parameters:
            participant_id (int): Participant ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        try:
            # Convert participant_id to integer if it's a string
            if isinstance(participant_id, str):
                participant_id = int(participant_id)

            with self.connection:
                cursor = self.connection.execute(
                    'SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data FROM climbing_tests WHERE participant_id = ?',
                    (participant_id,)
                )
                results = []
                for row in cursor.fetchall():
                    try:
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
                            try:
                                result['force_file'] = resolve_path(result['force_file'])
                            except Exception as e:
                                logging.warning(f"Error resolving force_file path: {str(e)}")
                                
                        if 'nirs_file' in result:
                            try:
                                result['nirs_file'] = resolve_path(result['nirs_file'])
                            except Exception as e:
                                logging.warning(f"Error resolving nirs_file path: {str(e)}")

                        results.append(result)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error decoding JSON for test ID {row[0]}: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Error processing test ID {row[0]}: {str(e)}")
                
                return results
                
        except sqlite3.Error as e:
            logging.error(f"Database error fetching results: {str(e)}")
            return []
        except ValueError as e:
            logging.error(f"Value error fetching results: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error fetching results: {str(e)}")
            return []

    def fetch_results_by_admin(self, admin_id):
        """
        Fetches test results for a specific admin.

        Parameters:
            admin_id (int): Admin ID.

        Returns:
            list: Matching rows from the climbing_tests table.
        """
        try:
            with self.connection:
                if isinstance(admin_id, str):
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
                    try:
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
                            try:
                                result['force_file'] = resolve_path(result['force_file'])
                            except Exception as e:
                                logging.warning(f"Error resolving force_file path: {str(e)}")
                                
                        if 'nirs_file' in result:
                            try:
                                result['nirs_file'] = resolve_path(result['nirs_file'])
                            except Exception as e:
                                logging.warning(f"Error resolving nirs_file path: {str(e)}")

                        results.append(result)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error decoding JSON for test ID {row[0]}: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Error processing test ID {row[0]}: {str(e)}")

                return results
        
        except sqlite3.Error as e:
            logging.error(f"Database error fetching admin results: {str(e)}")
            return []
        except ValueError as e:
            logging.error(f"Value error fetching admin results: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error fetching admin results: {str(e)}")
            return []

    def get_test_data(self, test_id):
        """
        Retrieves all field values for a test record based on test_id.

        Returns:
            dict: A dictionary of the test's field values if found; None otherwise.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, admin_id, participant_id, timestamp, test_metadata, test_data
                FROM climbing_tests
                WHERE id = ?;
            """, (test_id,))
            row = cursor.fetchone()

            if row:
                try:
                    test_id, admin_id, participant_id, timestamp, test_metadata, test_data = row
                    
                    try:
                        test_metadata_dict = json.loads(test_metadata)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error decoding test_metadata JSON: {str(e)}")
                        test_metadata_dict = {}
                        
                    try:
                        test_data_dict = json.loads(test_data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error decoding test_data JSON: {str(e)}")
                        test_data_dict = {}

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
                    if 'force_file' in result and result['force_file']:
                        try:
                            result['force_file'] = resolve_path(result['force_file'])
                        except Exception as e:
                            logging.warning(f"Error resolving force_file path: {str(e)}")
                            
                    if 'nirs_file' in result and result['nirs_file']:
                        try:
                            result['nirs_file'] = resolve_path(result['nirs_file'])
                        except Exception as e:
                            logging.warning(f"Error resolving nirs_file path: {str(e)}")

                    return result
                    
                except Exception as e:
                    logging.error(f"Error processing test data: {str(e)}")
                    return None
                    
            return None
            
        except sqlite3.Error as e:
            logging.error(f"Database error getting test data: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting test data: {str(e)}")
            return None

    def close_connection(self):
        """Closes the database connection."""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
        except sqlite3.Error as e:
            logging.error(f"Error closing database connection: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error closing connection: {str(e)}")