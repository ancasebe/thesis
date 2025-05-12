"""
Climber database management module for the Climbing Testing Application.

This module provides the ClimberDatabaseManager class for handling all climber data operations.

Key functionalities:
- Creating the climbers table if it doesn't exist
- Registering, updating, and deleting climber data linked to an admin
- Fetching climber data for viewing and editing
- Handling error conditions with appropriate user feedback
- Managing database connections and resources

The climber database manager ensures proper data persistence and retrieval for all
participant information, and maintains the relationship between climbers and their
administering researchers.
"""
import os
import sqlite3
import json
import logging
from PySide6.QtWidgets import QMessageBox


class ClimberDatabaseManager:
    """
    Manages the climber database, handling tasks such as adding, updating, and retrieving climber information.
    Each climber is linked to a specific admin.
    """

    def __init__(self, db_name="climber_database.db", parent=None):
        """
        Initializes the ClimberDatabaseManager and creates the climber database if it doesn't exist.

        Args:
            db_name (str): The name of the database file
            parent (QWidget): Parent widget for displaying error dialogs
        """
        self.parent = parent
        self.connection = None
        
        try:
            script_dir = os.path.dirname(__file__)
            # This is the folder where the current .py file (e.g. test_page.py) resides

            # Move one level up, then into the 'databases' folder:
            db_folder = os.path.join(script_dir, '..', 'databases')

            # Ensure that the target directory exists; if not, create it.
            if not os.path.exists(db_folder):
                os.makedirs(db_folder)

            # Construct the full path to a specific DB file:
            climber_db_path = os.path.join(db_folder, db_name)
            self.connection = sqlite3.connect(climber_db_path)
            # Enable JSON support
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.create_climbers_table()
            logging.info(f"Successfully initialized climber database: {climber_db_path}")
        except sqlite3.Error as e:
            error_msg = f"SQLite error initializing ClimberDatabaseManager: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            raise
        except Exception as e:
            error_msg = f"Error initializing ClimberDatabaseManager: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            raise

    def _show_error(self, title, message):
        """
        Display an error message dialog if parent widget is available.
        
        Args:
            title (str): Dialog title
            message (str): Error message
        """
        if self.parent:
            QMessageBox.critical(self.parent, title, message)

    def create_climbers_table(self):
        """Creates the climbers table in the climber database."""
        try:
            with self.connection:
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS climbers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        admin_id INTEGER NOT NULL,
                        basic_info JSON NOT NULL,
                        climbing_info JSON NOT NULL,
                        sport_info JSON,
                        timestamp TEXT
                    );
                """)
            logging.info("Climbers table created or confirmed to exist")
        except sqlite3.Error as e:
            error_msg = f"SQLite error creating climbers table: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            raise
        except Exception as e:
            error_msg = f"Error creating climbers table: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            raise

    def register_climber(self, admin_id, **climber_data):
        """
        Registers a new climber linked to a specific admin.
        If a climber with the same email already exists, returns their ID instead.
        
        Args:
            admin_id (int): The ID of the admin registering the climber
            **climber_data: Dictionary of climber attributes
            
        Returns:
            int: The ID of the newly inserted or existing climber
            None: If there was an error during registration
        """
        if not self.connection:
            logging.error("No database connection available")
            self._show_error("Database Error", "No database connection available")
            return None
            
        try:
            cursor = self.connection.cursor()
            
            # Convert admin_id to integer if it's a string
            if isinstance(admin_id, str):
                admin_id = int(admin_id)
                
            # Organize data into JSON objects
            basic_info = {
                'name': climber_data.get('name', ''),
                'surname': climber_data.get('surname', ''),
                'email': climber_data.get('email', ''),
                'gender': climber_data.get('gender', ''),
                'dominant_arm': climber_data.get('dominant_arm', ''),
                'weight': climber_data.get('weight', ''),
                'height': climber_data.get('height', ''),
                'age': climber_data.get('age', '')
            }

            climbing_info = {
                'ircra': climber_data.get('ircra', ''),
                'climbing_freq': climber_data.get('climbing_freq', ''),
                'climbing_hours': climber_data.get('climbing_hours', ''),
                'years_climbing': climber_data.get('years_climbing', ''),
                'bouldering': climber_data.get('bouldering', ''),
                'lead_climbing': climber_data.get('lead_climbing', ''),
                'climbing_indoor': climber_data.get('climbing_indoor', ''),
                'climbing_outdoor': climber_data.get('climbing_outdoor', '')
            }

            sport_info = {
                'sport_other': climber_data.get('sport_other', ''),
                'sport_freq': climber_data.get('sport_freq', ''),
                'sport_activity_hours': climber_data.get('sport_activity_hours', '')
            }

            cursor.execute("""
                INSERT INTO climbers (admin_id, basic_info, climbing_info, sport_info, timestamp)
                VALUES (?, ?, ?, ?, ?);
            """, (
                admin_id,
                json.dumps(basic_info),
                json.dumps(climbing_info),
                json.dumps(sport_info),
                climber_data.get('timestamp', '')
            ))
            self.connection.commit()
            new_id = cursor.lastrowid
            logging.info(f"New climber registered with ID: {new_id}")
            return new_id
        except ValueError as e:
            error_msg = f"Invalid value when registering climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Invalid Data", error_msg)
            return None
        except sqlite3.IntegrityError as e:
            error_msg = f"Database integrity error registering climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return None
        except sqlite3.Error as e:
            error_msg = f"Database error registering climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return None
        except Exception as e:
            error_msg = f"Error registering climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            return None

    def update_climber_data(self, admin_id, climber_id, updated_data):
        """
        Updates climber information based on admin_id and climber_id.
        
        Args:
            admin_id (int): The ID of the admin updating the climber
            climber_id (int): The ID of the climber to update
            updated_data (dict): Dictionary of updated climber attributes
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.connection:
            logging.error("No database connection available")
            self._show_error("Database Error", "No database connection available")
            return False
            
        try:
            # Convert IDs to integers if they're strings
            if isinstance(admin_id, str):
                admin_id = int(admin_id)
            if isinstance(climber_id, str):
                climber_id = int(climber_id)

            # Organize data into JSON objects
            basic_info = {
                'name': updated_data.get('name', ''),
                'surname': updated_data.get('surname', ''),
                'email': updated_data.get('email', ''),
                'gender': updated_data.get('gender', ''),
                'dominant_arm': updated_data.get('dominant_arm', ''),
                'weight': updated_data.get('weight', ''),
                'height': updated_data.get('height', ''),
                'age': updated_data.get('age', '')
            }

            climbing_info = {
                'ircra': updated_data.get('ircra', ''),
                'climbing_freq': updated_data.get('climbing_freq', ''),
                'climbing_hours': updated_data.get('climbing_hours', ''),
                'years_climbing': updated_data.get('years_climbing', ''),
                'bouldering': updated_data.get('bouldering', ''),
                'lead_climbing': updated_data.get('lead_climbing', ''),
                'climbing_indoor': updated_data.get('climbing_indoor', ''),
                'climbing_outdoor': updated_data.get('climbing_outdoor', '')
            }

            sport_info = {
                'sport_other': updated_data.get('sport_other', ''),
                'sport_freq': updated_data.get('sport_freq', ''),
                'sport_activity_hours': updated_data.get('sport_activity_hours', '')
            }

            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE climbers
                SET basic_info = ?, climbing_info = ?, sport_info = ?
                WHERE id = ? AND admin_id = ?;
            """, (
                json.dumps(basic_info),
                json.dumps(climbing_info),
                json.dumps(sport_info),
                climber_id, admin_id
            ))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                logging.info(f"Successfully updated climber ID {climber_id}")
                return True
            else:
                warning_msg = f"No rows updated for climber ID {climber_id}"
                logging.warning(warning_msg)
                return False
        except ValueError as e:
            error_msg = f"Invalid value when updating climber data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Invalid Data", error_msg)
            return False
        except sqlite3.Error as e:
            error_msg = f"Database error updating climber data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return False
        except Exception as e:
            error_msg = f"Error updating climber data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            return False

    def get_climbers_by_admin(self, admin_id):
        """
        Fetches all climbers registered by a specific admin.
        
        Args:
            admin_id (int): The ID of the admin whose climbers to fetch
            
        Returns:
            list: A list of dictionaries with climber information
            empty list: If there was an error or no climbers found
        """
        if not self.connection:
            logging.error("No database connection available")
            self._show_error("Database Error", "No database connection available")
            return []
            
        try:
            # Convert admin_id to integer if it's a string
            if isinstance(admin_id, str):
                admin_id = int(admin_id)

            cursor = self.connection.cursor()
            
            # Admin ID 1 is a superuser who can see all climbers
            if admin_id == 1:
                cursor.execute("""
                    SELECT id, 
                           json_extract(basic_info, '$.name') as name, 
                           json_extract(basic_info, '$.surname') as surname, 
                           json_extract(basic_info, '$.email') as email 
                    FROM climbers;
                """)
            else:
                cursor.execute("""
                    SELECT id, 
                           json_extract(basic_info, '$.name') as name, 
                           json_extract(basic_info, '$.surname') as surname, 
                           json_extract(basic_info, '$.email') as email 
                    FROM climbers 
                    WHERE admin_id = ?;
                """, (admin_id,))
            
            results = cursor.fetchall()
            
            # Ensure we don't have None values for name/surname
            climbers = []
            for row in results:
                climbers.append({
                    "id": row[0],
                    "name": row[1] if row[1] is not None else "Unknown",
                    "surname": row[2] if row[2] is not None else "Unknown"
                })
                
            if not climbers:
                logging.info(f"No climbers found for admin ID {admin_id}")
                
            return climbers
            
        except ValueError as e:
            error_msg = f"Invalid value when fetching climbers: {str(e)}"
            logging.error(error_msg)
            self._show_error("Invalid Data", error_msg)
            return []
        except sqlite3.Error as e:
            error_msg = f"Database error fetching climbers: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return []
        except Exception as e:
            error_msg = f"Error fetching climbers: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            return []

    def get_user_data(self, admin_id, climber_id):
        """
        Retrieves all field values for a registered climber, compatible with both column-based
        and JSON-based schemas.
        
        Args:
            admin_id (int): The ID of the admin currently logged in, used to restrict access.
            climber_id (int): The id of the climber whose data is being requested.
            
        Returns:
            dict: A dictionary of the climber's field values if accessible by the admin
            None: If access is denied or there was an error
        """
        if not self.connection:
            logging.error("No database connection available")
            self._show_error("Database Error", "No database connection available")
            return None
            
        try:
            # Convert parameters to integers to ensure correct comparison
            if isinstance(admin_id, str):
                admin_id = int(admin_id)
            if isinstance(climber_id, str):
                climber_id = int(climber_id)

            cursor = self.connection.cursor()
            
            # First try the JSON-based schema
            # Admin ID 1 is a superuser who can see all climbers
            if admin_id == 1:
                cursor.execute("""
                    SELECT basic_info, climbing_info, sport_info
                    FROM climbers
                    WHERE id = ?;
                """, (climber_id,))
            else:
                cursor.execute("""
                    SELECT basic_info, climbing_info, sport_info
                    FROM climbers
                    WHERE id = ? AND admin_id = ?;
                """, (climber_id, admin_id))
            
            result = cursor.fetchone()
            
            if result:
                try:
                    # Try JSON-based approach
                    basic_info = json.loads(result[0] if result[0] else '{}')
                    climbing_info = json.loads(result[1] if result[1] else '{}')
                    sport_info = json.loads(result[2] if result[2] else '{}')
                    combined_data = {**basic_info, **climbing_info, **sport_info}
                    return combined_data
                except (json.JSONDecodeError, TypeError) as e:
                    warning_msg = f"JSON parsing failed when getting user data: {str(e)}"
                    logging.warning(warning_msg)
                    # If JSON parsing fails, it might be using the column-based schema
            
            # Fall back to column-based schema if JSON approach failed
            if admin_id == 1:
                cursor.execute("""
                    SELECT name, surname, email, gender, dominant_arm, weight, height, age, ircra, 
                       climbing_freq, climbing_hours, years_climbing, bouldering, lead_climbing, 
                       climbing_indoor, climbing_outdoor, sport_other, sport_freq, sport_activity_hours
                    FROM climbers
                    WHERE id = ?;
                """, (climber_id,))
            else:
                cursor.execute("""
                    SELECT name, surname, email, gender, dominant_arm, weight, height, age, ircra, 
                       climbing_freq, climbing_hours, years_climbing, bouldering, lead_climbing, 
                       climbing_indoor, climbing_outdoor, sport_other, sport_freq, sport_activity_hours
                    FROM climbers
                    WHERE id = ? AND admin_id = ?;
                """, (climber_id, admin_id))
            
            result = cursor.fetchone()
            
            if result:
                fields = ["name", "surname", "email", "gender", "dominant_arm", "weight", "height", "age",
                          "ircra", "climbing_freq", "climbing_hours", "years_climbing", "bouldering",
                          "lead_climbing", "climbing_indoor", "climbing_outdoor", "sport_other",
                          "sport_freq", "sport_activity_hours"]
                return dict(zip(fields, result))
            
            warning_msg = f"No data found for climber ID {climber_id}"
            logging.warning(warning_msg)
            return None  # If no climber data is found or access is denied
            
        except ValueError as e:
            error_msg = f"Invalid value when fetching user data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Invalid Data", error_msg)
            return None
        except sqlite3.Error as e:
            error_msg = f"Database error fetching user data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return None
        except Exception as e:
            error_msg = f"Error fetching user data: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            return None

    def delete_climber(self, climber_id):
        """
        Deletes a climber by ID.
        
        Args:
            climber_id (int): The ID of the climber to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self.connection:
            logging.error("No database connection available")
            self._show_error("Database Error", "No database connection available")
            return False
            
        try:
            # Convert climber_id to integer if it's a string
            if isinstance(climber_id, str):
                climber_id = int(climber_id)
                
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM climbers WHERE id = ?", (climber_id,))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                logging.info(f"Successfully deleted climber ID {climber_id}")
                return True
            else:
                warning_msg = f"No climber found with ID {climber_id} to delete"
                logging.warning(warning_msg)
                return False
                
        except ValueError as e:
            error_msg = f"Invalid value when deleting climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Invalid Data", error_msg)
            return False
        except sqlite3.Error as e:
            error_msg = f"Database error deleting climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
            return False
        except Exception as e:
            error_msg = f"Error deleting climber: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)
            return False

    def close(self):
        """Closes the database connection."""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
                logging.info("Database connection closed successfully")
                self.connection = None
        except sqlite3.Error as e:
            error_msg = f"Database error closing connection: {str(e)}"
            logging.error(error_msg)
            self._show_error("Database Error", error_msg)
        except Exception as e:
            error_msg = f"Error closing database connection: {str(e)}"
            logging.error(error_msg)
            self._show_error("Error", error_msg)