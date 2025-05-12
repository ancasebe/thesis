"""
This module provides the `ClimberDatabaseManager` class for handling all climber data operations.

Key functionalities:
- Creating the climbers table if it doesn't exist.
- Registering, updating, and deleting climber data linked to an admin.
- Fetching climber data for viewing and editing.
"""
import os
import sqlite3
import json


class ClimberDatabaseManager:
    """
    Manages the climber database, handling tasks such as adding, updating, and retrieving climber information.
    Each climber is linked to a specific admin.
    """

    def __init__(self, db_name="climber_database.db"):
        """Initializes the ClimberDatabaseManager and creates the climber database if it doesn't exist."""
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

    def create_climbers_table(self):
        """Creates the climbers table in the climber database."""
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

    def register_climber(self, admin_id, **climber_data):
        """
        Registers a new climber linked to a specific admin.
        If a climber with the same email already exists, returns their ID instead.
        
        Returns:
            int: The ID of the newly inserted or existing climber
        """
        try:
            cursor = self.connection.cursor()
            # Organize data into JSON objects
            basic_info = {
                'name': climber_data.get('name'),
                'surname': climber_data.get('surname'),
                'email': climber_data.get('email'),
                'gender': climber_data.get('gender'),
                'dominant_arm': climber_data.get('dominant_arm'),
                'weight': climber_data.get('weight'),
                'height': climber_data.get('height'),
                'age': climber_data.get('age')
            }

            climbing_info = {
                'ircra': climber_data.get('ircra'),
                'climbing_freq': climber_data.get('climbing_freq'),
                'climbing_hours': climber_data.get('climbing_hours'),
                'years_climbing': climber_data.get('years_climbing'),
                'bouldering': climber_data.get('bouldering'),
                'lead_climbing': climber_data.get('lead_climbing'),
                'climbing_indoor': climber_data.get('climbing_indoor'),
                'climbing_outdoor': climber_data.get('climbing_outdoor')
            }

            sport_info = {
                'sport_other': climber_data.get('sport_other'),
                'sport_freq': climber_data.get('sport_freq'),
                'sport_activity_hours': climber_data.get('sport_activity_hours')
            }

            cursor.execute("""
                INSERT INTO climbers (admin_id, basic_info, climbing_info, sport_info, timestamp)
                VALUES (?, ?, ?, ?, ?);
            """, (
                admin_id,
                json.dumps(basic_info),
                json.dumps(climbing_info),
                json.dumps(sport_info),
                climber_data.get('timestamp')
            ))
            self.connection.commit()
            new_id = cursor.lastrowid
            print(f"New climber registered with ID: {new_id}")
            return new_id
        except sqlite3.IntegrityError as e:
            print(f"Error registering climber: {e}")
            return None

    def update_climber_data(self, admin_id, climber_id, updated_data):
        """Updates climber information based on admin_id and climber_id."""
        try:
            if admin_id is str:
                admin_id = int(admin_id)
            if climber_id is str:
                climber_id = int(climber_id)

                # Organize data into JSON objects
            basic_info = {
                'name': updated_data.get('name'),
                'surname': updated_data.get('surname'),
                'email': updated_data.get('email'),
                'gender': updated_data.get('gender'),
                'dominant_arm': updated_data.get('dominant_arm'),
                'weight': updated_data.get('weight'),
                'height': updated_data.get('height'),
                'age': updated_data.get('age')
            }

            climbing_info = {
                'ircra': updated_data.get('ircra'),
                'climbing_freq': updated_data.get('climbing_freq'),
                'climbing_hours': updated_data.get('climbing_hours'),
                'years_climbing': updated_data.get('years_climbing'),
                'bouldering': updated_data.get('bouldering'),
                'lead_climbing': updated_data.get('lead_climbing'),
                'climbing_indoor': updated_data.get('climbing_indoor'),
                'climbing_outdoor': updated_data.get('climbing_outdoor')
            }

            sport_info = {
                'sport_other': updated_data.get('sport_other'),
                'sport_freq': updated_data.get('sport_freq'),
                'sport_activity_hours': updated_data.get('sport_activity_hours')
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
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating climber data: {e}")
            return False

    def get_climbers_by_admin(self, admin_id):
        """Fetches all climbers registered by a specific admin."""
        if admin_id is str:
            admin_id = int(admin_id)

        cursor = self.connection.cursor()
        if admin_id == 1:
            cursor.execute(
                "SELECT id, json_extract(basic_info, '$.name') as name, json_extract(basic_info, '$.surname') as surname, json_extract(basic_info, '$.email') as email FROM climbers;")
        else:
            cursor.execute(
                "SELECT id, json_extract(basic_info, '$.name') as name, json_extract(basic_info, '$.surname') as surname, json_extract(basic_info, '$.email') as email FROM climbers WHERE admin_id = ?;",
                (admin_id,))
        return [{"id": row[0], "name": row[1], "surname": row[2]} for row in cursor.fetchall()]

    def get_user_data(self, admin_id, climber_id):
        """
        Retrieves all field values for a registered climber, compatible with both column-based
        and JSON-based schemas.
        
        Args:
            admin_id (int): The ID of the admin currently logged in, used to restrict access.
            climber_id (int): The id of the climber whose data is being requested.
            
        Returns:
            dict: A dictionary of the climber's field values if accessible by the admin; None if access is denied.
        """
        # Convert parameters to integers to ensure correct comparison
        if isinstance(admin_id, str):
            admin_id = int(admin_id)
        if isinstance(climber_id, str):
            climber_id = int(climber_id)

        cursor = self.connection.cursor()
        # print(f"Querying climber ID: {climber_id} for admin ID: {admin_id}")
        
        # First try the JSON-based schema
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
                basic_info = json.loads(result[0])
                climbing_info = json.loads(result[1])
                sport_info = json.loads(result[2])
                combined_data = {**basic_info, **climbing_info, **sport_info}
                return combined_data
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, it might be using the column-based schema
                pass
        
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
        # print(f"Query result: {result}")
        
        if result:
            fields = ["name", "surname", "email", "gender", "dominant_arm", "weight", "height", "age",
                      "ircra", "climbing_freq", "climbing_hours", "years_climbing", "bouldering",
                      "lead_climbing", "climbing_indoor", "climbing_outdoor", "sport_other",
                      "sport_freq", "sport_activity_hours"]
            return dict(zip(fields, result))
        
        return None  # If no climber data is found or access is denied

    def delete_climber(self, climber_id):
        """Deletes a climber by email if they belong to the specified admin."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM climbers WHERE id = ?", (climber_id,))
        self.connection.commit()
        return cursor.rowcount > 0

    def close(self):
        """Closes the database connection."""
        self.connection.close()