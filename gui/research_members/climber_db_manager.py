"""
This module provides the `ClimberDatabaseManager` class for handling all climber data operations.

Key functionalities:
- Creating the climbers table if it doesn't exist.
- Registering, updating, and deleting climber data linked to an admin.
- Fetching climber data for viewing and editing.
"""

import sqlite3


class ClimberDatabaseManager:
    """
    Manages the climber database, handling tasks such as adding, updating, and retrieving climber information.
    Each climber is linked to a specific admin.
    """

    def __init__(self, db_name="climber_database.db"):
        """Initializes the ClimberDatabaseManager and creates the climber database if it doesn't exist."""
        self.connection = sqlite3.connect(db_name)
        self.create_climbers_table()

    def create_climbers_table(self):
        """Creates the climbers table in the climber database."""
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS climbers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER NOT NULL,
                    name TEXT,
                    surname TEXT,
                    email TEXT,
                    gender TEXT,
                    dominant_arm TEXT,
                    weight INTEGER,
                    height INTEGER,
                    age INTEGER,
                    french_scale TEXT,
                    climbing_freq INTEGER,
                    climbing_hours INTEGER,
                    years_climbing INTEGER,
                    bouldering INTEGER,
                    lead_climbing INTEGER,
                    climbing_indoor INTEGER,
                    climbing_outdoor INTEGER,
                    sport_other TEXT,
                    sport_freq INTEGER,
                    sport_activity_hours INTEGER
                );
            """)

    def register_climber(self, admin_id, **climber_data):
        """Registers a new climber linked to a specific admin."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO climbers (admin_id, name, surname, email, gender, dominant_arm, weight, height,
                                      age, french_scale, climbing_freq, climbing_hours, years_climbing, 
                                      bouldering, lead_climbing, climbing_indoor, climbing_outdoor, 
                                      sport_other, sport_freq, sport_activity_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                admin_id,
                climber_data.get('name'),
                climber_data.get('surname'),
                climber_data.get('email'),
                climber_data.get('gender'),
                climber_data.get('dominant_arm'),
                climber_data.get('weight'),
                climber_data.get('height'),
                climber_data.get('age'),
                climber_data.get('french_scale'),
                climber_data.get('climbing_freq'),
                climber_data.get('climbing_hours'),
                climber_data.get('years_climbing'),
                climber_data.get('bouldering'),
                climber_data.get('lead_climbing'),
                climber_data.get('climbing_indoor'),
                climber_data.get('climbing_outdoor'),
                climber_data.get('sport_other'),
                climber_data.get('sport_freq'),
                climber_data.get('sport_activity_hours')
            ))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def update_climber_data(self, admin_id, climber_id, updated_data):
        """Updates climber information based on admin_id and climber_id."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE climbers
                SET name = ?, surname = ?, email = ?, gender = ?, dominant_arm = ?, weight = ?, height = ?, age = ?, 
                    french_scale = ?, climbing_freq = ?, climbing_hours = ?, years_climbing = ?, bouldering = ?, 
                    lead_climbing = ?, climbing_indoor = ?, climbing_outdoor = ?, sport_other = ?, 
                    sport_freq = ?, sport_activity_hours = ?
                WHERE id = ? AND admin_id = ?;
            """, (
                updated_data['name'], updated_data['surname'], updated_data['email'], updated_data['gender'],
                updated_data['dominant_arm'], updated_data['weight'], updated_data['height'], updated_data['age'],
                updated_data['french_scale'], updated_data['climbing_freq'], updated_data['climbing_hours'],
                updated_data['years_climbing'], updated_data['bouldering'], updated_data['lead_climbing'],
                updated_data['climbing_indoor'], updated_data['climbing_outdoor'], updated_data['sport_other'],
                updated_data['sport_freq'], updated_data['sport_activity_hours'], climber_id, admin_id
            ))
            self.connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating climber data: {e}")
            return False

    def get_climbers_by_admin(self, admin_id):
        """Fetches all climbers registered by a specific admin."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, name, surname, email FROM climbers WHERE admin_id = ?;", (admin_id,))
        return [{"id": row[0], "name": row[1], "surname": row[2]} for row in cursor.fetchall()]

    def get_all_climbers(self):
        """Fetches all registered climbers."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, name, surname, email FROM climbers;")
        return [{"id": row[0], "name": row[1], "surname": row[2]} for row in cursor.fetchall()]

    def get_user_data(self, admin_id, climber_id):
        """
        Retrieves all field values for a registered climber based on email, accessible only by the admin
        who registered them.

        Args:
            admin_id (int): The ID of the admin currently logged in, used to restrict access.
            climber_id (str): The id of the climber whose data is being requested.

        Returns:
            dict: A dictionary of the climber's field values if accessible by the admin; None if access is denied.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT name, surname, email, gender, dominant_arm, weight, height, age, french_scale, 
                   climbing_freq, climbing_hours, years_climbing, bouldering, lead_climbing, 
                   climbing_indoor, climbing_outdoor, sport_other, sport_freq, sport_activity_hours
            FROM climbers
            WHERE id = ? AND admin_id = ?;
        """, (climber_id, admin_id))
        result = cursor.fetchone()

        if result:
            fields = ["name", "surname", "email", "gender", "dominant_arm", "weight", "height", "age",
                      "french_scale", "climbing_freq", "climbing_hours", "years_climbing", "bouldering",
                      "lead_climbing", "climbing_indoor", "climbing_outdoor", "sport_other",
                      "sport_freq", "sport_activity_hours"]
            return dict(zip(fields, result))
        return None  # If no climber data is found or access is denied

    def delete_climber(self, climber_id, admin_id):
        """Deletes a climber by email if they belong to the specified admin."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM climbers WHERE id = ? AND admin_id = ?", (climber_id, admin_id))
        self.connection.commit()
        return cursor.rowcount > 0

    def close(self):
        """Closes the database connection."""
        self.connection.close()
