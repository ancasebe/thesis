"""
This module provides the `DatabaseManager` class for handling all database operations,
such as user registration, login verification, and managing user data.

Key functionalities:
- Creating the users table if it doesn't exist.
- Registering a new user with hashed password and additional profile data.
- Verifying user login credentials.
- Closing the database connection.
"""
import sqlite3
import hashlib


class DatabaseManager:
    """
    Manages the main database for superuser, admins, and climbers.
    Handles tasks such as initialization, superuser registration, and new admin registration.
    """

    def __init__(self, db_name="app_database.db"):
        """Initializes the SuperDatabaseManager and creates the main database if it doesn't exist."""
        self.db_name = db_name
        self.connection = sqlite3.connect(db_name)
        self.create_database()

    def create_database(self):
        """Creates the main database with required tables for superuser, admins, and climbers."""
        with self.connection:
            # Create superuser and admins table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    research_name TEXT
                );
            """)

            # Create climbers table linked to admins by admin_id (foreign key)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS climbers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
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
                    sport_activity_hours INTEGER,
                    FOREIGN KEY (admin_id) REFERENCES admins(id)
                );
            """)

    def register_superuser(self):
        """Registers a default superuser in the database."""
        try:
            cursor = self.connection.cursor()
            # Insert the superuser with a temporary placeholder password
            cursor.execute("""
                INSERT OR IGNORE INTO admins (username, password, research_name)
                VALUES (?, ?, ?);
            """, ("superuser", "placeholder_password", "Superuser Research"))
            self.connection.commit()

            # Fetch the user_id to use as salt for hashing the password
            cursor.execute("SELECT id FROM admins WHERE username = ?", ("superuser",))
            user_id = cursor.fetchone()[0]
            salt = str(user_id)
            hashed_password = self.hash_password("superhero", salt)

            # Update the superuser's password with the hashed value
            cursor.execute("UPDATE admins SET password = ? WHERE id = ?", (hashed_password, user_id))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error creating superuser: {e}")

    def is_superuser(self, admin_id):
        """
        Checks if the given admin_id corresponds to the superuser.

        Args:
            admin_id (int): The ID of the admin.

        Returns:
            bool: True if the admin is the superuser, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT username FROM admins WHERE id = ?", (admin_id,))
        result = cursor.fetchone()
        return result and result[0] == "superuser"

    def register_admin(self, username, password, **admin_data):
        """
        Registers a new admin in the database with a hashed password.

        Args:
            username (str): The username for the admin.
            password (str): The plain text password for the admin.
            admin_data (dict): Additional user information, such as research name...

        Returns:
            bool: True if registration is successful, False if the username already exists.
        """
        try:
            cursor = self.connection.cursor()

            # Define the additional data fields, here with research_name as example
            research_name = admin_data.get('research_name')

            # Insert the admin with a temporary placeholder password
            cursor.execute("""
                INSERT INTO admins (username, password, research_name)
                VALUES (?, ?, ?);
            """, (username, "placeholder_password", research_name))
            self.connection.commit()

            # Fetch the user_id to use as salt for hashing the password
            user_id = cursor.lastrowid
            salt = str(user_id)
            hashed_password = self.hash_password(password, salt)

            # Update the admin's password with the hashed value
            cursor.execute("UPDATE admins SET password = ? WHERE id = ?", (hashed_password, user_id))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists

    def register_climber(self, admin_id, **climber_data):
        """
        Registers a new climber in the database under a specific admin.

        Args:
            admin_id (int): ID of the admin responsible for the climber.
            climber_data (dict): Climber's details.

        Returns:
            bool: True if registration is successful, False otherwise.
        """
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

    def update_climber_data(self, admin_id, email, updated_data):
        """
        Updates the climber's information in the database.

        Args:
            admin_id (int): The ID of the admin.
            email (str): The email of the climber to update.
            updated_data (dict): Dictionary with updated fields and values.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE climbers
                SET name = ?, surname = ?, gender = ?, dominant_arm = ?, weight = ?, height = ?, age = ?, 
                    french_scale = ?, climbing_freq = ?, climbing_hours = ?, years_climbing = ?, bouldering = ?, 
                    lead_climbing = ?, climbing_indoor = ?, climbing_outdoor = ?, sport_other = ?, 
                    sport_freq = ?, sport_activity_hours = ?
                WHERE email = ? AND admin_id = ?;
            """, (
                updated_data['name'], updated_data['surname'], updated_data['gender'], updated_data['dominant_arm'],
                updated_data['weight'], updated_data['height'], updated_data['age'], updated_data['french_scale'],
                updated_data['climbing_freq'], updated_data['climbing_hours'], updated_data['years_climbing'],
                updated_data['bouldering'], updated_data['lead_climbing'], updated_data['climbing_indoor'],
                updated_data['climbing_outdoor'], updated_data['sport_other'], updated_data['sport_freq'],
                updated_data['sport_activity_hours'], email, admin_id
            ))
            self.connection.commit()
            return cursor.rowcount > 0  # Returns True if a row was updated
        except sqlite3.Error as e:
            print(f"Error updating climber data: {e}")
            return False

    def verify_user(self, username, password):
        """
        Verifies if an admin's username and password match an entry in the database.

        Args:
            username (str): The username to check.
            password (str): The plain text password to check.

        Returns:
            bool: True if the password matches the stored hashed password, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, password FROM admins WHERE username = ?;
        """, (username,))
        result = cursor.fetchone()
        if result:
            user_id, stored_password = result
            salt = str(user_id)
            hashed_input_password = self.hash_password(password, salt)
            return hashed_input_password == stored_password
        return False

    def get_admin_id(self, username):
        """
        Retrieves the admin_id for a given username.

        Args:
            username (str): The username of the logged-in admin.

        Returns:
            int: The admin_id if found, otherwise None.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None

    @staticmethod
    def hash_password(password, salt):
        """Hashes a password with SHA-512 and a salt."""
        hash_object = hashlib.sha512((password + salt).encode('utf-8'))
        return hash_object.hexdigest()

    def get_user_data(self, admin_id, email):
        """
        Retrieves all field values for a registered climber based on email, accessible only by the admin
        who registered them.

        Args:
            admin_id (int): The ID of the admin currently logged in, used to restrict access.
            email (str): The email of the climber whose data is being requested.

        Returns:
            dict: A dictionary of the climber's field values if accessible by the admin; None if access is denied.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT name, surname, email, gender, dominant_arm, weight, height, age, french_scale, 
                   climbing_freq, climbing_hours, years_climbing, bouldering, lead_climbing, 
                   climbing_indoor, climbing_outdoor, sport_other, sport_freq, sport_activity_hours
            FROM climbers
            WHERE email = ? AND admin_id = ?;
        """, (email, admin_id))
        result = cursor.fetchone()

        if result:
            fields = ["name", "surname", "email", "gender", "dominant_arm", "weight", "height", "age",
                      "french_scale", "climbing_freq", "climbing_hours", "years_climbing", "bouldering",
                      "lead_climbing", "climbing_indoor", "climbing_outdoor", "sport_other",
                      "sport_freq", "sport_activity_hours"]
            return dict(zip(fields, result))
        return None  # If no climber data is found or access is denied

    def get_climbers_by_admin(self, admin_id):
        """Fetches all climbers registered by the specified admin."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT name, surname, email FROM climbers WHERE admin_id = ?;
        """, (admin_id,))
        return [{"name": row[0], "surname": row[1], "email": row[2]} for row in cursor.fetchall()]

    def delete_climber(self, email, admin_id):
        """Deletes a climber by email if they belong to the specified admin."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM climbers WHERE email = ? AND admin_id = ?", (email, admin_id))
        self.connection.commit()
        return cursor.rowcount > 0  # Returns True if a row was deleted

    def close(self):
        """Closes the connection to the database."""
        self.connection.close()
