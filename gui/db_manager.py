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
import bcrypt


class DatabaseManager:
    """
    Class responsible for handling interactions with the SQLite database.

    Methods:
        create_users_table: Creates the users table in the database.
        register_user: Registers a new user with a hashed password.
        verify_user: Verifies if the username and password match.
        close: Closes the database connection.
    """

    def __init__(self, db_name="app_users.db"):
        """Initializes the DatabaseManager and creates the users table if it doesn't exist."""
        self.connection = sqlite3.connect(db_name)
        self.create_users_table()

    def create_users_table(self):
        """Creates a users table in the database if it doesn't already exist."""
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    name TEXT,
                    surname TEXT,
                    email TEXT,
                    gender TEXT,
                    dominant_arm TEXT,
                    arm_tested TEXT,
                    weight INTEGER,
                    height INTEGER,
                    age INTEGER,
                    french_scale TEXT,
                    climbing_freq INTEGER,
                    climbing_hours INTEGER,
                    years_climbing INTEGER,
                    climbing_indoor INTEGER,
                    climbing_outdoor INTEGER,
                    bouldering INTEGER,
                    lead_climbing INTEGER,
                    sport_other TEXT,
                    sport_freq INTEGER,
                    sport_activity_hours INTEGER
                );
            """)

    def register_user(self, username, password, **user_data):
        """
        Registers a new user in the database with hashed password and additional user data.

        Args:
            username (str): The username for the user.
            password (str): The plain text password for the user.
            user_data (dict): Additional user information, such as name, email, etc.

        Returns:
            bool: True if registration is successful, False if the username already exists.
        """
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            with self.connection:
                self.connection.execute("""
                    INSERT INTO users (username, password, name, surname, email, gender, dominant_arm, 
                    arm_tested, weight, height, age, french_scale, years_climbing, climbing_indoor, 
                    lead_climbing, bouldering, climbing_freq, climbing_hours, sport_freq, sport_activity_hours) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    username, hashed_password,
                    user_data.get('name'),
                    user_data.get('surname'),
                    user_data.get('email'),
                    user_data.get('gender'),
                    user_data.get('dominant_arm'),
                    user_data.get('arm_tested'),
                    user_data.get('weight'),
                    user_data.get('height'),
                    user_data.get('age'),
                    user_data.get('french_scale'),
                    user_data.get('years_climbing'),
                    user_data.get('climbing_freq'),
                    user_data.get('climbing_hours'),
                    user_data.get('climbing_indoor'),
                    user_data.get('climbing_outdoor'),
                    user_data.get('bouldering'),
                    user_data.get('lead_climbing'),
                    user_data.get('sport_other'),
                    user_data.get('sport_freq'),
                    user_data.get('sport_activity_hours')
                ))
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username, password):
        """
        Verifies if the username and password match an entry in the database.

        Args:
            username (str): The username to check.
            password (str): The plain text password to check.

        Returns:
            bool: True if the password matches the stored hashed password, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT password FROM users WHERE username = ?;
        """, (username,))
        result = cursor.fetchone()
        if result:
            stored_password = result[0]
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), stored_password)
        return False

    def get_user_data(self, username, field):
        """
        Retrieves a specific field value for a given user from the database.

        Args:
            username (str): The username of the user whose data is being requested.
            field (str): The specific field (column) to retrieve from the users table.

        Returns:
            The value of the requested field if the user exists and the field is valid.
            Returns `None` if the user does not exist or if the field is invalid.

        Raises:
            sqlite3.Error: If there is a problem executing the SQL query.
        """
        cursor = self.connection.cursor()
        query = f"SELECT {field} FROM users WHERE username = ?;"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        if result:
            return result[0]  # Returning the requested field's value

        return None  # If no user is found or field is invalid

    def close(self):
        """Closes the connection to the database."""
        self.connection.close()
