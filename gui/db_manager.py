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
    Class responsible for handling interactions with the SQLite database.

    Methods:
        create_users_table: Creates the users table in the database.
        register_user: Registers a new user with a hashed password using SHA-512 with salt.
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

    @staticmethod
    def hash_password(password, salt):
        """
        Hashes the password using SHA-512 and salts it with the user ID.

        Args:
            password (str): The plain text password.
            salt (str): The salt (usually the user's ID) to combine with the password.

        Returns:
            str: The resulting salted SHA-512 hash.
        """
        # Combine password and salt, then apply SHA-512
        salted_password = password + salt
        hash_object = hashlib.sha512(salted_password.encode('utf-8'))
        hashed_password = hash_object.hexdigest()
        return hashed_password

    def register_user(self, username, password, **user_data):
        """
        Registers a new user in the database with salted SHA-512 hashed password
        and additional user data.

        Args:
            username (str): The username for the user.
            password (str): The plain text password for the user.
            user_data (dict): Additional user information, such as name, email, etc.

        Returns:
            bool: True if registration is successful, False if the username already exists.
        """
        try:
            # Step 1: Insert the user with a temporary placeholder password
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO users (username, password, name, surname, email, gender, dominant_arm, 
                weight, height, age, french_scale, years_climbing, climbing_freq, climbing_hours, 
                climbing_indoor, climbing_outdoor, lead_climbing, bouldering, sport_other, 
                sport_freq, sport_activity_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                username, 'placeholder_password',  # Temporary password, to be updated after hashing
                user_data.get('name'),
                user_data.get('surname'),
                user_data.get('email'),
                user_data.get('gender'),
                user_data.get('dominant_arm'),
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

            # Commit the insertion
            self.connection.commit()

            # Step 2: Fetch the newly created user ID (for salting)
            user_id = cursor.lastrowid
            salt = str(user_id)

            # Step 3: Hash the password with the user ID as the salt
            hashed_password = self.hash_password(password, salt)

            # Step 4: Update the user's password with the hashed value
            cursor.execute("""
                UPDATE users SET password = ? WHERE id = ?;
            """, (hashed_password, user_id))

            # Commit the update
            self.connection.commit()
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
            SELECT id, password FROM users WHERE username = ?;
        """, (username,))
        result = cursor.fetchone()
        if result:
            user_id, stored_password = result
            salt = str(user_id)

            # Hash the input password with the salt and compare to the stored hash
            hashed_input_password = self.hash_password(password, salt)

            return hashed_input_password == stored_password
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
