"""
This module provides the `LoginDatabaseManager` class for handling admin and
superuser login and registration.

Key functionalities:
- Creating the admins table if it doesn't exist.
- Registering a new superuser or admin with hashed password.
- Verifying login credentials for admins and superuser.
"""

import sqlite3
import hashlib

class LoginDatabaseManager:
    """
    Manages the login database for superuser and admins, handling initialization,
    superuser registration, and admin registration.
    """

    def __init__(self, db_name="login_database.db"):
        """Initializes the LoginDatabaseManager and creates the login database if it doesn't exist."""
        self.connection = sqlite3.connect(db_name)
        self.create_admins_table()

    def create_admins_table(self):
        """Creates the admins table in the login database."""
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    research_name TEXT
                );
            """)

    def register_superuser(self):
        """Registers a default superuser in the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO admins (username, password, research_name)
                VALUES (?, ?, ?);
            """, ("superuser", "placeholder_password", "Superuser Research"))
            self.connection.commit()

            cursor.execute("SELECT id FROM admins WHERE username = ?", ("superuser",))
            user_id = cursor.fetchone()[0]
            salt = str(user_id)
            hashed_password = self.hash_password("superhero", salt)

            cursor.execute("UPDATE admins SET password = ? WHERE id = ?", (hashed_password, user_id))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error creating superuser: {e}")

    def register_admin(self, username, password, research_name):
        """Registers a new admin with hashed password."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO admins (username, password, research_name)
                VALUES (?, ?, ?);
            """, (username, "placeholder_password", research_name))
            self.connection.commit()

            user_id = cursor.lastrowid
            salt = str(user_id)
            hashed_password = self.hash_password(password, salt)

            cursor.execute("UPDATE admins SET password = ? WHERE id = ?", (hashed_password, user_id))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists

    def is_superuser(self, admin_id):
        """Checks if a given admin_id corresponds to the superuser."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT username FROM admins WHERE id = ?", (admin_id,))
        result = cursor.fetchone()
        return result and result[0] == "superuser"

    def verify_user(self, username, password):
        """Verifies if the username and password match an admin entry in the database."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, password FROM admins WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result:
            user_id, stored_password = result
            salt = str(user_id)
            hashed_input_password = self.hash_password(password, salt)
            return hashed_input_password == stored_password
        return False

    def get_admin_id(self, username):
        """Retrieves the admin_id for a given username."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM admins WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None

    @staticmethod
    def hash_password(password, salt):
        """Hashes a password with SHA-512 and a salt."""
        hash_object = hashlib.sha512((password + salt).encode('utf-8'))
        return hash_object.hexdigest()

    def close(self):
        """Closes the database connection."""
        self.connection.close()
