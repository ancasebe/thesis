import sqlite3
import bcrypt


class DatabaseManager:
    def __init__(self, db_name="app_users.db"):
        self.connection = sqlite3.connect(db_name)
        self.create_users_table()

    def create_users_table(self):
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
                    years_climbing INTEGER,
                    climbing_indoor INTEGER,
                    lead_climbing INTEGER,
                    bouldering INTEGER,
                    climbing_freq INTEGER,
                    climbing_hours INTEGER,
                    sport_freq INTEGER,
                    sport_activity_hours INTEGER
                );
            """)

    def register_user(self, username, password, **user_data):
        try:
            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert user data into the database
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
                    user_data.get('climbing_indoor'),
                    user_data.get('lead_climbing'),
                    user_data.get('bouldering'),
                    user_data.get('climbing_freq'),
                    user_data.get('climbing_hours'),
                    user_data.get('sport_freq'),
                    user_data.get('sport_activity_hours')
                ))
            return True
        except sqlite3.IntegrityError:
            # Username already exists
            return False

    def verify_user(self, username, password):
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT password FROM users WHERE username = ?;
        """, (username,))
        result = cursor.fetchone()
        # actual_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        actual_password = password.encode('utf-8')
        if result:
            stored_password = result[0]
            return bcrypt.checkpw(actual_password, stored_password)
        else:
            return False

    def get_user_data(self, username, field):
        cursor = self.connection.cursor()
        query = f"SELECT {field} FROM users WHERE username = ?;"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        if result:
            return result[0]  # Returning the requested field's value
        else:
            return None  # If no user is found or field is invalid

    def close(self):
        self.connection.close()
