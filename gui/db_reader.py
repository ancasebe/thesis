"""
This script demonstrates how to use the `DatabaseManager` class to fetch
specific user data from the SQLite database.

Key functionalities:
- Retrieves a specific field (e.g., password, email, etc.) for a given username.
- Uses `DatabaseManager.get_user_data()` to fetch the user data.
"""

from gui.admins.db_manager_additional import DatabaseManager

db_manager = DatabaseManager()

# General way to fetch any field
username = "terka"  # Replace with the username
field = "height"  # Replace with the field you want to retrieve
field_value = db_manager.get_user_data(username, field)

if field_value:
    print(f"The {field} for {username} is: {field_value}")
else:
    print(f"User {username} not found.")
