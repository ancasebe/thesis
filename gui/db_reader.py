from db_manager import DatabaseManager

db_manager = DatabaseManager()

# General way to fetch any field
username = "sebestin"  # Replace with the username
field = "password"  # Replace with the field you want to retrieve
field_value = db_manager.get_user_data(username, field)

if field_value:
    print(f"The {field} for {username} is: {field_value}")
else:
    print(f"User {username} not found.")
