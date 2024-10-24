from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QSpinBox, QLabel, QSlider, QComboBox, QPushButton, \
    QGridLayout, QMessageBox
from PySide6.QtCore import Qt


class SettingsPage(QWidget):
    """
    A class representing the settings page where users can adjust their registration data.

    Methods:
        setup_ui: Sets up the user interface for the settings page.
        populate_data: Fetches and populates the user's current data from the database.
        handle_save: Saves the updated user data into the database.
    """

    def __init__(self, username, db_manager):
        """Initializes the settings page and pre-fills user data."""
        super().__init__()
        self.username = username
        self.db_manager = db_manager
        self.first_column_widgets = {}
        self.second_column_widgets = {}
        self.setup_ui()
        self.populate_data()

    def setup_ui(self):
        """Sets up the layout and widgets for the settings page."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Create a grid layout for organizing form in columns
        grid_layout = QGridLayout()

        # Dictionary of inputs and their labels for the first and second columns (same as RegistrationPage)
        self.first_column_widgets = {
            "Username:": QLabel(self.username),  # Username is fixed, not editable
            "Name:": QLineEdit(),
            "Surname:": QLineEdit(),
            "Email:": QLineEdit(),
            "Gender:": QComboBox(),
            "Dominant Arm:": QComboBox(),
            "Weight (kg):": QSpinBox(),
            "Height (cm):": QSpinBox(),
            "Age (years):": QSpinBox(),
        }

        self.second_column_widgets = {
            "French Scale Level:": QComboBox(),
            "Years of Climbing:": QSpinBox(),
            "Climbing Frequency/week:": QSpinBox(),
            "Climbing Hours/week:": QSpinBox(),
            "Bouldering and Lead Climbing:": QSlider(Qt.Horizontal),
            "Indoor and Outdoor Climbing:": QSlider(Qt.Horizontal),
            "Other sports:": QLineEdit(),
            "Sport Frequency/week:": QSpinBox(),
            "Sport Activity (hours/week):": QSpinBox()
        }

        # Configuring certain widgets (just like in RegistrationPage)
        self.first_column_widgets["Gender:"].addItems(["-", "Male", "Female", "Other"])
        self.first_column_widgets["Dominant Arm:"].addItems(["-", "Left", "Right"])
        self.first_column_widgets["Weight (kg):"].setRange(0, 200)
        self.first_column_widgets["Height (cm):"].setRange(0, 240)
        self.first_column_widgets["Age (years):"].setRange(0, 100)

        self.second_column_widgets["French Scale Level:"].addItems(
            ["-", "5", "6a", "6a+", "6b", "6b+", "6c", "7a", "7a+", "7b", "7b+",
             "7c", "7c+", "8a", "8a+", "8b", "8b+", "8c", "8c+", "9a", "9a+", "9b", "9b+", "9c"]
        )
        self.second_column_widgets["Years of Climbing:"].setRange(0, 50)
        self.second_column_widgets["Climbing Frequency/week:"].setRange(0, 7)
        self.second_column_widgets["Climbing Hours/week:"].setRange(0, 50)
        self.second_column_widgets["Sport Frequency/week:"].setRange(0, 7)
        self.second_column_widgets["Sport Activity (hours/week):"].setRange(0, 50)
        self.second_column_widgets["Bouldering and Lead Climbing:"].setRange(0, 100)
        self.second_column_widgets["Bouldering and Lead Climbing:"].setSingleStep(5)
        self.second_column_widgets["Bouldering and Lead Climbing:"].setValue(50)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setRange(0, 100)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setSingleStep(5)
        self.second_column_widgets["Indoor and Outdoor Climbing:"].setValue(50)

        # Adding widgets to the grid layout in the first column (left)
        for i, (label, widget) in enumerate(self.first_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 0)
            grid_layout.addWidget(widget, i, 1)

        # Adding widgets to the grid layout in the second column (right)
        for i, (label, widget) in enumerate(self.second_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 2)
            grid_layout.addWidget(widget, i, 3)

        # Save and cancel buttons
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.handle_save)

        layout.addLayout(grid_layout)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def populate_data(self):
        """Fetches user data from the database and pre-populates the form fields."""
        user_data_fields = {
            "name": "Name:",
            "surname": "Surname:",
            "email": "Email:",
            "gender": "Gender:",
            "dominant_arm": "Dominant Arm:",
            "weight": "Weight (kg):",
            "height": "Height (cm):",
            "age": "Age (years):",
            "french_scale": "French Scale Level:",
            "years_climbing": "Years of Climbing:",
            "climbing_freq": "Climbing Frequency/week:",
            "climbing_hours": "Climbing Hours/week:",
            "bouldering": "Bouldering and Lead Climbing:",
            "lead_climbing": "Bouldering and Lead Climbing:",
            "climbing_indoor": "Indoor and Outdoor Climbing:",
            "climbing_outdoor": "Indoor and Outdoor Climbing:",
            "sport_other": "Other sports:",
            "sport_freq": "Sport Frequency/week:",
            "sport_activity_hours": "Sport Activity (hours/week):"
        }

        for field, widget_label in user_data_fields.items():
            data = self.db_manager.get_user_data(self.username, field)
            if field in ["bouldering", "lead_climbing", "climbing_indoor", "climbing_outdoor"]:
                # Special handling for sliders
                self.second_column_widgets[widget_label].setValue(int(data))
            elif field in ["weight", "height", "age", "years_climbing", "climbing_freq", "climbing_hours", "sport_freq",
                           "sport_activity_hours"]:
                # SpinBoxes for numeric values
                self.first_column_widgets[widget_label].setValue(int(data))
            elif field == "gender" or field == "dominant_arm" or field == "french_scale":
                # ComboBoxes for gender, arm, and scale
                self.first_column_widgets[widget_label].setCurrentText(data)
            else:
                # Text fields (QLineEdit)
                self.first_column_widgets[widget_label].setText(data)

    def handle_save(self):
        """Handles saving the updated data into the database."""
        updated_data = {
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            "email": self.first_column_widgets["Email:"].text(),
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            "french_scale": self.second_column_widgets["French Scale Level:"].currentText(),
            "years_climbing": self.second_column_widgets["Years of Climbing:"].value(),
            "climbing_freq": self.second_column_widgets["Climbing Frequency/week:"].value(),
            "climbing_hours": self.second_column_widgets["Climbing Hours/week:"].value(),
            "bouldering": self.second_column_widgets["Bouldering and Lead Climbing:"].value(),
            "lead_climbing": 100 - self.second_column_widgets["Bouldering and Lead Climbing:"].value(),
            "climbing_indoor": self.second_column_widgets["Indoor and Outdoor Climbing:"].value(),
            "climbing_outdoor": 100 - self.second_column_widgets["Indoor and Outdoor Climbing:"].value(),
            "sport_other": self.second_column_widgets["Other sports:"].text(),
            "sport_freq": self.second_column_widgets["Sport Frequency/week:"].value(),
            "sport_activity_hours": self.second_column_widgets["Sport Activity (hours/week):"].value()
        }

        # Save the updated data in the database
        success = self.db_manager.update_user_data(self.username, updated_data)

        if success:
            QMessageBox.information(self, "Success", "Settings updated successfully!")
        else:
            QMessageBox.warning(self, "Error", "An error occurred while updating settings.")
