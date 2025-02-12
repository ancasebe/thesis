"""
This module defines the `EditClimberInfoPage` class, which provides an interface for editing a climber's data.

Key functionalities:
- Displays the current data of the selected climber in a form layout.
- Allows admins to update fields such as personal info, climbing preferences, and activity levels.
- Saves changes to the database and includes validation to ensure restricted access based on admin ID.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QSpinBox, QLabel, QSlider,
                               QComboBox, QPushButton, QGridLayout, QMessageBox, QHBoxLayout)
from PySide6.QtCore import Qt


class EditClimberInfoPage(QWidget):
    """
    A class representing the settings page where users can edit their registration data.

    Methods:
        setup_ui: Sets up the user interface for the settings page.
        populate_data: Fetches and populates the user's current data from the database.
        handle_save: Saves the updated user data into the database.
    """

    def __init__(self, admin_id, email, switch_to_test_page, climber_db_manager):
        """Initializes the settings page and pre-fills user data."""
        super().__init__()
        self.admin_id = admin_id  # ID of the admin to restrict access
        self.email = email  # Climber's unique identifier
        self.climber_db_manager = climber_db_manager
        self.switch_to_test_page = switch_to_test_page
        self.first_column_widgets = {}
        self.second_column_widgets = {}
        self.bouldering_lead_slider = None
        self.indoor_outdoor_slider = None
        self.setup_ui()
        self.populate_data()
        # Flag to indicate whether the user is saving
        self.is_saving = False

    def setup_ui(self):
        """Sets up the layout and widgets for the settings page."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Create a grid layout for organizing form in columns
        grid_layout = QGridLayout()

        # Dictionary of inputs and their labels for the first and second columns (same as RegistrationPage)
        self.first_column_widgets = {
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
            "Other sports:": QLineEdit(),
            "Sport Frequency/week:": QSpinBox(),
            "Sport Activity (hours/week):": QSpinBox()
        }

        # Configuring certain widgets
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

        # Adding widgets to the grid layout in the first column (left)
        for i, (label, widget) in enumerate(self.first_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 0)
            grid_layout.addWidget(widget, i, 1)

        # Adding standard widgets to the grid layout in the second column (right)
        for i, (label, widget) in enumerate(self.second_column_widgets.items()):
            grid_layout.addWidget(QLabel(label), i, 2)
            grid_layout.addWidget(widget, i, 3)

        # Add the custom slider layouts for Bouldering/Lead and Indoor/Outdoor climbing
        bouldering_lead_layout, self.bouldering_lead_slider = self.create_slider_layout("Bouldering", "Lead Climbing")
        indoor_outdoor_layout, self.indoor_outdoor_slider = self.create_slider_layout("Indoor", "Outdoor Climbing")

        # Add these custom layouts directly into the grid layout without dictionary keys
        grid_layout.addLayout(bouldering_lead_layout, len(self.second_column_widgets), 2, 1, 2)
        grid_layout.addLayout(indoor_outdoor_layout, len(self.second_column_widgets) + 1, 2, 1, 2)

        # Save button
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.handle_save)

        # switch_button = QPushButton("Back to Test Selection")
        # switch_button.clicked.connect(self.switch_to_test_page)

        layout.addLayout(grid_layout)
        layout.addWidget(save_button)
        # layout.addWidget(switch_button)

        self.setLayout(layout)

    def create_slider_layout(self, left_label_text, right_label_text):
        """Creates a layout for a slider with labels on either side to represent percentages."""
        layout = QHBoxLayout()
        left_label = QLabel(f"{left_label_text}: 50%")
        right_label = QLabel(f"{right_label_text}: 50%")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setSingleStep(5)
        slider.setValue(50)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                background: gray; /* Sets the entire slider track to blue */
            }
            QSlider::handle:horizontal {
                background: white; /* Handle color */
                border: 1px solid black;
                width: 10px;
                margin: -5px 0; /* Centers the handle on the groove */
                border-radius: 7px;
            }
            """)
        slider.valueChanged.connect(lambda value: self.update_slider_labels(value, left_label, right_label))
        layout.addWidget(left_label)
        layout.addWidget(slider)
        layout.addWidget(right_label)
        return layout, slider

    @staticmethod
    def update_slider_labels(value, left_label, right_label):
        """Updates the labels for a slider to show switched percentage values."""
        left_label.setText(f"{left_label.text().split(':')[0]}: {100 - value}%")
        right_label.setText(f"{right_label.text().split(':')[0]}: {value}%")

    def populate_data(self):
        """Fetches and pre-fills the climber's data into the form fields based on their email."""
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
            "sport_other": "Other sports:",
            "sport_freq": "Sport Frequency/week:",
            "sport_activity_hours": "Sport Activity (hours/week):"
        }

        climber_data = self.climber_db_manager.get_user_data(self.admin_id, self.email)

        if climber_data:
            for field, widget_label in user_data_fields.items():
                widget = self.first_column_widgets.get(widget_label) or self.second_column_widgets.get(widget_label)
                data = climber_data.get(field)

                if widget:
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(data) if data else 0)
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(data or "")
                    elif isinstance(widget, QLineEdit):
                        widget.setText(data or "")

            # Set slider values based on database values
            self.bouldering_lead_slider.setValue(climber_data.get("bouldering", 50))
            self.indoor_outdoor_slider.setValue(climber_data.get("climbing_indoor", 50))
        else:
            QMessageBox.warning(self, "Access Denied", "You do not have permission to access this climber's data.")

    def handle_save(self):
        """Handles saving the updated data into the database."""

        self.is_saving = True  # Set the flag to indicate saving is in progress
        self.email = self.first_column_widgets["Email:"].text()

        updated_data = {
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            # "email": self.first_column_widgets["Email:"].text(),
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            "french_scale": self.second_column_widgets["French Scale Level:"].currentText(),
            "years_climbing": self.second_column_widgets["Years of Climbing:"].value(),
            "climbing_freq": self.second_column_widgets["Climbing Frequency/week:"].value(),
            "climbing_hours": self.second_column_widgets["Climbing Hours/week:"].value(),
            "bouldering": 100 - self.bouldering_lead_slider.value(),
            "lead_climbing": self.bouldering_lead_slider.value(),
            "climbing_indoor": 100 - self.indoor_outdoor_slider.value(),
            "climbing_outdoor": self.indoor_outdoor_slider.value(),
            "sport_other": self.second_column_widgets["Other sports:"].text(),
            "sport_freq": self.second_column_widgets["Sport Frequency/week:"].value(),
            "sport_activity_hours": self.second_column_widgets["Sport Activity (hours/week):"].value()
        }

        # Save the updated data in the database
        success = self.climber_db_manager.update_climber_data(self.admin_id, self.email, updated_data)

        if success:
            QMessageBox.information(self, "Success", "Settings updated successfully!")
            if hasattr(self, 'dialog'):
                self.dialog.close()
            self.switch_to_test_page()  # Redirect to the main page or next step
        else:
            # self.is_saving = False  # Reset the flag if saving fails
            QMessageBox.warning(self, "Error", "An error occurred while updating settings.")

    def close_event(self, event):
        # """
        # Intercepts the close event and prompts the user for confirmation.
        # """
        # print(self.is_saving)
        # # If saving, allow the window to close without showing a dialog
        # if self.is_saving:
        #     self.main_stacked_widget.setCurrentWidget(self.switch_to_test_page)
        #     # event.accept()
        #     return
        #
        # reply = QMessageBox.question(
        #     self,
        #     "Confirm Exit",
        #     "Are you sure you want to leave without saving?",
        #     QMessageBox.Yes | QMessageBox.No,
        #     QMessageBox.No
        # )
        # # if reply == QMessageBox.Yes:
        # #     self.main_stacked_widget.setCurrentWidget(self.switch_to_test_page)  # Navigate back
        # if reply == QMessageBox.Yes:
        #     # self.switch_to_test_page()  # Redirect to the test page
        #     self.main_stacked_widget.setCurrentWidget(self.switch_to_test_page)
        #     # event.accept()  # Close the window
        # else:
        #     event.ignore()  # Cancel the close action

        """Intercepts the close event to confirm unsaved changes."""
        if not self.is_saving:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to leave without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
