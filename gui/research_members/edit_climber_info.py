"""
Climber information editing module for the Climbing Testing Application.

This module defines the EditClimberInfoPage class which provides the user interface
for modifying existing climber information. It allows researchers to update
participant details, climbing experience, and other relevant information.

Key functionalities:
- Display existing climber information in an editable form
- Organize input fields in a structured layout with appropriate validation
- Update climber records in the database when changes are confirmed
- Implement interactive sliders for preference distributions (bouldering/lead, indoor/outdoor)
- Confirm unsaved changes before navigation away from the edit page
- Restrict access based on admin permissions

The climber editing module ensures that participant information can be maintained
accurately throughout the research process while preserving data integrity.
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

    def __init__(self, admin_id, climber_id, switch_to_test_page, climber_db_manager):
        """Initializes the settings page and pre-fills user data."""
        super().__init__()
        self.admin_id = admin_id  # ID of the admin to restrict access
        self.climber_id = climber_id  # Climber's unique identifier
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
            "IRCRA Scale Level:": QComboBox(),
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

        self.second_column_widgets["IRCRA Scale Level:"].addItems(
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']
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
        """Fetches climber data from the database and populates the form fields."""
        climber_data = self.climber_db_manager.get_user_data(self.admin_id, self.climber_id)
        
        if not climber_data:
            return
        
        # Populate fields from the dictionaries instead of direct attributes
        # Basic info fields (first column)
        self.first_column_widgets["Name:"].setText(climber_data.get('name', ''))
        self.first_column_widgets["Surname:"].setText(climber_data.get('surname', ''))
        self.first_column_widgets["Email:"].setText(climber_data.get('email', ''))
        
        gender_index = self.first_column_widgets["Gender:"].findText(climber_data.get('gender', ''))
        if gender_index >= 0:
            self.first_column_widgets["Gender:"].setCurrentIndex(gender_index)
        
        arm_index = self.first_column_widgets["Dominant Arm:"].findText(climber_data.get('dominant_arm', ''))
        if arm_index >= 0:
            self.first_column_widgets["Dominant Arm:"].setCurrentIndex(arm_index)
        
        # Set numeric values properly
        self.first_column_widgets["Weight (kg):"].setValue(int(climber_data.get('weight', 0) or 0))
        self.first_column_widgets["Height (cm):"].setValue(int(climber_data.get('height', 0) or 0))
        self.first_column_widgets["Age (years):"].setValue(int(climber_data.get('age', 0) or 0))
        
        # Climbing info fields (second column)
        ircra_index = self.second_column_widgets["IRCRA Scale Level:"].findText(str(climber_data.get('ircra', '')))
        if ircra_index >= 0:
            self.second_column_widgets["IRCRA Scale Level:"].setCurrentIndex(ircra_index)
        
        # Set numeric values properly
        self.second_column_widgets["Years of Climbing:"].setValue(int(climber_data.get('years_climbing', 0) or 0))
        self.second_column_widgets["Climbing Frequency/week:"].setValue(int(climber_data.get('climbing_freq', 0) or 0))
        self.second_column_widgets["Climbing Hours/week:"].setValue(int(climber_data.get('climbing_hours', 0) or 0))
        
        # Update sliders - assuming values are between 0-100
        bouldering_value = int(climber_data.get('lead_climbing', 50) or 50)
        self.bouldering_lead_slider.setValue(bouldering_value)
        
        indoor_value = int(climber_data.get('climbing_outdoor', 50) or 50)
        self.indoor_outdoor_slider.setValue(indoor_value)
        
        # Sport info fields
        self.second_column_widgets["Other sports:"].setText(climber_data.get('sport_other', ''))
        self.second_column_widgets["Sport Frequency/week:"].setValue(int(climber_data.get('sport_freq', 0) or 0))
        self.second_column_widgets["Sport Activity (hours/week):"].setValue(int(climber_data.get('sport_activity_hours', 0) or 0))

    def handle_save(self):
        """Handles saving the updated data into the database."""

        self.is_saving = True  # Set the flag to indicate saving is in progress
        # self.climber_id = self.first_column_widgets["Email:"].text()

        updated_data = {
            "name": self.first_column_widgets["Name:"].text(),
            "surname": self.first_column_widgets["Surname:"].text(),
            "email": self.first_column_widgets["Email:"].text(),
            "gender": self.first_column_widgets["Gender:"].currentText(),
            "dominant_arm": self.first_column_widgets["Dominant Arm:"].currentText(),
            "weight": self.first_column_widgets["Weight (kg):"].value(),
            "height": self.first_column_widgets["Height (cm):"].value(),
            "age": self.first_column_widgets["Age (years):"].value(),
            "ircra": self.second_column_widgets["IRCRA Scale Level:"].currentText(),
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
        success = self.climber_db_manager.update_climber_data(self.admin_id, self.climber_id, updated_data)

        if success:
            QMessageBox.information(self, "Success", "Settings updated successfully!")
            if hasattr(self, 'dialog'):
                self.dialog.close()
            editing = True
            self.switch_to_test_page(editing)  # Redirect to the main page or next step
        else:
            # self.is_saving = False  # Reset the flag if saving fails
            QMessageBox.warning(self, "Error", "An error occurred while updating settings.")

    def closeEvent(self, event):
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
# TOD: superuser handling - vybrani admina pro testovani, editace uzivatelu? pro jakeho admina pridat uzivatele?