"""
This module defines the `TestPage` class, which provides an interface for conducting various
tests with research members.

Key functionalities:
- Allows selection of a test and the arm to be tested.
- Requires selection of a research member before a test can be started.
- Integrates with the database to save test data linked to the chosen research member.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QSizePolicy, QComboBox, QMessageBox, QFormLayout
)
from PySide6.QtCore import Qt


class TestPage(QWidget):
    """
    The Test Page represents the testing interface, allowing the user to select a test,
    choose a research member, specify the arm being tested, and save the data in a database.

    Args:
        db_manager (DatabaseManager): Database manager to handle data saving and retrieval.
        admin_id (int): ID of the logged-in admin.
        main_stacked_widget (QStackedWidget): The main stacked widget to manage page transitions.
    """

    def __init__(self, db_manager, admin_id, main_stacked_widget):
        super().__init__()
        self.db_manager = db_manager
        self.admin_id = admin_id
        self.main_stacked_widget = main_stacked_widget
        self.selected_test = None  # Store the currently selected test
        self.setup_ui()
        self.load_climbers()

    def setup_ui(self):
        """Sets up the user interface for the Test Page."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Title
        title_label = QLabel("Testing Interface")
        title_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(title_label)

        # ComboBox for selecting a research member (climber)
        self.climber_selector = QComboBox()
        self.climber_selector.addItem("-")  # Default prompt
        # main_layout.addWidget(self.climber_selector)

        # Form layout for Research Member
        form_layout = QFormLayout()
        form_layout.addRow("Climber Tested:", self.climber_selector)
        main_layout.addLayout(form_layout)

        # Arm Tested ComboBox
        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["-", "dominant", "non-dominant"])

        # Form layout for Arm Tested
        form_layout = QFormLayout()
        form_layout.addRow("Arm Tested:", self.arm_tested_combo)
        main_layout.addLayout(form_layout)

        # Add test selection buttons
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        self.add_test_button(button_layout, "Maximal Voluntary Contraction")
        self.add_test_button(button_layout, "All Out")
        self.add_test_button(button_layout, "Hang Test")
        self.add_test_button(button_layout, "Test to Exhaustion")

        main_layout.addLayout(button_layout)

        # # Save Button
        # save_button = QPushButton("Save Selection")
        # save_button.clicked.connect(self.save_test_selection)
        # main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def load_climbers(self):
        """Loads all climbers registered by the current admin into the ComboBox."""
        climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
        self.climber_selector.clear()
        self.climber_selector.addItem("-")  # Default option
        for climber in climbers:
            display_name = f"{climber['name']} {climber['surname']}"
            self.climber_selector.addItem(display_name, climber["email"])
            print(f"Loaded climber: {display_name} with email {climber['email']}")  # Debugging

    def add_test_button(self, layout, test_name):
        """
        Adds a test button to the layout.

        Args:
            layout (QVBoxLayout): The layout where the button will be added.
            test_name (str): The name of the test.
        """
        button = QPushButton(test_name)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.clicked.connect(lambda: self.select_test(test_name))
        layout.addWidget(button)

    def select_test(self, test_name):
        """Sets the selected test and validates the selections."""
        selected_climber = self.climber_selector.currentData()
        selected_arm = self.arm_tested_combo.currentText()

        # Validate climber and arm selections
        if selected_climber == "Select a member" or selected_climber is None:
            QMessageBox.warning(self, "No Climber Selected", "Please select a research member to proceed.")
            return
        if selected_arm == "-":
            QMessageBox.warning(self, "No Arm Selected", "Please select an arm to test.")
            return

        self.selected_test = test_name
        QMessageBox.information(self, "Test Selected", f"Selected Test: {test_name} for {selected_climber}")

    def save_test_selection(self):
        """
        Saves the selected test, arm, and climber to the database.
        """
        selected_climber = self.climber_selector.currentData()
        selected_arm = self.arm_tested_combo.currentText()
        selected_test = self.selected_test

        if not selected_climber or selected_climber == "Select a member":
            QMessageBox.warning(self, "No Climber Selected", "Please select a research member before saving.")
            return
        if selected_arm == "Select Arm":
            QMessageBox.warning(self, "No Arm Selected", "Please select an arm before saving.")
            return
        if not selected_test:
            QMessageBox.warning(self, "No Test Selected", "Please select a test before saving.")
            return

        # Placeholder for saving to the database
        QMessageBox.information(
            self, "Save Confirmation",
            f"Test: {selected_test}\nClimber: {selected_climber}\nArm: {selected_arm}\nData ready to be saved."
        )

        # Implement code to save `selected_test`, `selected_climber`, and `selected_arm` to the database here
