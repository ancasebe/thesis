from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QSizePolicy, QComboBox, QMessageBox, QFormLayout
)
from PySide6.QtCore import Qt


class TestPage(QWidget):
    """
    The Test Page represents the testing interface with a menu on the left and additional test options
    in the center. It allows the user to select a test, choose the arm tested, and prepares the data
    to be saved in a database.
    """

    def __init__(self, username, logout_callback):
        """Initializes the Test Page with a username and a logout callback."""
        super().__init__()
        self.username = username
        self.logout_callback = logout_callback
        self.setup_ui()

    def setup_ui(self):
        """Sets up the user interface for the Test Page."""
        main_layout = QHBoxLayout()
        center_layout_wrapper = QVBoxLayout()  # This will hold the center layout and center it on the page

        # Test buttons in the center
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)  # Center the buttons vertically

        # Test Options
        self.add_test_button(center_layout, "Maximal Voluntary Contraction")
        self.add_test_button(center_layout, "All Out")
        self.add_test_button(center_layout, "Hang Test")
        self.add_test_button(center_layout, "Test to Exhaustion")

        # ComboBox for Arm Tested selection
        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["-", "dominant", "non-dominant"])

        # Create a layout for the combo box and label
        form_layout = QFormLayout()
        form_layout.addRow("Arm Tested:", self.arm_tested_combo)

        # Add everything to the center layout wrapper
        center_layout.addLayout(form_layout)

        # Add the center layout to the wrapper and align it in the middle
        center_layout_wrapper.addLayout(center_layout)
        center_layout_wrapper.setAlignment(Qt.AlignCenter)

        # Add the wrapper layout to the main layout
        main_layout.addLayout(center_layout_wrapper)

        self.setLayout(main_layout)

    def add_test_button(self, layout, test_name):
        """
        Adds a test button to the layout.

        Args:
            layout (QVBoxLayout): The layout where the button will be added.
            test_name (str): The name of the test.
        """
        button = QPushButton(test_name)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.clicked.connect(lambda: self.save_test_selection(test_name))
        layout.addWidget(button)

    def save_test_selection(self, test_name):
        """
        Prepares the selected test and arm option to be saved to the database.

        Args:
            test_name (str): The name of the selected test.
        """
        selected_arm = self.arm_tested_combo.currentText()

        # Display the chosen test and arm as a placeholder for saving to the database
        QMessageBox.information(self, "Test Selection",
                                f"Selected Test: {test_name}\nArm Tested: {selected_arm}")

        # Here, you would prepare the data to be saved to the database for the current user.

    def show_main_page(self):
        """Placeholder for navigating to the main page."""
        QMessageBox.information(self, "Main Page", "Navigating to the main page...")

    def show_settings(self):
        """Placeholder for navigating to the settings page."""
        QMessageBox.information(self, "Settings", "Navigating to the settings page...")

    # def validate_inputs(self):
    #     """
    #     Validates that all input fields are filled.
    #
    #     Returns:
    #         bool: True if all inputs are valid, False otherwise.
    #     """
    #     value = self.arm_tested_combo.currentText()
    #
    #     if isinstance(value, str) and value == "-":
    #         errors.append(f"{field_name} must be filled.")

