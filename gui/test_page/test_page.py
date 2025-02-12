"""
This module defines the `TestPage` class, which provides a unified interface for managing climbers
and conducting tests.

Key functionalities:
- Allows admins to add, edit, and delete climbers.
- Enables selection of a test, climber, and arm for testing.
- Ensures validation before conducting a test and saves test data to the database.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QFormLayout, QStackedWidget, QSizePolicy, QDialog, QGridLayout
)
from PySide6.QtCore import Qt

from gui.research_members.edit_climber_info import EditClimberInfoPage
from gui.research_members.new_climber import NewClimber


class TestPage(QWidget):
    """
    The Test Page allows admins to manage climbers and conduct tests.

    Features:
    - Add, edit, delete climbers.
    - Select a climber, choose a test and arm, and save the test data.

    Args:
        db_manager (DatabaseManager): Database manager for data saving and retrieval.
        admin_id (int): ID of the logged-in admin.
        main_stacked_widget (QStackedWidget): Manages page transitions.
    """

    def __init__(self, db_manager, admin_id, main_stacked_widget):
        super().__init__()
        self.db_manager = db_manager
        self.admin_id = admin_id
        self.main_stacked_widget = main_stacked_widget
        self.selected_test = None
        self.setup_ui()
        self.load_climbers()

    '''
    def setup_ui(self):
        """Sets up the user interface."""
        
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Title
        # title_label = QLabel("Manage Research Members & Testing")
        # title_label.setStyleSheet("font-size: 24px;")
        # main_layout.addWidget(title_label)
        
        # Arm selection
        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["Select Arm", "dominant", "non-dominant"])
        form_layout = QFormLayout()
        form_layout.addRow("Arm Tested:", self.arm_tested_combo)

        # Climber selection
        self.climber_selector = QComboBox()
        self.climber_selector.addItem("Select a climber")  # Default option
        form_layout.addRow("Climber Tested:", self.climber_selector)
        main_layout.addLayout(form_layout)

        # Climber management buttons
        climber_button_layout = QHBoxLayout()
        edit_button = QPushButton("Edit Climber Info")
        edit_button.clicked.connect(self.edit_climber_info)
        climber_button_layout.addWidget(edit_button)

        delete_button = QPushButton("Delete Climber")
        delete_button.clicked.connect(self.delete_climber)
        climber_button_layout.addWidget(delete_button)

        add_button = QPushButton("Add New Climber")
        add_button.clicked.connect(self.add_new_climber)
        climber_button_layout.addWidget(add_button)

        main_layout.addLayout(climber_button_layout)

        # Test selection buttons
        button_layout = QVBoxLayout()
        self.add_test_button(button_layout, "Maximal Voluntary Contraction")
        self.add_test_button(button_layout, "All Out")
        self.add_test_button(button_layout, "Hang Test")
        self.add_test_button(button_layout, "Test to Exhaustion")

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        # Title
        title_label = QLabel("Handle with research members and test selection")
        title_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(title_label)

        # Form layout for arm and climber selection
        form_layout = QGridLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setHorizontalSpacing(20)

        # Arm Tested ComboBox
        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["Select Arm", "dominant", "non-dominant"])
        form_layout.addWidget(QLabel("Arm Tested:"), 0, 0)
        form_layout.addWidget(self.arm_tested_combo, 0, 1)

        # Climber Tested ComboBox
        self.climber_selector = QComboBox()
        self.climber_selector.addItem("Select a climber")
        form_layout.addWidget(QLabel("Climber Tested:"), 1, 0)
        form_layout.addWidget(self.climber_selector, 1, 1)

        main_layout.addLayout(form_layout)

        # Climber management buttons
        climber_button_layout = QHBoxLayout()
        edit_button = QPushButton("Edit Climber Info")
        edit_button.clicked.connect(self.edit_climber_info)
        climber_button_layout.addWidget(edit_button)

        delete_button = QPushButton("Delete Climber")
        delete_button.clicked.connect(self.delete_climber)
        climber_button_layout.addWidget(delete_button)

        add_button = QPushButton("Add New Climber")
        add_button.clicked.connect(self.add_new_climber)
        climber_button_layout.addWidget(add_button)

        main_layout.addLayout(climber_button_layout)

        # Test selection buttons in two columns
        button_layout = QGridLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setHorizontalSpacing(20)
        button_layout.setVerticalSpacing(10)

        test_buttons = [
            "Maximal Voluntary Contraction",
            "All Out",
            "Hang Test",
            "Test to Exhaustion"
        ]

        for i, test_name in enumerate(test_buttons):
            button = QPushButton(test_name)
            button.setMinimumSize(150, 75)  # Make the buttons larger
            button.clicked.connect(lambda _, name=test_name: self.select_test(name))
            row, col = divmod(i, 2)  # Two columns
            button_layout.addWidget(button, row, col)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
    '''
    def setup_ui(self):
        """Sets up the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        # Title
        title_label = QLabel("Manage Research Members & Testing")
        title_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(title_label)

        # Form layout for arm and climber selection
        form_layout = QGridLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setHorizontalSpacing(20)

        # Arm Tested ComboBox
        self.arm_tested_combo = QComboBox()
        self.arm_tested_combo.addItems(["Select Arm", "dominant", "non-dominant"])
        form_layout.addWidget(QLabel("Arm Tested:"), 0, 0)
        form_layout.addWidget(self.arm_tested_combo, 0, 1)

        # Climber Tested ComboBox
        self.climber_selector = QComboBox()
        self.climber_selector.addItem("Select a climber")
        form_layout.addWidget(QLabel("Climber Tested:"), 1, 0)
        form_layout.addWidget(self.climber_selector, 1, 1)

        main_layout.addLayout(form_layout)

        # Climber management buttons
        climber_button_layout = QHBoxLayout()

        # Define a common style for buttons
        button_style = """
            QPushButton {
                font-size: 16px;  /* Larger text for readability */
                padding: 10px;    /* Larger padding for a better click area */
                border-radius: 5px;  /* Rounded corners */
                background-color: #007BFF; /* Blue color */
                color: white; /* White text */
            }
            QPushButton:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
            QPushButton:pressed {
                background-color: #003f7f; /* Even darker blue when pressed */
            }
        """

        edit_button = QPushButton("Edit Climber Info")
        edit_button.setStyleSheet(button_style)
        edit_button.clicked.connect(self.edit_climber_info)
        climber_button_layout.addWidget(edit_button)

        delete_button = QPushButton("Delete Climber")
        delete_button.setStyleSheet(button_style)
        delete_button.clicked.connect(self.delete_climber)
        climber_button_layout.addWidget(delete_button)

        add_button = QPushButton("Add New Climber")
        add_button.setStyleSheet(button_style)
        add_button.clicked.connect(self.add_new_climber)
        climber_button_layout.addWidget(add_button)

        main_layout.addLayout(climber_button_layout)

        # Test selection buttons in two columns
        button_layout = QGridLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setHorizontalSpacing(20)
        button_layout.setVerticalSpacing(10)

        test_buttons = [
            "Maximal Voluntary Contraction",
            "All Out",
            "Hang Test",
            "Test to Exhaustion"
        ]

        for i, test_name in enumerate(test_buttons):
            button = QPushButton(test_name)
            button.setStyleSheet(button_style)  # Apply the same style
            button.setMinimumSize(150, 75)  # Larger button size
            button.clicked.connect(lambda _, name=test_name: self.select_test(name))
            row, col = divmod(i, 2)  # Two columns
            button_layout.addWidget(button, row, col)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_climbers(self):
        """Loads all climbers registered by the current admin into the ComboBox."""
        climbers = self.db_manager.get_climbers_by_admin(self.admin_id)
        self.climber_selector.clear()
        self.climber_selector.addItem("Select a climber")
        for climber in climbers:
            display_name = f"{climber['name']} {climber['surname']}"
            self.climber_selector.addItem(display_name, climber["email"])

    def add_test_button(self, layout, test_name):
        """Adds a test button to the layout."""
        button = QPushButton(test_name)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.clicked.connect(lambda: self.select_test(test_name))
        layout.addWidget(button)

    def select_test(self, test_name):
        """Sets the selected test and validates climber and arm selections."""
        selected_climber = self.climber_selector.currentData()
        selected_arm = self.arm_tested_combo.currentText()

        if selected_climber == "Select a climber" or selected_climber is None:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber to proceed.")
            return
        if selected_arm == "Select Arm":
            QMessageBox.warning(self, "No Arm Selected", "Please select an arm to test.")
            return

        self.selected_test = test_name
        QMessageBox.information(self, "Test Selected", f"Selected Test: {test_name} for {selected_climber}")

    def save_test_selection(self):
        """Saves the selected test, arm, and climber to the database."""
        selected_climber = self.climber_selector.currentData()
        selected_arm = self.arm_tested_combo.currentText()
        selected_test = self.selected_test

        if not selected_climber or selected_climber == "Select a climber":
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber before saving.")
            return
        if selected_arm == "Select Arm":
            QMessageBox.warning(self, "No Arm Selected", "Please select an arm before saving.")
            return
        if not selected_test:
            QMessageBox.warning(self, "No Test Selected", "Please select a test before saving.")
            return

        QMessageBox.information(
            self, "Save Confirmation",
            f"Test: {selected_test}\nClimber: {selected_climber}\nArm: {selected_arm}\nData ready to be saved."
        )

    def edit_climber_info(self):
        """Opens an edit dialog for the selected climber."""
        email = self.climber_selector.currentData()
        if email:
            # Create a new dialog for editing climber info
            edit_dialog = QDialog(self)
            edit_dialog.setWindowTitle("Edit Climber Info")
            edit_dialog.setMinimumSize(600, 400)

            # Create an instance of EditClimberInfoPage and set it as the dialog's layout
            edit_page = EditClimberInfoPage(self.admin_id, email, self.reload_climbers, self.db_manager)
            edit_page.dialog = edit_dialog  # Pass the dialog reference to the page
            layout = QVBoxLayout()
            layout.addWidget(edit_page)
            edit_dialog.setLayout(layout)

            # Override the closeEvent of the dialog
            def dialog_close_event(event):
                """Asks for confirmation before closing the dialog."""
        #         reply = QMessageBox.question(
        #             edit_dialog,
        #             "Confirm Exit",
        #             "Are you sure you want to leave without saving?",
        #             QMessageBox.Yes | QMessageBox.No,
        #             QMessageBox.No
        #         )
        #         if reply == QMessageBox.Yes:
        #             self.reload_climbers()  # Reload climbers if needed
        #             event.accept()  # Close the dialog
        #         else:
        #             event.ignore()  # Keep the dialog open
        #
        #     # Bind the close event to the custom function
        #     edit_dialog.closeEvent = dialog_close_event
        #
        #     # Show the dialog modally
        #     edit_dialog.exec()
        # else:
        #     QMessageBox.warning(self, "No Climber Selected", "Please select a climber to edit.")
                if not edit_page.is_saving:  # Check if changes were saved
                    reply = QMessageBox.question(
                        edit_dialog,
                        "Confirm Exit",
                        "Are you sure you want to leave without saving?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.reload_climbers()  # Reload climbers
                        event.accept()
                    else:
                        event.ignore()
                else:
                    event.accept()

            edit_dialog.closeEvent = dialog_close_event
            edit_dialog.exec()
        else:
            QMessageBox.warning(self, "No Climber Selected", "Please select a climber to edit.")

    def add_new_climber(self):
        """Opens a dialog to add a new climber."""
        # Create a new dialog for adding a climber
        add_dialog = QDialog(self)
        add_dialog.setWindowTitle("Add New Climber")
        add_dialog.setMinimumSize(600, 400)

        # Create an instance of NewClimber and set it as the dialog's layout
        add_page = NewClimber(self.admin_id, self.reload_climbers, self.db_manager)
        add_page.dialog = add_dialog  # Pass the dialog reference to the page
        add_page.show()  # Open as a standalone window
        layout = QVBoxLayout()
        layout.addWidget(add_page)
        add_dialog.setLayout(layout)

        # Override the closeEvent of the dialog
        def dialog_close_event(event):
            """Asks for confirmation before closing the dialog."""
            # reply = QMessageBox.question(
            #     add_dialog,
            #     "Confirm Exit",
            #     "Are you sure you want to leave without saving?",
            #     QMessageBox.Yes | QMessageBox.No,
            #     QMessageBox.No
            # )
            # if reply == QMessageBox.Yes:
            #     self.reload_climbers()  # Reload climbers if needed
            #     event.accept()  # Close the dialog
            # else:
            #     event.ignore()  # Keep the dialog open
            if not add_page.is_saving:  # Check if changes were saved
                reply = QMessageBox.question(
                    add_dialog,
                    "Confirm Exit",
                    "Are you sure you want to leave without saving?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.reload_climbers()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()

        # Bind the close event to the custom function
        add_dialog.closeEvent = dialog_close_event

        # Show the dialog modally
        add_dialog.exec()

    def delete_climber(self):
        """Deletes the selected climber after confirmation."""
        email = self.climber_selector.currentData()
        name = self.climber_selector.currentText()
        if email:
            reply = QMessageBox.question(
                self, "Delete Climber", f"Are you sure you want to delete climber {name}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.db_manager.delete_climber(email, self.admin_id):
                    QMessageBox.information(self, "Success", f"Climber {name} deleted successfully.")
                    self.load_climbers()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete climber.")

    def reload_climbers(self):
        """Reloads the members list after adding a new member."""
        self.load_climbers()
        self.main_stacked_widget.setCurrentWidget(self)

    # def dialog_close_event(self, event, dialog):
    #     """Asks for confirmation before closing the dialog."""
    #     reply = QMessageBox.question(
    #         dialog,
    #         "Confirm Exit",
    #         "Are you sure you want to leave without saving?",
    #         QMessageBox.Yes | QMessageBox.No,
    #         QMessageBox.No
    #     )
    #     if reply == QMessageBox.Yes:
    #         self.reload_climbers()  # Reload climbers if needed
    #         event.accept()  # Close the dialog
    #     else:
    #         event.ignore()  # Keep the dialog open
