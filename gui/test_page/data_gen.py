import serial
import serial.tools
import serial.tools.list_ports
import time
import queue
import threading
import numpy as np

FORCE_SENSOR_ID = 1


class SerialCommunicator:
    """
    Handles communication with a serial interface for data collection.
    Supports asynchronous data retrieval using threads and queue-based storage.
    """

    def __init__(self, baudrate: int = 19200, data_viz_downsampling: int = 5, debugging=False) -> None:
        """
        Initializes the SerialCommunicator with default values.
        """
        self.db_queque = queue.Queue(maxsize=850)  # Stores raw serial data
        self.viz_queque = queue.Queue()  # Stores downsampled data for visualization
        self.baudrate = baudrate  # Serial baud rate
        self.connection = False  # Connection status
        self.sercon = None  # Serial connection object
        self.collect_data = False  # Flag to control data collection
        self.writing_command = False  # Flag for sending commands
        self.data_collection_thread = None  # Thread for collecting data
        self.downsampling = data_viz_downsampling  # Downsampling factor
        self.downsampling_counter = data_viz_downsampling  # Counter for downsampling
        self.collected = 0  # Counter for collected data points
        self.show_raw = True  # Flag for using raw or calibrated data
        self.calibration_slope = 0  # Calibration slope
        self.calibration_intercept = 0  # Calibration intercept
        self.command = None  # Command to send to the serial device
        self.reset_timer = None  # Timer to handle resets
        self.debugging = debugging  # Debugging mode flag
        self.debugging_data = []  # Placeholder for simulated data
        # Only initialize the port if debugging is disabled
        self.port = None if self.debugging else self.__serial_get_port()  # Get the serial port

    def __serial_start_connection(self) -> serial.Serial:
        """Establishes a serial connection if one does not exist."""
        if self.debugging:
            return None  # Do not attempt a serial connection in debug mode
        if self.connection:
            return
        connection = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
        self.connection = True
        return connection

    def __serial_get_force_data(self) -> str:
        """Reads force data from the serial device and applies calibration if needed."""
        if self.debugging:
            return None  # Do not attempt to read from serial in debug mode

        if not self.connection:
            self.sercon = self.__serial_start_connection()
        sercon = self.sercon

        if sercon and sercon.in_waiting > 0 and not self.writing_command:
            serial_data = sercon.readline().decode('utf-8').rstrip()
            try:
                serial_data = int(serial_data)
                if not self.show_raw and self.calibration_intercept != 0:
                    serial_data = (serial_data - self.calibration_intercept) / self.calibration_slope
            except ValueError:
                pass
            self.collected += 1
            return f"{FORCE_SENSOR_ID}_{time.time():.3f}_{serial_data}"
        elif self.writing_command and self.command is not None:
            sercon.write(self.command.encode())
            self.command = None
            self.writing_command = False

    def __serial_list_ports(self) -> list:
        """Lists available serial ports and returns them as a list."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def __serial_get_port(self):
        """Selects a serial port from the available list or prompts user if multiple exist."""
        ports = self.__serial_list_ports()
        if len(ports) == 0:
            self.debugging = True
        elif len(ports) == 1:
            return ports[0]
        else:
            text = 'Enter number of selected port:\n'
            for i, port in enumerate(ports, 1):
                text += f'{i} - {port}\n'
            while True:
                try:
                    port_index = int(input(text))
                    if 1 <= port_index <= len(ports):
                        return ports[port_index - 1]
                except ValueError:
                    pass

    def start_collection(self):
        """Starts the data collection process in a separate thread."""
        if self.data_collection_thread is not None:
            return

        self.collect_data = True

        if not self.debugging:
            self.data_collection_thread = threading.Thread(target=self.__data_generator)
        else:
            self.data_collection_thread = threading.Thread(target=self.__debugging_data_force_generator)

        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()

    def stop_collection(self):
        """Stops the data collection process and terminates the serial connection."""
        if self.data_collection_thread is not None:
            self.collect_data = False
            self.data_collection_thread.join()
            self.data_collection_thread = None
            if not self.debugging:
                self.__serial_terminate_connection()
            self.connection = False

    def __data_generator(self):
        """Continuously retrieves and processes data from the serial device."""
        while self.collect_data:
            data = self.__serial_get_force_data()
            if data is not None:
                self.__data_generator_que_manager(data)
            time.sleep(0.005)  # Prevents CPU overload

    def __debugging_data_force_generator(self):
        """Simulates force data for debugging mode."""
        while self.collect_data:
            if len(self.debugging_data) <= 1:
                self.debugging_data = self.__debugging_generate_force_profile(
                    noise_level=np.random.uniform(0.2, 0.6)
                ).tolist()
            serial_data = self.debugging_data.pop(0)
            data = f"{FORCE_SENSOR_ID}_{time.time():.3f}_{serial_data}"
            self.__data_generator_que_manager(data)
            time.sleep(0.01)

    def __debugging_generate_force_profile(self, noise_level=0.4):
        """Generates synthetic data that mimics a force profile on a hangboard."""
        n_samples = int(np.random.uniform(300,800))
        t = np.linspace(0, np.pi, n_samples)
        max_force = np.random.uniform(1, 60)
        force_profile = max_force * np.sin(t)
        noise = np.random.normal(0, noise_level, n_samples)
        force_profile = np.clip(force_profile + noise, 0, max_force)
        return force_profile

    def __data_generator_que_manager(self, data):
        """Manages the data queue, ensuring that data is stored and downsampled correctly."""
        try:
            self.db_queque.put_nowait(data)
        except queue.Full:
            self.db_queque.get()
            self.db_queque.put_nowait(data)

        self.downsampling_counter -= 1
        if self.downsampling_counter == 0:
            if self.viz_queque.qsize() >= 200:
                self.viz_queque.get()
            self.viz_queque.put_nowait(data)
            self.downsampling_counter = self.downsampling

    def write_command(self, command: str):
        """Sends a command to the serial device."""
        self.writing_command = True
        self.command = command

    def input_calibration_profile(self, slope: float, intercept: float):
        """Applies a calibration profile to the force data."""
        self.show_raw = False
        self.calibration_intercept = intercept
        self.calibration_slope = slope

    def __serial_terminate_connection(self):
        """Closes the serial connection and resets parameters."""
        connection = self.sercon
        port = connection.port
        connection.close()
        print(f"Connection {port} closed")
        self.connection = False
