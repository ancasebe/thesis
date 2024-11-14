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
    Class to facilitate communication with serial interface for data collection.

    This class manages the connection to a serial device, handles data retrieval
    and command writing, and allows for asynchronous data collection using threads.

    Attributes:
        db_queque (queue.Queue): Queue for storing raw data from the serial interface.
        viz_queque (queue.Queue): Queue for downsampled data intended for visualization.
        baudrate (int): Baud rate for the serial communication (default is 19200).
        data_viz_downsampling: (int): Factor for downsampling data for visualization.
        collected (int): Count of collected data entries.
        show_raw (bool): Attribute to set whether calibration profile is applied to data or none

    Methods:
        write_command(command: str):
            Sends a command to the serial device.

        start_collection():
            Starts the data collection process in a separate thread.

        stop_collection():
            Stops the data collection and terminates the serial connection.

        get_calibration_profile(slope, intercept):
            Sets "show_raw" attribute to False, and input calibration profile


    Exceptions:
        Raises ReferenceError if no serial devices are connected.
    """

    def __init__(self, baudrate: int = 19200, data_viz_downsampling: int = 5) -> None:
        self.db_queque = queue.Queue(maxsize=850)
        self.viz_queque = queue.Queue()
        self.baudrate = baudrate
        self.connection = False
        self.sercon = None
        self.collect_data = False
        self.writing_command = False
        self.data_collection_thread = None
        self.downsampling = data_viz_downsampling
        self.downsampling_counter = data_viz_downsampling
        self.collected = 0
        self.show_raw = True
        self.calibration_slope = 0
        self.calibration_intercept = 0
        self.command = None
        self.reset_timer = None
        self.debugging = False
        self.debugging_data = []
        self.port = self.__serial_get_port()

    def __serial_get_force_data(self) -> str:
        if self.connection == False:
            self.sercon = self.__serial_start_connection()
        sercon = self.sercon
        if sercon.in_waiting > 0 and self.writing_command == False:
            serial_data = sercon.readline().decode('utf-8').rstrip()
            try:
                serial_data = int(serial_data)
                if self.show_raw == False and self.calibration_intercept != 0:
                    serial_data = (
                        serial_data-self.calibration_intercept)/self.calibration_slope
            except ValueError:
                pass
            self.collected += 1
            return f'{FORCE_SENSOR_ID}_{time.time():.3f}_{serial_data}'
        elif self.writing_command and self.command is not None:
            sercon.write(self.command.encode())
            self.command = None
            self.writing_command = False
        elif self.writing_command and self.command is None:
            self.writing_command = False

    def __serial_start_connection(self) -> serial.Serial:
        if self.connection == True:
            return
        connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1
        )
        self.connection = True
        return connection

    def __serial_list_ports(self) -> list:
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        return port_list

    def __serial_get_port(self):
        ports = self.__serial_list_ports()
        if len(ports) == 0:
            self.debugging = True
        elif len(ports) == 1:
            return ports[0]
        else:
            text = f'Enter number of selected port:\n'
            for i in range(len(ports)):
                text += f'{i+1} - {ports[i]}\n'
            while (True):
                try:
                    port = int(input(text))
                    if port <= len(ports):
                        return (ports[port-1])
                except:
                    pass

    def __serial_terminate_connection(self):
        connection = self.sercon
        port = connection.port
        connection.close()
        print(f'Connection {port} closed')
        return 1

    def __debugging_generate_force_profile(self, noise_level=0.4):
        """Generates synthetic data that mimics a force profile on a hangboard."""
        n_samples = int(np.random.uniform(300,800))
        t = np.linspace(0, np.pi, n_samples)
        max_force = np.random.uniform(1, 60)
        force_profile = max_force * np.sin(t)
        noise = np.random.normal(0, noise_level, n_samples)
        force_profile = np.clip(force_profile + noise, 0, max_force)
        return force_profile

    def __debugging_data_force_generator(self):
        while self.collect_data:
            if len(self.debugging_data) <= 1:
                self.debugging_data = self.__debugging_generate_force_profile(noise_level=np.random.uniform(0.2,0.6)).tolist()
                zeros_at_end = np.random.uniform(0,0.8,int(np.random.uniform(100,1000))).tolist()
                self.debugging_data.extend(zeros_at_end)
            serial_data = self.debugging_data.pop(0)
            data = f'{FORCE_SENSOR_ID}_{time.time():.3f}_{serial_data}'
            self.__data_generator_que_manager(data)
            time.sleep(0.01)

    def __data_generator_que_manager(self, data):
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

    def __data_generator(self):
        while self.collect_data:
            data = self.__serial_get_force_data()
            if data is not None:
                self.__data_generator_que_manager(data)
            elif self.reset_timer is None or self.reset_timer + 10 > time.time():
                self.reset_timer = time.time()
            elif self.reset_timer + 10 < time.time():
                self.connection = False
                self.sercon = None
                self.reset_timer = None
                while not self.viz_queque.empty():
                    self.viz_queque.get()
            time.sleep(0.005)

    def start_collection(self):
        if self.data_collection_thread is not None:
            return
        if self.debugging == False:
            self.data_collection_thread = threading.Thread(
                target=self.__data_generator)
        elif self.debugging == True:
            self.data_collection_thread = threading.Thread(
                target=self.__debugging_data_force_generator)
        self.collect_data = True
        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()
        return

    def stop_collection(self):
        if self.data_collection_thread is not None:
            self.collect_data = False
            self.data_collection_thread.join()
            self.data_collection_thread = None
            if self.debugging == False:
                self.__serial_terminate_connection()
            self.connection = False
            return 1
        else:
            return

    def write_command(self, command: str):
        self.writing_command = True
        self.command = command
        return 1

    def input_calibration_profile(self, slope: float, intercept: float):
        self.show_raw = False
        self.calibration_intercept = intercept
        self.calibration_slope = slope
