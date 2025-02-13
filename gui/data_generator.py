import serial
import serial.serialutil
import serial.tools
import serial.tools.list_ports
import numpy as np
from bleak import BleakClient, BleakScanner
import time
import threading
import queue
import struct
import asyncio

# from backend_scripts.constants import HHB_SENSOR_ID, O2HB_SENSOR_ID, SMO2_SENSOR_ID, FORCE_SENSOR_ID
# from backend_scripts.utils import JSONCommunicator

HHB_SENSOR_ID = 1
O2HB_SENSOR_ID = 2
SMO2_SENSOR_ID = 3
FORCE_SENSOR_ID = 4

# class ForceBluetoothCommunicator:
#     """
#     A class for managing Bluetooth Low Energy (BLE) communication with a specified device. This class connects to a BLE
#     device, starts receiving notifications, and processes data for visualization and database logging. If the connection
#     fails, it switches to a debug mode, generating synthetic data.
#
#     Attributes:
#         db_queque (queue.Queue): Queue for sending data to the database.
#         viz_queque (queue.Queue): Queue for sending data to the visualization.
#         downsampling (int): Interval at which data is downsampled for visualization purposes.
#         device_name (str): The name of the target BLE device.
#         device_uuid (str): The UUID of the target BLE device for data notifications.
#         reading (bool): Flag to indicate if the device is actively reading data.
#         debugging (bool): Indicates if the class is generating debug data instead of Bluetooth data.
#         data_collection_flag (bool): Indicates if data collection has started.
#         start_reading_time (float): Timestamp for the start of the reading process.
#         connection_attempts (int): Counts connection attempts made to the BLE device.
#         valid_connection_attempts (int): Maximum number of valid connection attempts before switching to debug mode.
#         bluetooth_thread (threading.Thread): Thread running the Bluetooth read operations.
#         data_collection_thread (threading.Thread): Thread to manage data collection for Bluetooth or debug mode.
#
#     Methods:
#
#         - start_bluetooth_collector(): Initializes and starts data collection from the BLE device.
#         - stop_bluetooth_collector(): Stops data collection from the BLE device or debug mode.
#     """
#
#     def __init__(self, visualization_que: queue.Queue, database_que: queue.Queue) -> None:
#         self.device_name = ""  # JSONCommunicator().get_ble_force_config()['dev_name']
#         self.device_uuid = ""  # JSONCommunicator().get_ble_force_config()['dev_data_uuid']
#         self.downsampling = ""  # JSONCommunicator().get_ble_force_config()['force_viz_ds']
#         self.downsampling_counter = self.downsampling
#         self.db_queque = database_que
#         self.viz_queque = visualization_que
#         self.valid_connection_attempts = ""  # JSONCommunicator().get_ble_force_config()['connection_attempts']
#         self.__default_variables()
#
#     def __default_variables(self):
#         self.client = None
#         self.data_queue = queue.Queue()
#         self.reading = False
#         self.bluetooth_thread = None
#         self.start_reading_time = None
#         self.data_collection_flag = False
#         self.connection_attempts = 0
#         self.data_collection_thread = None
#         self.debugging = False
#         self.debugging_data = []
#         self.last_datapoint = None
#         self.intercept = 0
#         self.slope = 0
#         self.show_raw = True
#         self.start_time = time.time()
#         self.sample_counter = 0
#
#     def update_calibration_parameters(self, slope: float, intercept: float):
#         self.slope = slope
#         self.intercept = intercept
#         self.show_raw = False
#
#     async def __async_connect(self):
#         """Connects to the BLE device and ensures services are discovered."""
#         devices = await BleakScanner.discover()
#         for device in devices:
#             if device.name == self.device_name:
#                 self.client = BleakClient(device)
#                 await self.client.connect()
#                 return 1
#         return
#
#     async def __async_start_notifications(self):
#         """Starts receiving notifications from the Bluetooth device."""
#         if self.client and self.client.is_connected:
#             await self.client.start_notify(self.device_uuid, self.__async_notification_handler)
#             self.last_datapoint = time.time()
#             print(f'{time.strftime("%H:%M:%S")} - Started receiving data from {self.device_name}')
#
#     async def __async_stop_notifications(self):
#         if self.client and self.client.is_connected:
#             await self.client.stop_notify(self.device_uuid)
#             print(f'{time.strftime("%H:%M:%S")} - Stopped receiving data from {self.device_name}')
#
#     def decode_unsigned_little_endian(self, data):
#         """Decode 4-byte little-endian data into a float."""
#         time.sleep(0.002)
#         try:
#             int_value = struct.unpack('<B', data)[0]
#         except:
#             int_value = None
#
#         return int_value
#
#     def __process_data(self, data):
#         """Unpacks the data according to the expected format."""
#         float_value = self.decode_unsigned_little_endian(data)
#         if float_value is None:
#             return None
#         if not self.show_raw:
#             float_value = (float_value-self.intercept)/self.slope
#         timestamp = time.time()
#         self.last_datapoint = timestamp
#         formated_vals = f'{FORCE_SENSOR_ID}_{timestamp}_{float_value:.2f}'
#         return formated_vals
#
#     async def __async_notification_handler(self, sender, data):
#         try:
#             data_str = self.__process_data(data)
#             if data_str is None:
#                 return
#             self.data_queue.put(data_str)  # Send data to the queue
#             if self.data_collection_flag == False:
#                 self.data_collection_flag = True
#         except Exception as e:
#             print(f'{time.strftime("%H:%M:%S")} - {e}- Could not process data')
#             print(data)
#
#     def __run(self):
#         """Runs the asynchronous Bluetooth tasks in the thread."""
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#
#         loop.run_until_complete(self.__async_connect())
#         try:
#             loop.run_until_complete(self.__async_start_notifications())
#         except Exception as e:
#             print(f'{time.strftime("%H:%M:%S")} - {e} - Could not start notifications')
#         try:
#             while self.reading:
#                 loop.run_until_complete(asyncio.sleep(0.001))
#         finally:
#             if self.client is not None:
#                 try:
#                     loop.run_until_complete(self.__async_stop_notifications())
#                     loop.run_until_complete(self.client.disconnect())
#                     print(
#                         f'{time.strftime("%H:%M:%S")} - Disconnected from {self.device_name}')
#                 except Exception as e:
#                     print(f'{time.strftime("%H:%M:%S")} - {e}')
#
#     def __start_asynchronous_thread(self):
#         """Starts the Bluetooth communicator in a separate thread."""
#         self.reading = True
#         self.start_reading_time = time.time()
#         self.bluetooth_thread = threading.Thread(target=self.__run)
#         self.bluetooth_thread.daemon = True
#         self.bluetooth_thread.start()
#         while self.data_collection_flag == False:
#             if self.start_reading_time+15 < time.time():
#                 self.start_reading_time = time.time()
#                 self.connection_attempts += 1
#                 print(f'{time.strftime("%H:%M:%S")} - Could not Connect to  {self.device_name}. '
#                       f'Reconnecting ({self.connection_attempts}/{self.valid_connection_attempts})')
#                 self.bluetooth_thread.join(timeout=1)
#                 self.bluetooth_thread = threading.Thread(target=self.__run)
#                 self.bluetooth_thread.daemon = True
#                 self.bluetooth_thread.start()
#             if self.connection_attempts == self.valid_connection_attempts:
#                 if self.bluetooth_thread is not None:
#                     self.bluetooth_thread.join(1)
#                 self.bluetooth_thread = None
#                 # this should eventualy get to the data generator function
#                 self.data_collection_flag = True
#                 self.debugging = True
#                 text = f'{time.strftime("%H:%M:%S")} - Could not connect to {self.device_name}, starting random generator mode'
#                 print(text)
#                 self.bluetooth_thread = threading.Thread(
#                     target=self.__debugging_data_nirs_generator)
#                 self.bluetooth_thread.daemon = True
#                 self.bluetooth_thread.start()
#             time.sleep(0.5)
#         while self.reading:
#             while not self.debugging and not self.data_queue.empty():
#                 data = str(self.data_queue.get())
#                 self.__data_generator_que_manager(
#                     data)
#             time.sleep(0.005)
#
#         if self.bluetooth_thread is None:
#             return
#         self.bluetooth_thread.join(1)
#
#     def __data_generator_que_manager(self, data):
#         self.downsampling_counter -= 1
#         viz_put = False
#         if self.downsampling_counter == 0:
#             self.downsampling_counter = self.downsampling
#             viz_put = True
#         try:
#             self.db_queque.put(data)
#             if viz_put:
#                 self.viz_queque.put(data)
#         except queue.Full:
#             self.db_queque.get()
#             self.db_queque.put(data)
#
#     def __debugging_generate_smo2_profile(self, noise_level=0.4):
#         """Generates synthetic data that mimics a force profile on a hangboard."""
#         n_samples = int(np.random.uniform(300, 800))
#         t = np.linspace(0, np.pi, n_samples)
#         max_force = np.random.uniform(30, 80)
#         force_profile = max_force * np.sin(t)
#         noise = np.random.normal(0, noise_level, n_samples)
#         force_profile = np.clip(force_profile + noise, 20, max_force)
#         return force_profile
#
#     def __debugging_data_nirs_generator(self):
#         while self.reading:
#             timestamp = time.time()
#             if len(self.debugging_data) <= 1:
#                 self.debugging_data = self.__debugging_generate_smo2_profile(
#                     noise_level=np.random.uniform(0.2, 0.6)).tolist()
#                 zeros_at_end = np.random.uniform(
#                     self.debugging_data[-1], self.debugging_data[-1]*1.02, int(np.random.uniform(30, 400))).tolist()
#                 self.debugging_data.extend(zeros_at_end)
#             serial_data = self.debugging_data.pop(0)
#             data = f'{FORCE_SENSOR_ID}_{timestamp:.3f}_{serial_data}'
#             self.__data_generator_que_manager(data)
#             time.sleep(0.005)
#
#     def start_bluetooth_collector(self):
#         if self.data_collection_thread is not None:
#             return
#         self.__default_variables()
#         self.data_collection_thread = threading.Thread(
#             target=self.__start_asynchronous_thread)
#         self.data_collection_thread.daemon = True
#         self.data_collection_thread.start()
#         return
#
#     def stop_bluetooth_collector(self):
#         if self.data_collection_thread is not None:
#             self.data_collection_flag = True
#             self.reading = False
#             self.data_collection_thread.join(timeout=2)
#             self.data_collection_thread = None
#             if self.debugging:
#                 print(f'{time.strftime("%H:%M:%S")} - Random Data Generator Stopped')
#             else:
#                 text = f'{time.strftime("%H:%M:%S")} - Bluetooth communicator with {self.device_name} sucesfully closed'
#                 print(text)
#             return 1
#         else:
#             return


class SerialCommunicator:
    """
    Class to facilitate communication with serial interface for data collection.

    This class manages the connection to a serial device, handles data retrieval
    and command writing, and allows for asynchronous data collection using threads.

    Attributes:

        - db_queque (queue.Queue): Queue for storing raw data from the serial interface.
        - visualization_que (queue.Queue): Queue for downsampled data intended for visualization.
        - baudrate (int): Baud rate for the serial communication (default is 19200).
        - data_viz_downsampling: (int): Factor for downsampling data for visualization.
        - show_raw (bool): Attribute to set whether calibration profile is applied to data or none

    Methods:
        - send_serial_command(command: str):
            - Sends a command to the serial device.

        - start_serial_collection():
            - Starts the data collection process in a separate thread.

        - stop_serial_collection():
            - Stops the data collection and terminates the serial connection.

        - update_calibration_parameters(slope, intercept):
            - Sets "show_raw" attribute to False, and input calibration profile


    Exceptions:
        If no serial devices are connected, class enters automatically into debugging mode (generating data with 100hz frequency).
    """

    def __init__(self, visualization_que: queue.Queue, database_que: queue.Queue, baudrate: int = 19200, data_viz_downsampling: int = 5) -> None:
        self.db_queque = database_que
        self.viz_queque = visualization_que
        self.baudrate = baudrate
        self.downsampling = data_viz_downsampling
        self.downsampling_counter = data_viz_downsampling
        self.port = None
        self.data_collection_thread = None

    def __default_variables(self):
        self.connection = False
        self.sercon = None
        self.collect_data = False
        self.writing_command = False
        self.data_collection_thread = None
        self.show_raw = True
        self.calibration_slope = 0
        self.calibration_intercept = 0
        self.command = None
        self.reset_timer = None
        self.debugging = False
        self.debugging_data = []

    def set_port(self, port: str):
        self.port = port

    def __serial_get_force_data(self) -> str:
        if self.port is None:
            return
        if self.connection == False:
            self.sercon = self.__serial_start_connection()
        sercon = self.sercon
        try:
            if sercon.in_waiting > 0 and self.writing_command == False:
                serial_data = sercon.readline().decode('utf-8').rstrip()
                serial_data = int(serial_data)
                if self.show_raw == False and self.calibration_intercept != 0:
                    serial_data = (
                        serial_data-self.calibration_intercept)/self.calibration_slope
                return f'{FORCE_SENSOR_ID}_{time.time():.3f}_{serial_data}'
            elif self.writing_command and self.command is not None:
                sercon.write(self.command.encode())
                self.command = None
                self.writing_command = False
            elif self.writing_command and self.command is None:
                self.writing_command = False
        except ValueError:
            pass
        except serial.serialutil.SerialException:
            self.collect_data = False
            self.port = None

    def __serial_start_connection(self) -> serial.Serial:
        if self.connection:
            return
        connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1
        )
        self.connection = True
        return connection

    def __serial_terminate_connection(self):
        connection = self.sercon
        port = connection.port
        connection.close()
        print(f'{time.strftime("%H:%M:%S")} - Connection {port} closed')
        return 1

    def __debugging_generate_force_profile(self, noise_level=0.4):
        """Generates synthetic data that mimics a force profile on a hangboard."""
        n_samples = int(np.random.uniform(300, 800))
        t = np.linspace(0, np.pi, n_samples)
        max_force = np.random.uniform(1, 60)
        force_profile = max_force * np.sin(t)
        noise = np.random.normal(0, noise_level, n_samples)
        force_profile = np.clip(force_profile + noise, 0, max_force)
        return force_profile

    def __debugging_data_force_generator(self):
        while self.collect_data:
            if len(self.debugging_data) <= 1:
                self.debugging_data = self.__debugging_generate_force_profile(
                    noise_level=np.random.uniform(0.2, 0.6)).tolist()
                zeros_at_end = np.random.uniform(
                    0, 0.8, int(np.random.uniform(100, 1000))).tolist()
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
                continue
            elif self.reset_timer is None or self.reset_timer + 10 > time.time():
                self.reset_timer = time.time()
            elif self.reset_timer + 10 < time.time():
                self.connection = False
                self.sercon = None
                self.reset_timer = None
                while not self.viz_queque.empty():
                    self.viz_queque.get()
            time.sleep(0.025)

    def start_serial_collection(self):
        self.__default_variables()
        if self.port is None:
            self.debugging = True
        if self.data_collection_thread is not None:
            return
        if not self.debugging:
            self.data_collection_thread = threading.Thread(
                target=self.__data_generator)
        elif self.debugging:
            self.data_collection_thread = threading.Thread(
                target=self.__debugging_data_force_generator)
        self.collect_data = True
        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()
        return

    def stop_serial_collection(self):
        if self.data_collection_thread is not None:
            self.collect_data = False
            self.data_collection_thread.join()
            self.data_collection_thread = None
            if not self.debugging:
                self.__serial_terminate_connection()
            self.connection = False
            return 1
        else:
            return

    def send_serial_command(self, command: str):
        self.writing_command = True
        self.command = command
        return 1

    def update_calibration_parameters(self, slope: float, intercept: float):
        self.show_raw = False
        self.calibration_intercept = intercept
        self.calibration_slope = slope


class BluetoothCommunicator:
    """
    A class for managing Bluetooth Low Energy (BLE) communication with a specified device. This class connects to a BLE
    device, starts receiving notifications, and processes data for visualization and database logging. If the connection
    fails, it switches to a debug mode, generating synthetic data.

    Attributes:
        db_queque (queue.Queue): Queue for sending data to the database.
        viz_queque (queue.Queue): Queue for sending data to the visualization.
        downsampling (int): Interval at which data is downsampled for visualization purposes.
        device_name (str): The name of the target BLE device.
        device_uuid (str): The UUID of the target BLE device for data notifications.
        reading (bool): Flag to indicate if the device is actively reading data.
        debugging (bool): Indicates if the class is generating debug data instead of Bluetooth data.
        data_collection_flag (bool): Indicates if data collection has started.
        start_reading_time (float): Timestamp for the start of the reading process.
        connection_attempts (int): Counts connection attempts made to the BLE device.
        valid_connection_attempts (int): Maximum number of valid connection attempts before switching to debug mode.
        bluetooth_thread (threading.Thread): Thread running the Bluetooth read operations.
        data_collection_thread (threading.Thread): Thread to manage data collection for Bluetooth or debug mode.

    Methods:

        - start_bluetooth_collector(): Initializes and starts data collection from the BLE device.
        - stop_bluetooth_collector(): Stops data collection from the BLE device or debug mode.
    """

    def __init__(self, visualization_que: queue.Queue, database_que: queue.Queue, data_viz_downsampling: int = 1) -> None:
        # config = JSONCommunicator().get_ble_nirs_config()
        self.device_name = ""  # config['dev_name']
        self.device_uuid = ""  # config['dev_data_uuid']
        self.downsampling = data_viz_downsampling
        self.downsampling_counter = data_viz_downsampling
        self.db_queque = database_que
        self.viz_queque = visualization_que
        self.valid_connection_attempts = 1  # config['connection_attempts']
        self.__default_variables()

    def __default_variables(self):
        self.client = None
        self.data_queue = queue.Queue()
        self.reading = False
        self.bluetooth_thread = None
        self.start_reading_time = None
        self.data_collection_flag = False
        self.connection_attempts = 0
        self.data_collection_thread = None
        self.debugging = False
        self.debugging_data = []
        self.poll_battery_time = time.time()
        self.battery_time = 30
        self.last_datapoint = None

    async def __async_connect(self):
        """Connects to the BLE device and ensures services are discovered."""
        devices = await BleakScanner.discover()
        for device in devices:
            if device.name == self.device_name:
                self.client = BleakClient(device)
                await self.client.connect()
                print(f'{time.strftime("%H:%M:%S")} - Connected to {self.device_name}')
                return 1
        return

    async def __async_start_notifications(self):
        """Starts receiving notifications from the Bluetooth device."""
        if self.client and self.client.is_connected:
            await self.client.start_notify(self.device_uuid, self.__async_notification_handler)
            self.last_datapoint = time.time()
            print(f'{time.strftime("%H:%M:%S")} - Started receiving data from {self.device_name}')

    async def __async_stop_notifications(self):
        if self.client and self.client.is_connected:
            await self.client.stop_notify(self.device_uuid)
            print(f'{time.strftime("%H:%M:%S")} - Stopped receiving data from {self.device_name}')

    def __process_data(self, data):
        """Unpacks the data according to the expected format."""

        timestamp = time.time()
        format_string = '<fffB I'
        values = struct.unpack(format_string, data)
        # format 2_time_SmO2_O2Hb_HHb
        self.last_datapoint = timestamp
        formated_vals = f'2_{timestamp}_{values[0]}_{values[1]}_{values[2]}'
        return formated_vals

    async def __async_notification_handler(self, sender, data):
        try:
            data_str = self.__process_data(data)
            self.data_queue.put(data_str)  # Send data to the queue
            time.sleep(0.05)
            if self.data_collection_flag == False:
                self.data_collection_flag = True
            if time.time() > self.poll_battery_time:
                await self.__poll_battery_level()
                self.poll_battery_time = time.time()+30
        except Exception as e:
            print(
                f'{time.strftime("%H:%M:%S")} - BLECOM __Async not. handler: {data} - {e}')

    async def __read_battery_level(self):
        """Reads the battery level from the device."""
        if self.client and self.client.is_connected:
            try:
                battery_data = await self.client.read_gatt_char()  # JSONCommunicator().get_ble_nirs_config()['dev_battery_uuid'])
                battery_level = struct.unpack("B", battery_data)[
                    0]  # Unpack as an unsigned byte
                battery_data_str = f'b_{time.time()}_{battery_level}'
                self.viz_queque.put(battery_data_str)
            except Exception as e:
                print(f'{time.strftime("%H:%M:%S")} - Failed to read battery level: {e}')

    async def __poll_battery_level(self):
        """Periodically reads battery level at the specified interval (in seconds)."""
        if self.client and self.client.is_connected:
            await self.__read_battery_level()
            return

    def __run(self):
        """Runs the asynchronous Bluetooth tasks in the thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.__async_connect())
        try:
            loop.run_until_complete(self.__async_start_notifications())
        except Exception as e:
            print(f'{time.strftime("%H:%M:%S")} - BLECOM.__Run: {e}')
        try:
            while self.reading:
                loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            if self.client is not None:
                try:
                    loop.run_until_complete(self.__async_stop_notifications())
                    loop.run_until_complete(self.client.disconnect())
                    print(f'{time.strftime("%H:%M:%S")} - Disconnected from {self.device_name}')
                except Exception as e:
                    print(f'{time.strftime("%H:%M:%S")} - BLECOM.__run: {e}')

    def __start_asynchronous_thread(self):
        """Starts the Bluetooth communicator in a separate thread."""
        self.reading = True
        self.start_reading_time = time.time()
        self.bluetooth_thread = threading.Thread(target=self.__run)
        self.bluetooth_thread.daemon = True
        self.bluetooth_thread.start()
        while self.data_collection_flag == False:
            if self.start_reading_time+15 < time.time():
                self.start_reading_time = time.time()
                self.connection_attempts += 1
                print(f'{time.strftime("%H:%M:%S")} - Could not Connect to  {self.device_name}. '
                      f'Reconnecting ({self.connection_attempts}/{self.valid_connection_attempts})')
                self.bluetooth_thread.join(timeout=1)
                self.bluetooth_thread = threading.Thread(target=self.__run)
                self.bluetooth_thread.daemon = True
                self.bluetooth_thread.start()
            if self.connection_attempts == self.valid_connection_attempts:
                if self.bluetooth_thread is not None:
                    self.bluetooth_thread.join(1)
                self.bluetooth_thread = None
                # this should eventualy get to the data generator function
                self.data_collection_flag = True
                self.debugging = True
                text = f'{time.strftime("%H:%M:%S")} - Could not connect to {self.device_name}, starting random generator mode'
                print(text)
                self.bluetooth_thread = threading.Thread(
                    target=self.__debugging_data_nirs_generator)
                self.bluetooth_thread.daemon = True
                self.bluetooth_thread.start()
            time.sleep(0.5)
        while self.reading == True:
            if not self.debugging and not self.data_queue.empty():
                data = str(self.data_queue.get())
                _, timestamp, smo2, o2hb, hhb = data.split('_')
                smo2text = f'{SMO2_SENSOR_ID}_{timestamp}_{smo2}'
                o2hbtext = f'{O2HB_SENSOR_ID}_{timestamp}_{o2hb}'
                hhbtext = f'{HHB_SENSOR_ID}_{timestamp}_{hhb}'
                self.__data_generator_que_manager(
                    [o2hbtext, smo2text, hhbtext])
            time.sleep(0.05)

        if self.bluetooth_thread is None:
            return
        self.bluetooth_thread.join(1)

    def __data_generator_que_manager(self, data):
        self.downsampling_counter -= 1
        viz_put = False
        if self.downsampling_counter == 0:
            self.downsampling_counter = self.downsampling
            viz_put = True

        for i in data:
            try:
                self.db_queque.put(i)
                if viz_put:
                    self.viz_queque.put(i)
            except queue.Full:
                self.db_queque.get()
                self.db_queque.put(i)

    def __debugging_generate_smo2_profile(self, noise_level=0.4):
        """Generates synthetic data that mimics a force profile on a hangboard."""
        n_samples = int(np.random.uniform(300, 800))
        t = np.linspace(0, np.pi, n_samples)
        max_force = np.random.uniform(30, 80)
        force_profile = max_force * np.sin(t)
        noise = np.random.normal(0, noise_level, n_samples)
        force_profile = np.clip(force_profile + noise, 20, max_force)
        return force_profile

    def __debugging_data_nirs_generator(self):
        while self.reading:
            timestamp = time.time()
            if len(self.debugging_data) <= 1:
                self.debugging_data = self.__debugging_generate_smo2_profile(
                    noise_level=np.random.uniform(0.2, 0.6)).tolist()
                zeros_at_end = np.random.uniform(
                    self.debugging_data[-1], self.debugging_data[-1]*1.02, int(np.random.uniform(50, 100))).tolist()
                self.debugging_data.extend(zeros_at_end)
            serial_data = self.debugging_data.pop(0)
            data = [f'{SMO2_SENSOR_ID}_{timestamp:.3f}_{serial_data}',
                    f'{HHB_SENSOR_ID}_{timestamp:.3f}_{np.random.randint(50, 100)}',
                    f'{O2HB_SENSOR_ID}_{timestamp:.3f}_{np.random.randint(120, 150)}']
            self.__data_generator_que_manager(data)
            if self.poll_battery_time < timestamp:
                self.poll_battery_time = timestamp+self.battery_time
                battery_level = np.random.randint(30, 70)
                self.__data_generator_que_manager(
                    [f'b_{timestamp}_{battery_level}'])
            time.sleep(0.1)

    def start_bluetooth_collector(self):
        if self.data_collection_thread is not None:
            return
        self.__default_variables()
        self.data_collection_thread = threading.Thread(
            target=self.__start_asynchronous_thread)
        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()
        return

    def stop_bluetooth_collector(self):
        if self.data_collection_thread is not None:
            self.data_collection_flag = True
            self.reading = False
            self.data_collection_thread.join(timeout=2)
            self.data_collection_thread = None
            if self.debugging:
                print(f'{time.strftime("%H:%M:%S")} - Random Data Generator Stopped')
            else:
                text = f'{time.strftime("%H:%M:%S")} - Bluetooth communicator with {self.device_name} sucesfully closed'
                print(text)
            return 1
        else:
            return


class DataGenerator:
    """
    DataGenerator manages data collection from two sources: NIRS via Bluetooth and another 
    source via Serial communication. This class initializes and coordinates data queues, 
    Bluetooth and Serial communicators, and provides control functions to start, stop, 
    and debug data collection processes.

    Attributes:
        db_queque (queue.Queue): Queue to store data for database storage.
        viz_queque (queue.Queue): Queue to store data for visualization.
        bluetooth_communicator (BluetoothCommunicator): Handles data collection from 
            NIRS via Bluetooth.
        serial_communicator (SerialCommunicator): Manages data collection over Serial 
            communication.

    Methods:
        start_collection(): Begins data collection from both Bluetooth and Serial sources.
        stop_collection(): Halts data collection from both Bluetooth and Serial sources.
        stop_if_debugging(): Stops collections if debugging mode is enabled for any source.
        send_serial_command(command): Sends a command over Serial communication.
        update_calibration_parameters(slope, intercept): Updates calibration parameters 
            for data collection.
    """

    def __init__(self, baudrate: int = 19200, force_viz_downsampling: int = 5, nirs_viz_downsampling: int = 2) -> None:
        self.db_queque = queue.Queue(maxsize=850)
        self.viz_queque = queue.Queue()
        self.bluetooth_communicator = BluetoothCommunicator(
            visualization_que=self.viz_queque, database_que=self.db_queque, data_viz_downsampling=nirs_viz_downsampling)
        self.serial_communicator = SerialCommunicator(
            self.viz_queque, database_que=self.db_queque, baudrate=baudrate, data_viz_downsampling=force_viz_downsampling)
        # self.ble_force_communicator = ForceBluetoothCommunicator(
        #     visualization_que=self.viz_queque, database_que=self.db_queque)
        self.serial_force_def = True  # bool(JSONCommunicator().get_serial_force_config()['default'])
        self.nirs_active = False

    # def _force_start_collection(self):
    #     self.serial_communicator.start_serial_collection()
    #     if not self.serial_force_def:
    #         self.serial_communicator.stop_serial_collection()
    #         self.ble_force_communicator.start_bluetooth_collector()
    #
    # def _force_stop_collection(self):
    #     self.serial_communicator.stop_serial_collection()
    #     if not self.serial_force_def:
    #         self.ble_force_communicator.stop_bluetooth_collector()

    def _nirs_start_collection(self):
        self.bluetooth_communicator.start_bluetooth_collector()

    def _nirs_stop_collection(self):
        self.bluetooth_communicator.stop_bluetooth_collector()
        self.bluetooth_communicator.connection_attempts = 0

    def _force_check_if_debugging(self) -> bool | str:
        """
        Possible Outcomes: 
            - True: if debugging is active
            - False: if debugging is not active
            - str('tba'): if it is not possible to asses whether debugging is active
            - str('off'): if sensor is not used
        """
        if self.serial_force_def:
            if self.serial_communicator.debugging:
                return True
            if self.serial_communicator.debugging == False:
                if self.serial_communicator.collect_data == False and self.serial_communicator.data_collection_thread is not None:
                    return 'off'
                return False
            return 'off'
        else:
            if self.ble_force_communicator.debugging:
                return True
            if self.ble_force_communicator.debugging == False:
                if self.ble_force_communicator.data_collection_thread is None:
                    return 'off'
                else:
                    if self.ble_force_communicator.data_collection_flag == False:
                        return 'tba'
                if self.ble_force_communicator.last_datapoint is None:
                    return 'tba'
                elif time.time() - self.ble_force_communicator.last_datapoint > 1.5:
                    return 'off'
                return False

    def _nirs_check_if_debugging(self) -> bool | str:
        """
        Possible Outcomes: 
            - True: if debugging is active
            - False: if debugging is not active
            - str('tba'): if it is not possible to asses whether debugging is active
            - str('off'): if sensor is not used
        """
        if self.bluetooth_communicator.debugging:
            return True
        if self.bluetooth_communicator.debugging == False:
            if self.bluetooth_communicator.data_collection_thread is None:
                return 'off'
            else:
                if self.bluetooth_communicator.data_collection_flag == False:
                    return 'tba'
            if self.bluetooth_communicator.last_datapoint is None:
                return 'tba'
            elif time.time() - self.bluetooth_communicator.last_datapoint > 1.5:
                return 'off'
            return False

    # public functions start here
    def set_force_port(self, port):
        self.serial_communicator.set_port(port)

    def start_collection(self):
        if self.nirs_active:
            self._nirs_start_collection()
        self._force_start_collection()

    def stop_collection(self):
        if self.bluetooth_communicator.data_collection_flag or self.bluetooth_communicator.reading:
            self._nirs_stop_collection()
        self._force_stop_collection()

    def stop_if_debugging(self):
        """
        Stops data collection processes if debugging mode is active for either communicator.

        Returns:
            str or None: Message indicating which collection process was stopped, or None 
            if neither was stopped.
        """
        text = ''
        if self._nirs_check_if_debugging():
            self._nirs_stop_collection()
            text = text + f'NIRS Collector stopped\n'
        if self._force_check_if_debugging():
            self._force_stop_collection
            text = text + f'Force Collector stopped\n'
        return text if len(text) > 0 else None

    # misc function start here
    def send_serial_command(self, command: str):
        """
        Sends a command over the Serial communicator.

        Args:
            command (str): The command string to send via Serial.

        Returns:
            str: Message indicating success or failure of command transmission.
        """
        if not self.serial_force_def:
            return 'Serial force sensor not in use'
        text = self.serial_communicator.send_serial_command(command)
        if text == 1:
            text = 'Command sucesfully sent'
        else:
            text = 'Could not send command'
        return text

    def update_calibration_parameters(self, slope: float, intercept: float) -> None:
        """
        Updates calibration parameters for the Serial communicator.

        Args:
            slope (float): The slope for calibration adjustment.
            intercept (float): The intercept for calibration adjustment.
        """
        if self.serial_force_def:
            self.serial_communicator.update_calibration_parameters(
                slope=slope, intercept=intercept)
        # else:
        #     self.ble_force_communicator.update_calibration_parameters(
        #         slope=slope, intercept=intercept)

    def show_raw_force(self, raw: bool) -> None:
        if self.serial_force_def:
            self.serial_communicator.show_raw = raw
        # else:
        #     self.ble_force_communicator.show_raw = raw


if __name__ == '__main__':
    db_queque = queue.Queue()
    dg = BluetoothCommunicator(queue.Queue(), db_queque)
    dg.start_bluetooth_collector()
    for i in range(500):
        time.sleep(0.1)
        print(db_queque.get())
    dg.stop_bluetooth_collector()
