"""
Data generation and acquisition module for the Climbing Testing Application.

This module defines classes for sensor communication and data acquisition, 
providing interfaces for both real hardware connections and simulated data.
It manages force sensors through serial connections and NIRS devices through
Bluetooth Low Energy (BLE) connections, with fallback to simulation when
hardware is unavailable.

Key functionalities:
- Serial communication for force sensors
- Bluetooth communication for NIRS devices
- Data calibration and preprocessing
- Simulated data generation for testing
- Thread management for concurrent data collection
- Data buffering and callback management

The module serves as the hardware abstraction layer, handling all sensor 
interactions and providing consistent data access for the application.
"""

import serial
import serial.serialutil
import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
import threading
import struct
import asyncio


class SerialCommunicator:
    """
    Handles serial communication with force sensors.
    
    This class manages the connection to force sensors via serial ports,
    reads and processes incoming data, applies calibration, and provides
    the option to generate synthetic data when hardware is unavailable.
    
    Attributes:
        data_callback: Function to call with new force data
        port: Serial port identifier
        baudrate: Baud rate for serial communication
        running: Flag indicating if the communicator is active
        thread: Thread for serial reading loop
        show_raw: Flag to show raw (uncalibrated) values
        calibration_slope: Slope for linear calibration
        calibration_intercept: Intercept for linear calibration
    """
    def __init__(self, data_callback, port: str = None, baudrate: int = 19200):
        """
        Initialize the serial communicator.
        
        Args:
            data_callback: Function to call with each new data point (format: "1_timestamp_value")
            port: Serial port to connect to (if None, uses synthetic data)
            baudrate: Baud rate for serial connection
        """
        self.data_callback = data_callback
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.thread = None
        self.show_raw = False
        self.calibration_slope = 1.0
        self.calibration_intercept = 0.0

    def set_port(self, port: str):
        """
        Set the serial port to use.
        
        Args:
            port: Serial port identifier
            
        This method sets the port to connect to for force sensor data.
        The actual connection is established when start() is called.
        """
        self.port = port

    def update_calibration(self, slope: float, intercept: float):
        """
        Update the calibration parameters for force conversion.
        
        Args:
            slope: Calibration slope
            intercept: Calibration intercept
            
        These parameters are used in a linear transformation to convert 
        raw sensor values to calibrated force measurements (typically in kg).
        """
        self.calibration_slope = slope
        self.calibration_intercept = intercept

    def _read_force(self):
        """
        Read loop for force data acquisition.
        
        This method runs in a separate thread and either reads from a real 
        serial port or generates synthetic data. It calls the data_callback
        with each new data point in the format "1_timestamp_value".
        """
        if self.port:
            try:
                ser = serial.Serial(self.port, self.baudrate, timeout=1)
                try:
                    while self.running:
                        if ser.in_waiting:
                            try:
                                raw = ser.readline().decode('utf-8').strip()
                                try:
                                    val = int(raw)
                                    if not self.show_raw:
                                        val = (val - self.calibration_intercept) / self.calibration_slope
                                    timestamp = time.time()
                                    # format: "sensorId_timestamp_value"
                                    self.data_callback(f"1_{timestamp:.3f}_{val}")
                                except ValueError:
                                    print(f"Invalid data received: {raw}")
                            except UnicodeDecodeError:
                                print("Could not decode data from serial port")
                        time.sleep(0.01)
                finally:
                    ser.close()
            except serial.SerialException as e:
                print(f"Serial connection error: {e}")
                print("Falling back to synthetic data generation")
                self._generate_synthetic_data()
        else:
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """
        Generate synthetic force data when no serial connection is available.
        
        Creates a simulated force profile with realistic characteristics and
        calls the data_callback with the generated values at regular intervals.
        """
        while self.running:
            try:
                # Generate sinusoidal profile with random amplitude
                profile = np.sin(np.linspace(0, np.pi, 500)) * np.random.uniform(40,60)
                for v in profile:
                    if not self.running:
                        break
                    t = time.time()
                    print('FORCE data:', f"1_{t:.3f}_{v:.2f}")
                    self.data_callback(f"1_{t:.3f}_{v:.2f}")
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in synthetic data generation: {e}")
                time.sleep(1)  # Prevent rapid error loops

    def start(self):
        """
        Start the force data acquisition thread.
        
        Creates and starts a daemon thread for reading force data either
        from the serial port or generating synthetic data.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_force, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the force data acquisition thread.
        
        Sets the running flag to False and waits for the thread to finish.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


class BluetoothCommunicator:
    """
    Handles BLE communication with NIRS sensors.
    
    This class manages Bluetooth Low Energy connections to NIRS devices,
    processes incoming notification data, and provides simulated data when
    hardware connections fail or are unavailable.
    
    Attributes:
        data_callback: Function to call with new NIRS data
        device_name: Name of the BLE device to connect to
        data_uuid: UUID of the data characteristic to subscribe to
        client: BLE client instance
        connection_state: Current connection state message
        running: Flag indicating if the communication thread is running
        connected: Flag indicating actual BLE connection state
        thread: Thread for BLE operations
        debugging: Flag indicating if in simulation mode
        debug_data_buffer: Buffer for simulated data
        connection_attempts: Counter for connection attempts
        max_attempts: Maximum number of connection attempts before fallback
    """
    def __init__(self, data_callback, device_name: str, data_uuid: str, max_attempts: int = 3):
        """
        Initialize the Bluetooth communicator.
        
        Args:
            data_callback: Function to call with each new data set
            device_name: Name of the BLE device to connect to
            data_uuid: UUID of the data characteristic to subscribe to
            max_attempts: Maximum number of connection attempts before falling back to simulation
        """
        self.data_callback = data_callback
        self.device_name = device_name
        self.data_uuid = data_uuid
        self.client = None
        self.connection_state = ''
        self.running = False  # Whether the communication thread is running
        self.connected = False  # Actual BLE connection state
        self.thread = None
        self.debugging = False
        self.debug_data_buffer = []
        self.connection_attempts = 0
        self.max_attempts = max_attempts
        
    @staticmethod
    def _generate_nirs_profile(noise_level=0.4):
        """
        Generate a synthetic NIRS data profile for simulation.
        
        Args:
            noise_level: Amplitude of random noise to add
            
        Returns:
            list: List of simulated NIRS data points
            
        Creates a realistic NIRS profile with a sinusoidal baseline and added noise,
        similar to what would be seen during actual muscle oxygenation measurements.
        """
        n_samples = int(np.random.uniform(300, 800))
        t = np.linspace(0, np.pi, n_samples)
        base = np.sin(t) * np.random.uniform(30, 80)
        noise = np.random.normal(0, noise_level, n_samples)
        return np.clip(base + noise, 20, np.max(base)).tolist()

    def _simulate_nirs_data_generator(self):
        """
        Generate simulated NIRS data and call the callback function.
        
        This method runs in a loop when BLE connection fails, generating
        realistic NIRS data for testing and development purposes.
        Generates data for three NIRS channels: SmO2 (ID 2), HHb (ID 3), and O2Hb (ID 4).
        """
        while self.running:
            try:
                if len(self.debug_data_buffer) <= 1:
                    level = np.random.uniform(0.2, 0.6)
                    self.debug_data_buffer = self._generate_nirs_profile(level)
                v = self.debug_data_buffer.pop(0)
                ts = time.time()
                smo2 = v
                hhb = np.random.randint(50, 100)
                o2hb = np.random.randint(120, 150)
                items = [f"2_{ts:.3f}_{smo2}", f"3_{ts:.3f}_{hhb}", f"4_{ts:.3f}_{o2hb}"]
                print('NIRS data:', items)
                self.data_callback(items)
                time.sleep(0.1)
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} - NIRS simulation error: {e}")
                time.sleep(1)  # Prevent rapid error loops

    async def _connect_and_listen(self):
        """
        Scan for and connect to the BLE device, then start notification handling.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        This async method scans for BLE devices, connects to the specified device,
        and sets up notification handling for the NIRS data characteristic.
        """
        try:
            from bleak import BleakScanner, BleakClient
            devices = await BleakScanner.discover()
            for d in devices:
                if d.name == self.device_name:
                    self.client = BleakClient(d)
                    await self.client.connect()
                    await self.client.start_notify(self.data_uuid, self._handler)
                    return True
            return False
        except Exception as e:
            print(f"Error during BLE connection: {e}")
            return False

    async def _stop_and_disconnect(self):
        """
        Stop notifications and disconnect from the BLE device.
        
        This async method properly cleans up the BLE connection by stopping
        notifications and disconnecting from the device.
        """
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(self.data_uuid)
                await self.client.disconnect()
            except Exception as e:
                print(f"Error during BLE disconnection: {e}")

    def _handler(self, _sender, data: bytes):
        """
        Handle incoming BLE notification data.
        
        Args:
            _sender: BLE sender object (unused)
            data: Raw bytes from BLE notification
            
        Processes raw BLE data packets, unpacks the binary data into
        floating point values, and calls the data callback with the
        processed values for each NIRS channel.
        """
        try:
            ts = time.time()
            # Unpack just the first three floats (4 bytes each)
            values = struct.unpack('<fff', data[:12])
            f1, f2, f3 = values
            items = [f"2_{ts:.3f}_{f1}", f"3_{ts:.3f}_{f2}", f"4_{ts:.3f}_{f3}"]
            print('NIRS data:', items)
            self.data_callback(items)
            
            # Explicitly set connected flag when we successfully receive data
            if not self.connected:
                self.connected = True
        except struct.error as e:
            print(f"{time.strftime('%H:%M:%S')} - NIRS handler error: {e} - Data length: {len(data)}")
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} - NIRS handler error: {e}")
            
    def _run_loop(self):
        """
        Main thread loop for BLE operations.
        
        This method manages the BLE connection lifecycle, including:
        - Setting up the asyncio event loop
        - Attempting to connect with retries
        - Monitoring the connection state
        - Falling back to simulation mode if connection fails
        - Properly disconnecting when stopped
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Attempt to connect with retries
            self.connected = False
            for attempt in range(1, self.max_attempts + 1):
                try:
                    self.connected = loop.run_until_complete(self._connect_and_listen())
                except Exception as e:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - NIRS connect attempt {attempt}/{self.max_attempts} error: {e}"
                    self.connected = False
                if self.connected:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - Connected to {self.device_name} on attempt {attempt}"
                    break
                else:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - Could not connect to {self.device_name}. Reconnecting ({attempt}/{self.max_attempts})"
                    time.sleep(1)

            if not self.connected:
                self.debugging = True
                self.connection_state = f"{time.strftime('%H:%M:%S')} - Failed to connect after {self.max_attempts} attempts. Starting simulation mode"
                self._simulate_nirs_data_generator()
                return

            # Connected: listen for data or until stopped
            try:
                while self.running:
                    loop.run_until_complete(asyncio.sleep(0.1))
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} - BLE loop error: {e}")
            finally:
                try:
                    loop.run_until_complete(self._stop_and_disconnect())
                    self.connected = False
                except Exception as e:
                    print(f"{time.strftime('%H:%M:%S')} - Error on disconnect: {e}")
        except Exception as e:
            print(f"Critical error in BLE thread: {e}")
            if self.running:
                print("Falling back to simulation mode")
                self._simulate_nirs_data_generator()

    def start(self):
        """
        Start the BLE communication thread.
        
        Creates and starts a daemon thread for managing BLE connections
        and processing NIRS data.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the BLE communication thread.
        
        Sets the running flag to False and waits for the thread to finish.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


class DataGenerator:
    """
    Coordinates both serial and BLE communicators, providing a unified interface
    for data acquisition from force sensors and NIRS devices.
    
    This class manages the lifecycle of sensor connections, buffers incoming data,
    handles callbacks, and provides methods to access the collected data.
    
    Attributes:
        force_data: List of collected force data points
        nirs_data: List of collected NIRS data points
        pause_callbacks: Flag to control if data should be passed to external callbacks
        external_force_callback: External callback for force data
        external_nirs_callback: External callback for NIRS data
        serial_com: SerialCommunicator instance for force data
        bluetooth_com: BluetoothCommunicator instance for NIRS data
    """
    def __init__(self):
        """
        Initialize the DataGenerator with communicators for force and NIRS data.
        
        Sets up data storage, callback management, and creates communicator instances
        with appropriate internal callbacks.
        """
        # Initialize internal data storage
        self.force_data = []
        self.nirs_data = []
        
        # Flag to control if data should be passed to external callbacks
        self.pause_callbacks = False
        self.external_force_callback = None
        self.external_nirs_callback = None
        
        # Initialize internal callbacks first
        # These must exist before creating the communicators
        self._original_force_callback = None
        self._original_nirs_callback = None
        
        # Create communicators with internal callbacks
        self.serial_com = SerialCommunicator(data_callback=self._on_force, port='/dev/tty.usbserial-14130')
        self.bluetooth_com = BluetoothCommunicator(
            data_callback=self._on_nirs,
            device_name="Train.Red FYER 0707",
            data_uuid="00002201-D578-4741-9B5B-7C64E958CFC6",
            max_attempts=2
        )

    def _on_force(self, data_str: str):
        """
        Internal callback for force data processing.
        
        Args:
            data_str: String containing force data in format "sensorID_timestamp_value"
            
        Always stores data internally and forwards to external callback if set and not paused.
        """
        try:
            sid, ts, val = data_str.split('_')
            # Always store data internally
            self.force_data.append({
                'sensor_id': int(sid),
                'timestamp': float(ts),
                'value': float(val)
            })
            
            # Only forward if we have an external callback and are not paused
            if not self.pause_callbacks and self.external_force_callback:
                self.external_force_callback(data_str)
                
        except (ValueError, IndexError) as e:
            print(f"Error parsing force data ({data_str}): {e}")

    def _on_nirs(self, data_list: list):
        """
        Internal callback for NIRS data processing.
        
        Args:
            data_list: List of strings containing NIRS data
            
        Processes each NIRS data string, stores it internally, and forwards
        to external callback if set and not paused.
        """
        for data_str in data_list:
            try:
                sid, ts, val = data_str.split('_')
                # Always store data internally
                self.nirs_data.append({
                    'sensor_id': int(sid),
                    'timestamp': float(ts),
                    'value': float(val)
                })
                
                # Only forward if we have an external callback and are not paused
                if not self.pause_callbacks and self.external_nirs_callback:
                    self.external_nirs_callback(data_str)
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing NIRS data ({data_str}): {e}")
    
    def set_callbacks(self, force_callback=None, nirs_callback=None):
        """
        Set external callbacks for the data generator.
        
        Args:
            force_callback: Callback function for force data
            nirs_callback: Callback function for NIRS data
            
        Sets the external callback functions that will receive data from
        the respective sensors. Automatically enables data forwarding.
        """
        self.external_force_callback = force_callback
        self.external_nirs_callback = nirs_callback
        # Resume data forwarding by default when setting new callbacks
        self.pause_callbacks = False
    
    def pause_data_forwarding(self):
        """
        Pause forwarding data to external callbacks.
        
        Data will still be collected and stored internally, but will not be
        forwarded to external callbacks until resume_data_forwarding is called.
        """
        if not self.pause_callbacks:
            self.pause_callbacks = True

    def resume_data_forwarding(self):
        """
        Resume forwarding data to external callbacks.
        
        Enables the forwarding of collected data to external callbacks
        after it has been paused.
        """
        if self.pause_callbacks:
            self.pause_callbacks = False

    def is_nirs_connected(self):
        """
        Check if NIRS is connected and running.

        Returns:
            bool: True if NIRS connection is active (real, not simulated)
            
        For the connection to be considered active, the thread must be running,
        the device must be connected, and it must not be in simulation mode.
        """
        thread_running = self.bluetooth_com.running
        real_connection = self.bluetooth_com.connected
        not_simulating = not self.bluetooth_com.debugging
        
        connection_active = thread_running and real_connection and not_simulating
        
        return connection_active

    def set_force_port(self, port: str):
        """
        Set the serial port for force sensor.
        
        Args:
            port: Serial port identifier
            
        Raises:
            ValueError: If port is empty
        """
        if not port:
            raise ValueError("Serial port cannot be empty")
        self.serial_com.set_port(port)

    def start_nirs_connection(self):
        """
        Start the BLE connection to NIRS device.

        Returns:
            bool: True if the connection is successfully started
            
        Clears existing data and initiates the NIRS device connection.
        """
        if self.bluetooth_com.running:
            print("NIRS connection already running")
            return True

        # Clear any existing data before starting connection
        self.clear_data()

        # Start the bluetooth connection
        self.bluetooth_com.start()
        return True

    def update_calibration(self, slope: float, intercept: float):
        """
        Update force sensor calibration parameters.
        
        Args:
            slope: Calibration slope
            intercept: Calibration intercept
            
        Raises:
            ValueError: If slope is zero (would cause division by zero)
            
        Updates the calibration parameters used to convert raw force sensor
        values to calibrated force measurements.
        """
        if slope == 0:
            raise ValueError("Calibration slope cannot be zero (division by zero)")
        self.serial_com.update_calibration(slope, intercept)

    def start(self, nirs=True, force=True):
        """
        Start data acquisition from selected sensors.
        
        Args:
            nirs: Whether to start NIRS data acquisition
            force: Whether to start force data acquisition
            
        Starts the communication threads for the selected sensor types.
        """
        if nirs:
            self.bluetooth_com.start()
        if force:
            self.serial_com.start()

    def stop(self, nirs=True, force=True):
        """
        Stop data acquisition and ensure threads are properly terminated.
        
        Args:
            nirs: Whether to stop NIRS data acquisition
            force: Whether to stop force data acquisition
            
        Stops the communication threads for the selected sensor types,
        pauses callbacks, and verifies thread termination.
        """
        # First pause any callbacks to prevent UI updates during shutdown
        self.pause_callbacks = True
        
        # Stop each sensor communicator as requested
        if nirs:
            print("Stopping NIRS communication...")
            self.bluetooth_com.stop()
        
        if force:
            print("Stopping force sensor communication...")
            self.serial_com.stop()
        
        # Wait for a moment to ensure threads have properly terminated
        time.sleep(0.5)
        
        # Check if threads have properly stopped
        if nirs and self.bluetooth_com.thread and self.bluetooth_com.thread.is_alive():
            print("Warning: NIRS thread did not terminate properly")
        
        if force and self.serial_com.thread and self.serial_com.thread.is_alive():
            print("Warning: Force sensor thread did not terminate properly")
        
    def get_force_dataframe(self):
        """
        Convert collected force data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing force data
            
        Creates a DataFrame from the internally stored force data points,
        with columns for sensor_id, timestamp, and value.
        """
        if not self.force_data:
            print("Warning: No force data collected")
        return pd.DataFrame(self.force_data)

    def get_nirs_dataframe(self):
        """
        Convert collected NIRS data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing NIRS data
            
        Creates a DataFrame from the internally stored NIRS data points,
        with columns for sensor_id, timestamp, and value.
        """
        if not self.nirs_data:
            print("Warning: No NIRS data collected")
        return pd.DataFrame(self.nirs_data)

    def clear_data(self):
        """
        Clear all collected data.
        
        Resets the internal data storage for both force and NIRS data.
        """
        self.force_data = []
        self.nirs_data = []