"""
Data Acquisition System for Force and NIRS Sensors

This module provides classes for collecting data from force sensors via serial connection
and NIRS (Near-Infrared Spectroscopy) sensors via Bluetooth Low Energy (BLE). It handles
data acquisition, sensor calibration, and fallback to simulated data when hardware
connections fail.

Main components:
- SerialCommunicator: Handles force sensor data via serial port
- BluetoothCommunicator: Handles NIRS sensor data via BLE
- DataGenerator: Coordinates both communicators and stores collected data

The system will automatically fall back to generating synthetic data if hardware
connections cannot be established.
"""

import serial
import serial.serialutil
import serial.tools.list_ports
import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner
import time
import threading
import struct


class SerialCommunicator:
    """
    Handles serial communication for a force sensor, calling back with each new data point.
    """
    def __init__(self, data_callback, port: str = None, baudrate: int = 19200):
        """
        Initialize the serial communicator.
        
        Args:
            data_callback: Function to call with each new data point
            port (str, optional): Serial port to connect to. If None, uses synthetic data. Defaults to None.
            baudrate (int, optional): Baud rate for serial connection. Defaults to 19200.
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
            port (str): Serial port identifier
        """
        self.port = port

    def update_calibration(self, slope: float, intercept: float):
        """
        Update the calibration parameters for force conversion.
        
        Args:
            slope (float): Calibration slope
            intercept (float): Calibration intercept
        """
        self.calibration_slope = slope
        self.calibration_intercept = intercept

    def _read_force(self):
        """
        Read loop: either real serial or synthetic; calls data_callback on each value.
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
        """Generate synthetic force data when no serial connection is available."""
        while self.running:
            try:
                # synthetic
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
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_force, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the force data acquisition thread.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


class BluetoothCommunicator:
    """
    Handles BLE notifications for NIRS sensors; falls back to simulated data on connection failure.
    """
    def __init__(self, data_callback, device_name: str, data_uuid: str, max_attempts: int = 3):
        """
        Initialize the Bluetooth communicator.
        
        Args:
            data_callback: Function to call with each new data set
            device_name (str): Name of the BLE device to connect to
            data_uuid (str): UUID of the data characteristic to subscribe to
            max_attempts (int, optional): Maximum number of connection attempts. Defaults to 3.
        """
        self.data_callback = data_callback
        self.device_name = device_name
        self.data_uuid = data_uuid
        self.client = None
        self.connection_state = ''
        self.running = False
        self.thread = None
        self.debugging = False
        self.debug_data_buffer = []
        self.connection_attempts = 0
        self.max_attempts = max_attempts

    @staticmethod
    def _generate_nirs_profile(noise_level=0.4):
        """
        Generate synthetic NIRS data profile.
        
        Args:
            noise_level (float, optional): Amplitude of random noise. Defaults to 0.4.
            
        Returns:
            list: List of NIRS data points
        """
        n_samples = int(np.random.uniform(300, 800))
        t = np.linspace(0, np.pi, n_samples)
        base = np.sin(t) * np.random.uniform(30, 80)
        noise = np.random.normal(0, noise_level, n_samples)
        return np.clip(base + noise, 20, np.max(base)).tolist()

    def _simulate_nirs_data_generator(self):
        """
        Generate simulated NIRS data and call the callback function.
        Used when BLE connection fails.
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
        """
        try:
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
            data (bytes): Raw bytes from BLE notification
        """
        try:
            ts = time.time()
            f1, f2, f3, _ = struct.unpack('<fffB I', data)
            items = [f"2_{ts:.3f}_{f1}", f"3_{ts:.3f}_{f2}", f"4_{ts:.3f}_{f3}"]
            print('NIRS data:', items)
            self.data_callback(items)
        except struct.error:
            print(f"Malformed BLE data received, incorrect format")
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} - NIRS handler error: {e}")

    def _run_loop(self):
        """
        Main thread loop for BLE operations. Handles connection attempts and switches
        to simulation mode if connection fails after max attempts.
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Attempt to connect with retries
            connected = False
            for attempt in range(1, self.max_attempts + 1):
                try:
                    connected = loop.run_until_complete(self._connect_and_listen())
                except Exception as e:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - NIRS connect attempt {attempt}/{self.max_attempts} error: {e}"
                    print(self.connection_state)
                    connected = False
                if connected:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - Connected to {self.device_name} on attempt {attempt}"
                    print(self.connection_state)
                    break
                else:
                    self.connection_state = f"{time.strftime('%H:%M:%S')} - Could not connect to {self.device_name}. Reconnecting ({attempt}/{self.max_attempts})"
                    print(self.connection_state)
                    time.sleep(1)

            if not connected:
                self.debugging = True
                self.connection_state = f"{time.strftime('%H:%M:%S')} - Failed to connect after {self.max_attempts} attempts. Starting simulation mode"
                print(self.connection_state)
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
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the BLE communication thread.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


class DataGenerator:
    """
    Coordinates both serial and BLE communicators, storing data in lists.
    """
    def __init__(self):
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
        self.serial_com = SerialCommunicator(data_callback=self._on_force)
        self.bluetooth_com = BluetoothCommunicator(
            data_callback=self._on_nirs,
            device_name="Train.Red FYER 0707",
            data_uuid="00002201-D578-4741-9B5B-7C64E958CFC6",
            max_attempts=2
        )

    def _on_force(self, data_str: str):
        """
        Callback for force data processing.
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
        Callback for NIRS data processing.
        Always stores data internally and forwards to external callback if set and not paused.
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
        """
        self.external_force_callback = force_callback
        self.external_nirs_callback = nirs_callback
        # Resume data forwarding by default when setting new callbacks
        self.pause_callbacks = False
    
    def pause_data_forwarding(self):
        """
        Pause forwarding data to external callbacks.
        Data will still be collected and stored internally.
        """
        self.pause_callbacks = True
        print("Data forwarding paused - still collecting data")
        
    def resume_data_forwarding(self):
        """
        Resume forwarding data to external callbacks.
        """
        self.pause_callbacks = False
        print("Data forwarding resumed")

    def set_force_port(self, port: str):
        """
        Set the serial port for force sensor.
        
        Args:
            port (str): Serial port identifier
        """
        if not port:
            raise ValueError("Serial port cannot be empty")
        self.serial_com.set_port(port)

    def start_nirs_connection(self):
        """
        Start the BLE connection to NIRS device.

        Returns:
            bool: True if the connection is successfully started
        """
        if self.bluetooth_com.running:
            print("NIRS connection already running")
            return True

        # Clear any existing data before starting connection
        self.clear_data()

        # Start the bluetooth connection
        self.bluetooth_com.start()
        return True

    def is_nirs_connected(self):
        """
        Check if NIRS is connected and running.

        Returns:
            bool: True if NIRS connection is active
        """
        # connected_nirs = self.bluetooth_com.running
        return self.bluetooth_com.running and not self.bluetooth_com.debugging

    def update_calibration(self, slope: float, intercept: float):
        """
        Update force sensor calibration parameters.
        
        Args:
            slope (float): Calibration slope
            intercept (float): Calibration intercept
            
        Raises:
            ValueError: If slope is zero (would cause division by zero)
        """
        if slope == 0:
            raise ValueError("Calibration slope cannot be zero (division by zero)")
        self.serial_com.update_calibration(slope, intercept)

    def start(self, nirs=True, force=True):
        """
        Start data acquisition.
        
        Args:
            nirs (bool, optional): Whether to start NIRS data acquisition. Defaults to True.
            force (bool, optional): Whether to start force data acquisition. Defaults to True.
        """
        if nirs:
            self.bluetooth_com.start()
        if force:
            self.serial_com.start()

    def stop(self, nirs=True, force=True):
        """
        Stop data acquisition.
        
        Args:
            nirs (bool, optional): Whether to stop NIRS data acquisition. Defaults to True.
            force (bool, optional): Whether to stop force data acquisition. Defaults to True.
        """
        if nirs:
            self.bluetooth_com.stop()
        if force:
            self.serial_com.stop()

    def get_force_dataframe(self):
        """
        Convert collected force data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing force data
        """
        if not self.force_data:
            print("Warning: No force data collected")
        return pd.DataFrame(self.force_data)

    def get_nirs_dataframe(self):
        """
        Convert collected NIRS data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing NIRS data
        """
        if not self.nirs_data:
            print("Warning: No NIRS data collected")
        return pd.DataFrame(self.nirs_data)

    def clear_data(self):
        """
        Clear all collected data.
        """
        self.force_data = []
        self.nirs_data = []
        print("All collected data has been cleared")

# Example usage:
# dg = DataGenerator()
# dg.set_force_port("/dev/ttyUSB0")
# dg.start()
# time.sleep(10)
# dg.stop()
# df_force = dg.get_force_dataframe()
# df_nirs = dg.get_nirs_dataframe()