from data_gen import SerialCommunicator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from collections import deque

# Initialize the serial communicator
communicator = SerialCommunicator(debugging=True)
communicator.start_collection()
print("Data collection started...")

# Real-time plot parameters
window_size = 100  # Number of points to display
force_values = deque([0] * window_size, maxlen=window_size)
time_values = deque([0] * window_size, maxlen=window_size)

fig, ax = plt.subplots()
line, = ax.plot(time_values, force_values, color='b', linestyle='-', marker='o', markersize=3)

# Graph settings
ax.set_xlim(0, window_size)
ax.set_ylim(-10, 100)  # Adjust based on expected force sensor range
ax.set_xlabel("Time (last few seconds)")
ax.set_ylabel("Force Value")
ax.set_title("Real-Time Force Sensor Data")
ax.grid(True)


# Function to update the plot
def update(frame):
    if not communicator.db_queque.empty():
        raw_data = communicator.db_queque.get()
        try:
            _, timestamp, force_value = raw_data.split("_")  # Extract time and force value
            force_value = float(force_value)
        except ValueError:
            return  # Skip faulty readings

        # Append new values
        time_values.append(float(timestamp))
        force_values.append(force_value)

        # Update plot data
        line.set_data(range(len(time_values)), force_values)
        ax.set_xlim(0, len(time_values))

    return line,


# Animation function
ani = animation.FuncAnimation(fig, update, interval=50, blit=False, save_count=100)

try:
    plt.show()  # Start the real-time graph
except KeyboardInterrupt:
    print("Stopping data collection...")
    communicator.stop_collection()
