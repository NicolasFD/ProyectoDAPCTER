from dronekit import connect, VehicleMode
import time

# Connect to the vehicle
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)