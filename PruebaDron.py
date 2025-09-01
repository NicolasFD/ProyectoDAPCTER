from dronekit import connect, VehicleMode
import time

# Connect to the vehicle
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
# Arm and takeoff
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
while not vehicle.armed:
    time.sleep(1)
vehicle.simple_takeoff(10)  # Take off to 10 meters
# Wait until the drone reaches the altitude
while True:
    if vehicle.location.global_relative_frame.alt >= 9.5:
        break
    time.sleep(1)
vehicle.mode = VehicleMode("LAND")