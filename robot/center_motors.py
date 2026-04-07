from lx16a import *
import time

LX16A.initialize("/dev/ttyUSB0")

def set_angle(motor_id, angle=120):
    servo = LX16A(motor_id)
    servo.move(angle, time=1000)
    print(f"Motor {motor_id} set to {angle}")

