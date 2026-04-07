"""
Main run of application code for LBS-1A
"""

import time
import sys
import json
from math import sin, cos
from pylx16a.lx16a import LX16A, ServoTimeoutError
from center_motors import set_angle

NEUTRAL = 120 # motor mid-point (0 - 240 degrees)
HIP_SWEEP = 20
KNEE_SWEEP = 30
SWEEP_MS = 600 # duration of each move in ms

# Leg name -> hip ID (knee = hip + 4)
LEGS = {"RL": 1, "FR": 4, "RR": 2, "FL": 3}


# Load neutral values for each motor
with open("calibration.json") as f:
    neutral = json.load(f)

def move_to_neutral(motor_id):
    set_angle(motor_id, neutral[str(motor_id)])

def move_relative(motor_id, offset):
    set_angle(motor_id, neutral[str(motor_id)] + offset)