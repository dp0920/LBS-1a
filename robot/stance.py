#!/usr/bin/env python3
"""
Optimus Primal - Stance Tuning Tool
Interactively adjust hip/knee angles to find a stable standing pose.
Test stability by lifting individual legs.
"""

import json
import time
from pylx16a.lx16a import LX16A

# ============================================================
# CONFIG
# ============================================================
CALIBRATION_FILE = "calibration.json"
PROBE_SERVO_ID = 1

LEGS = {
    "FL": {"hip": 3, "knee": 7, "dir": -1},
    "FR": {"hip": 4, "knee": 8, "dir":  1},
    "RL": {"hip": 1, "knee": 5, "dir": -1},
    "RR": {"hip": 2, "knee": 6, "dir":  1},
}

# ============================================================
# INIT
# ============================================================
def find_servo_bus(probe_id=PROBE_SERVO_ID):
    """Auto-detect which /dev/ttyUSB* has the LX-16A bus by probing a known servo."""
    import serial.tools.list_ports
    candidates = [p.device for p in serial.tools.list_ports.comports() if "USB" in p.device]
    for dev in sorted(candidates):
        try:
            LX16A.initialize(dev, timeout=0.2)
            LX16A(probe_id).get_physical_angle()
            print(f"Found servo bus on {dev}")
            return dev
        except Exception:
            continue
    raise RuntimeError("No LX-16A bus found on any /dev/ttyUSB*")

SERIAL_PORT = find_servo_bus()

with open(CALIBRATION_FILE) as f:
    neutral = json.load(f)

servos = {}
for leg in LEGS.values():
    servos[leg["hip"]] = LX16A(leg["hip"])
    servos[leg["knee"]] = LX16A(leg["knee"])

# Current tuning values
hip_offset = 35
knee_offset = -80

# Per-leg knee trim — keep in sync with gait_controller.py
# Positive = more bent (lower); negative = straighter (higher)
KNEE_TRIM = {"FL": -20, "FR": -20, "RL": 0, "RR": 0}

def get_neutral(motor_id):
    return neutral[str(motor_id)]

def move_servo(motor_id, angle, duration=500):
    angle = max(0, min(240, angle))
    servos[motor_id].move(angle, time=duration)

def apply_stance(duration=500):
    """Apply current hip/knee offsets to all 4 legs (with per-leg knee trim)."""
    for name, leg in LEGS.items():
        d = leg["dir"]
        knee_off_trimmed = knee_offset - KNEE_TRIM[name]
        hip_target = get_neutral(leg["hip"]) + (hip_offset * d * -1)
        knee_target = get_neutral(leg["knee"]) + (knee_off_trimmed * d)
        move_servo(leg["hip"], hip_target, duration)
        move_servo(leg["knee"], knee_target, duration)
    print(f"  Stance applied: hip={hip_offset}°, knee={knee_offset}° (+ per-leg trim)")

def lift_leg(leg_name, lift_amount=30):
    """Lift one leg by bending the knee, keep other 3 planted."""
    if leg_name not in LEGS:
        print(f"  Unknown leg: {leg_name}. Use FL, FR, RL, RR")
        return
    
    leg = LEGS[leg_name]
    d = leg["dir"]
    
    # Lift by pulling knee up
    knee_target = get_neutral(leg["knee"]) + (knee_offset * d) + (lift_amount * d)
    # Also swing hip forward slightly to clear ground
    hip_target = get_neutral(leg["hip"]) + (hip_offset * d * -1) + (10 * d * -1)
    
    move_servo(leg["knee"], knee_target, 750)
    move_servo(leg["hip"], hip_target, 750)
    print(f"  Lifted {leg_name} (knee +{lift_amount}°)")

def plant_leg(leg_name):
    """Return a lifted leg to the current stance."""
    if leg_name not in LEGS:
        print(f"  Unknown leg: {leg_name}. Use FL, FR, RL, RR")
        return
    
    leg = LEGS[leg_name]
    d = leg["dir"]
    
    hip_target = get_neutral(leg["hip"]) + (hip_offset * d * -1)
    knee_target = get_neutral(leg["knee"]) + (knee_offset * d)
    
    move_servo(leg["hip"], hip_target, 400)
    move_servo(leg["knee"], knee_target, 400)
    print(f"  Planted {leg_name}")

def stand():
    """Return all servos to neutral."""
    for motor_id in servos:
        move_servo(motor_id, get_neutral(motor_id), 500)
    print("  All servos at neutral")

def read_positions():
    """Read current physical position of all servos."""
    print("\n  Current servo positions:")
    for name, leg in LEGS.items():
        try:
            hip_pos = servos[leg["hip"]].get_physical_angle()
            knee_pos = servos[leg["knee"]].get_physical_angle()
            hip_n = get_neutral(leg["hip"])
            knee_n = get_neutral(leg["knee"])
            print(f"    {name}: hip={hip_pos:.1f}° (neutral={hip_n}°, diff={hip_pos-hip_n:+.1f}°)  "
                  f"knee={knee_pos:.1f}° (neutral={knee_n}°, diff={knee_pos-knee_n:+.1f}°)")
        except Exception as e:
            print(f"    {name}: error reading - {e}")

def test_all_lifts():
    """Lift each leg one at a time, pause to check stability."""
    print("\n  Testing stability - lifting each leg...")
    for leg_name in ["FL", "FR", "RL", "RR"]:
        input(f"  Press Enter to lift {leg_name}...")
        lift_leg(leg_name)
        time.sleep(0.5)
        input(f"  {leg_name} lifted. Stable? Press Enter to plant...")
        plant_leg(leg_name)
        time.sleep(0.3)
    print("  Test complete!")

def set_leg_individual(leg_name, hip_off, knee_off, duration=500):
    """Set a specific leg to custom offsets (independent of global)."""
    if leg_name not in LEGS:
        print(f"  Unknown leg: {leg_name}. Use FL, FR, RL, RR")
        return
    leg = LEGS[leg_name]
    d = leg["dir"]
    hip_target = get_neutral(leg["hip"]) + (hip_off * d * -1)
    knee_target = get_neutral(leg["knee"]) + (knee_off * d)
    move_servo(leg["hip"], hip_target, duration)
    move_servo(leg["knee"], knee_target, duration)
    print(f"  {leg_name}: hip_offset={hip_off}°, knee_offset={knee_off}°")

def help():
    print("""
  ╔═══════════════════════════════════════════════════════╗
  ║         Optimus Primal - Stance Tuner                 ║
  ╠═══════════════════════════════════════════════════════╣
  ║  hip(angle)         - set hip offset for all legs     ║
  ║  knee(angle)        - set knee offset for all legs    ║
  ║  stance()           - apply current offsets            ║
  ║  lift(leg)          - lift one leg: 'FL','FR','RL','RR'║
  ║  plant(leg)         - put a lifted leg back down       ║
  ║  test()             - test all 4 lifts sequentially    ║
  ║  leg(name,hip,knee) - set one leg independently       ║
  ║  pos()              - read all servo positions         ║
  ║  sit()              - all servos to neutral            ║
  ║  save()             - print current values to copy     ║
  ╚═══════════════════════════════════════════════════════╝
    """)

# Convenience aliases
def hip(angle):
    global hip_offset
    hip_offset = angle
    apply_stance()

def knee(angle):
    global knee_offset
    knee_offset = angle
    apply_stance()

def step():
      # Starting stance - front low, rear high
    leg("FL", 35, -100)
    leg("FR", 35, -100)
    leg("RL", 35, -50)
    leg("RR", 35, -50)
    time.sleep(0.5)
    
    # Raise FL + RR, drop FR to free RL
    lift("FL", 15)
    lift("RR", 15)
    leg("FR", 25, -110)
    time.sleep(0.3)
    
    # Swing RL forward
    leg("RL", 10, -50)
    time.sleep(0.3)
    
    # Plant FL + RR back down
    leg("FL", 35, -100)
    leg("RR", 35, -70)
    time.sleep(0.3)
    
    # Swing FR forward
    leg("FR", 10, -75)
    time.sleep(0.3)
    
    # Reset to starting stance
    leg("FL", 35, -100)
    leg("FR", 35, -100)
    leg("RL", 35, -50)
    leg("RR", 35, -50)
    time.sleep(0.5)
    
    print("Step complete")

def full_stride():
    import time
    
    # Starting stance - front low, rear high
    leg("FL", 35, -100)
    leg("FR", 35, -100)
    leg("RL", 35, -50)
    leg("RR", 35, -50)
    time.sleep(0.5)
    
    # === Side 1: RL + FR step ===
    # Raise FL + RR, drop FR to free RL
    lift("FL", 15)
    lift("RR", 15)
    leg("FR", 25, -110)
    time.sleep(0.3)
    
    # Swing RL forward
    leg("RL", 10, -50)
    time.sleep(0.3)
    
    # Plant FL + RR
    leg("FL", 35, -100)
    leg("RR", 35, -70)
    time.sleep(0.3)
    
    # Swing FR forward
    leg("FR", 10, -75)
    time.sleep(0.3)
    
    # === Side 2: RR + FL step ===
    # Raise FR + RL, drop FL to free RR
    lift("FR", 15)
    lift("RL", 15)
    leg("FL", 25, -110)
    time.sleep(0.3)
    
    # Swing RR forward
    leg("RR", 10, -50)
    time.sleep(0.3)
    
    # Plant FR + RL
    leg("FR", 35, -100)
    leg("RL", 35, -70)
    time.sleep(0.3)
    
    # Swing FL forward
    leg("FL", 10, -75)
    time.sleep(0.3)
    
    # Reset stance
    leg("FL", 35, -100)
    leg("FR", 35, -100)
    leg("RL", 35, -50)
    leg("RR", 35, -50)
    time.sleep(0.5)
    
    print("Full stride complete")

def walk(steps=10):
    for i in range(steps):
        print(f"Step {i+1}/{steps}")
        full_stride()
    print("Walk complete")

def stance():
    apply_stance()

def lift(leg_name, amount=30):
    lift_leg(leg_name.upper(), amount)

def plant(leg_name):
    plant_leg(leg_name.upper())

def test():
    test_all_lifts()

def leg(name, h, k):
    set_leg_individual(name.upper(), h, k)

def pos():
    read_positions()

def save():
    print(f"\n  Current stance config:")
    print(f"    STANDING_HIP_OFFSET = {hip_offset}")
    print(f"    STANDING_KNEE_OFFSET = {knee_offset}")
    print(f"\n  Paste these into gait_controller.py")

# ============================================================
# START
# ============================================================
print("=== Optimus Primal Stance Tuner ===")
help()
print("  Starting with hip=20°, knee=20°...")
apply_stance(duration=800)
print("\n  Ready! Type commands below.\n")