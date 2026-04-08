#!/usr/bin/env python3
"""
Optimus Primal - Quadruped Gait Controller
WASD keyboard control via evdev (Rii Mini i8)
Trot gait: diagonal pairs alternate (FL+RR, FR+RL)
"""

import json
import sys
import time
import threading
from pylx16a.lx16a import LX16A

# ============================================================
# CONFIG
# ============================================================
CALIBRATION_FILE = "calibration.json"
PROBE_SERVO_ID = 1   # known-present servo used to verify the bus

# Leg definitions: (hip_id, knee_id, direction_multiplier)
# direction_multiplier: -1 for left side, +1 for right side
# This means positive offset = "forward" for all legs
LEGS = {
    "FL": {"hip": 3, "knee": 7, "dir": 1},  # Front Left
    "FR": {"hip": 4, "knee": 8, "dir":  -1},  # Front Right
    "RL": {"hip": 1, "knee": 5, "dir": 1},  # Rear Left
    "RR": {"hip": 2, "knee": 6, "dir":  -1},  # Rear Right
}

# Gait parameters
STANDING_HIP_OFFSET = 65    # degrees back from neutral for standing pose
STANDING_KNEE_OFFSET = -80  # degrees bent for standing pose

STRIDE_LENGTH = 20          # degrees of hip swing per step (half forward, half back)
KNEE_LIFT = 15              # degrees of knee lift during swing phase
STEP_DURATION = 300         # ms per gait phase (trot only)
MOVE_DURATION = 500         # ms for servo move commands

# ============================================================
# INITIALIZATION
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
for leg_name, leg in LEGS.items():
    servos[leg["hip"]] = LX16A(leg["hip"])
    servos[leg["knee"]] = LX16A(leg["knee"])

# Shared state between threads
state = {
    "mode": "idle",       # "idle", "forward", "backward"
    "running": True,
}

# ============================================================
# MOTOR HELPERS
# ============================================================
def get_neutral(motor_id):
    return neutral[str(motor_id)]

def move_servo(motor_id, angle, duration=MOVE_DURATION):
    """Move servo to absolute angle, clamped to 0-240."""
    angle = max(0, min(240, angle))
    servos[motor_id].move(angle, time=duration)

def set_leg(leg_name, hip_offset, knee_offset, duration=MOVE_DURATION):
    """
    Move a leg relative to its standing pose.
    Positive hip_offset = swing forward.
    Positive knee_offset = lift/extend.
    Direction multiplier handles left/right mirroring.
    """
    leg = LEGS[leg_name]
    d = leg["dir"]
    
    hip_target = get_neutral(leg["hip"]) + (STANDING_HIP_OFFSET * d * -1) + (hip_offset * d * -1)
    knee_target = get_neutral(leg["knee"]) + (STANDING_KNEE_OFFSET * d) + (knee_offset * d)
    
    move_servo(leg["hip"], hip_target, duration)
    move_servo(leg["knee"], knee_target, duration)

# ============================================================
# POSES
# ============================================================
def stand():
    """Move to standing pose."""
    print("Standing...")
    for leg_name in LEGS:
        set_leg(leg_name, 0, 0, duration=500)
    time.sleep(0.6)

def sit():
    """Move all servos to neutral (legs straight down)."""
    print("Sitting...")
    for motor_id in servos:
        move_servo(motor_id, get_neutral(motor_id), 500)
    time.sleep(0.6)

# ============================================================
# TROT GAIT
# ============================================================
def trot_cycle(direction=1):
    """
    One full trot cycle. direction=1 for forward, -1 for backward.
    
    Trot gait: diagonal pairs move together.
    Phase 1: FL+RR swing forward, FR+RL push back
    Phase 2: FR+RL swing forward, FL+RR push back
    """
    stride = STRIDE_LENGTH * direction
    
    # Phase 1: FL+RR swing (lift + move forward), FR+RL stance (push back)
    set_leg("FL",  stride/2,  KNEE_LIFT)   # swing forward + lift
    set_leg("RR",  stride/2,  KNEE_LIFT)   # swing forward + lift
    set_leg("FR", -stride/2,  0)           # push back on ground
    set_leg("RL", -stride/2,  0)           # push back on ground
    time.sleep(STEP_DURATION / 1000.0)
    
    # Plant FL+RR
    set_leg("FL",  stride/2,  0)
    set_leg("RR",  stride/2,  0)
    time.sleep(0.05)
    
    # Phase 2: FR+RL swing (lift + move forward), FL+RR stance (push back)
    set_leg("FR",  stride/2,  KNEE_LIFT)   # swing forward + lift
    set_leg("RL",  stride/2,  KNEE_LIFT)   # swing forward + lift
    set_leg("FL", -stride/2,  0)           # push back on ground
    set_leg("RR", -stride/2,  0)           # push back on ground
    time.sleep(STEP_DURATION / 1000.0)
    
    # Plant FR+RL
    set_leg("FR",  stride/2,  0)
    set_leg("RL",  stride/2,  0)
    time.sleep(0.05)

# ============================================================
# CRAWL (WAVE) GAIT — ported from stance.py:full_stride()
# ============================================================
# stance.py uses *opposite* dir signs from the trot LEGS dict above
# and absolute hip/knee offsets (not deltas from STANDING_*_OFFSET).
# Keep this self-contained so the trot gait is unaffected.
LEGS_CRAWL = {
    "FL": {"hip": 3, "knee": 7, "dir": -1},
    "FR": {"hip": 4, "knee": 8, "dir":  1},
    "RL": {"hip": 1, "knee": 5, "dir": -1},
    "RR": {"hip": 2, "knee": 6, "dir":  1},
}

# Per-side knee trim (deg). Positive = more bent (lower that side).
# Robot leans left → straighten left knees (negative trim) and/or bend right (positive).
# Computed from leveled pose: FL=-70, FR=-75, RL=-65, RR=-80
# vs crawl_stance baseline (front=-100, rear=-50). trim = baseline - target.
KNEE_TRIM = {"FL": -30, "FR": -25, "RL": +15, "RR": +30}

def leg_abs(name, hip_off, knee_off, duration=MOVE_DURATION):
    """Set one leg to absolute hip/knee offsets (stance.py convention)."""
    leg = LEGS_CRAWL[name]
    d = leg["dir"]
    knee_off_trimmed = knee_off - KNEE_TRIM[name]   # more negative = more bent
    hip_target  = get_neutral(leg["hip"])  + (hip_off  * d * -1)
    knee_target = get_neutral(leg["knee"]) + (knee_off_trimmed * d)
    move_servo(leg["hip"],  hip_target,  duration)
    move_servo(leg["knee"], knee_target, duration)

def recenter_leg(name, stance_knee):
    """Lift a forward leg, swing hip back to center, plant. No dragging."""
    leg_abs(name, 10, stance_knee + 35)   # lift (knee straighter)
    time.sleep(0.35)
    leg_abs(name, 35, stance_knee + 35)   # swing hip back to center
    time.sleep(0.35)
    leg_abs(name, 35, stance_knee)        # plant
    time.sleep(0.35)

def crawl_stance():
    """Starting stance: front low, rear high."""
    leg_abs("FL", 35, -100)
    leg_abs("FR", 35, -100)
    leg_abs("RL", 35,  -50)
    leg_abs("RR", 35,  -50)
    time.sleep(0.8)

def full_stride():
    """One full crawl stride: RL+FR step, then RR+FL step."""
    crawl_stance()

    # === Side 1: RL + FR step ===
    # Raise FL + RR (lift 15), drop FR to free RL diagonal
    leg_abs("FL", 45, -65)
    leg_abs("RR", 45, -65)
    leg_abs("FR", 25, -110)
    time.sleep(0.6)

    # Swing RL forward
    leg_abs("RL", 10, -50)
    time.sleep(0.6)

    # Plant FL + RR
    leg_abs("FL", 35, -100)
    leg_abs("RR", 35,  -70)
    time.sleep(0.6)

    # Swing FR forward
    leg_abs("FR", 10, -75)
    time.sleep(0.6)

    # === Side 2: RR + FL step ===
    # Raise FR + RL (lift 15), drop FL to free RR diagonal
    leg_abs("FR", 45, -65)
    leg_abs("RL", 45, -65)
    leg_abs("FL", 25, -110)
    time.sleep(0.6)

    # Swing RR forward
    leg_abs("RR", 10, -50)
    time.sleep(0.6)

    # Plant FR + RL
    leg_abs("FR", 35, -100)
    leg_abs("RL", 35,  -70)
    time.sleep(0.6)

    # Swing FL forward
    leg_abs("FL", 10, -75)
    time.sleep(0.6)

    # Recenter the two legs still forward (FL front, RR rear) — lift, swing, plant
    recenter_leg("FL", -100)
    recenter_leg("RR",  -50)

    # Reset to clean stance
    crawl_stance()

def walk_and_measure(n=10):
    """Run n full strides, time them, prompt for measured distance."""
    crawl_stance()
    input("Place tape at start, then press Enter to walk...")
    t0 = time.time()
    for i in range(n):
        full_stride()
        print(f"  stride {i+1}/{n} done")
    elapsed = time.time() - t0
    print(f"\n=== {n} strides in {elapsed:.2f} s ({elapsed/n:.2f} s/stride) ===")
    try:
        distance_cm = float(input("Measured distance traveled (cm): "))
        speed = (distance_cm / 100.0) / elapsed
        print(f"Distance: {distance_cm} cm | Speed: {speed:.3f} m/s")
    except ValueError:
        print("(no distance entered, skipping speed calc)")
    sit()

# ============================================================
# GAIT THREAD
# ============================================================
def gait_loop():
    """Runs continuously, checks state and executes gait."""
    stand()
    
    while state["running"]:
        if state["mode"] == "forward":
            trot_cycle(direction=1)
        elif state["mode"] == "backward":
            trot_cycle(direction=-1)
        else:
            time.sleep(0.05)  # idle, don't burn CPU
    
    sit()

# ============================================================
# KEYBOARD INPUT (evdev)
# ============================================================
def find_keyboard():
    """Find the Rii Mini i8 or first available keyboard."""
    try:
        import evdev
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            caps = dev.capabilities(verbose=True)
            if ('EV_KEY', 1) in caps:
                print(f"Found keyboard: {dev.name} ({dev.path})")
                return dev
    except ImportError:
        print("evdev not installed! Run: pip install evdev")
        return None
    
    print("No keyboard found!")
    return None

def input_loop():
    """Read keypresses and update state."""
    import evdev
    from evdev import ecodes
    
    keyboard = find_keyboard()
    if not keyboard:
        print("Falling back to terminal input (press w/s/q + Enter)")
        terminal_input_loop()
        return
    
    keyboard.grab()  # exclusive access
    print("Keyboard grabbed. W=forward, S=backward, Q=quit")
    
    try:
        for event in keyboard.read_loop():
            if not state["running"]:
                break
            
            if event.type == ecodes.EV_KEY:
                key = evdev.categorize(event)
                
                if key.keystate == key.key_down:
                    if key.scancode == ecodes.KEY_W:
                        state["mode"] = "forward"
                        print(">> Forward")
                    elif key.scancode == ecodes.KEY_S:
                        state["mode"] = "backward"
                        print(">> Backward")
                    elif key.scancode == ecodes.KEY_Q or key.scancode == ecodes.KEY_ESC:
                        print(">> Quitting...")
                        state["running"] = False
                        break
                
                elif key.keystate == key.key_up:
                    if key.scancode in (ecodes.KEY_W, ecodes.KEY_S):
                        state["mode"] = "idle"
                        print(">> Idle")
    finally:
        keyboard.ungrab()

def terminal_input_loop():
    """Fallback if evdev isn't available."""
    print("Terminal mode: w=forward, s=backward, x=stop, q=quit")
    while state["running"]:
        try:
            cmd = input("> ").strip().lower()
            if cmd == "w":
                state["mode"] = "forward"
                print(">> Forward")
            elif cmd == "s":
                state["mode"] = "backward"
                print(">> Backward")
            elif cmd == "x":
                state["mode"] = "idle"
                print(">> Idle")
            elif cmd == "q":
                state["running"] = False
        except EOFError:
            state["running"] = False

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Optimus Primal Gait Controller ===")

    if "--measure" in sys.argv:
        # Crawl gait benchmark mode: 10 iterations, manual distance entry
        n = 10
        for arg in sys.argv:
            if arg.startswith("--n="):
                n = int(arg.split("=", 1)[1])
        walk_and_measure(n=n)
        sys.exit(0)

    print("Starting up...")

    # Start gait loop in background thread
    gait_thread = threading.Thread(target=gait_loop, daemon=True)
    gait_thread.start()
    
    # Run input on main thread
    try:
        input_loop()
    except KeyboardInterrupt:
        print("\nInterrupted!")
        state["running"] = False
    
    # Wait for gait thread to finish
    gait_thread.join(timeout=3)
    print("Shutdown complete.")