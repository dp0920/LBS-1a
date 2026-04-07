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
SERIAL_PORT = "/dev/ttyUSB1"
CALIBRATION_FILE = "calibration.json"

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
STEP_DURATION = 300         # ms per gait phase
MOVE_DURATION = 200         # ms for servo move commands

# ============================================================
# INITIALIZATION
# ============================================================
LX16A.initialize(SERIAL_PORT)

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
# CRAWL (WAVE) GAIT — one leg at a time
# ============================================================
CRAWL_ORDER = ["FR", "RR", "FL", "RL"]   # one full iteration
CRAWL_STRIDE = 20      # deg of hip swing (tune)
CRAWL_LIFT   = 15      # deg of knee lift during swing
CRAWL_PHASE_MS = 250   # dwell per phase

def crawl_step(leg_name, direction=1):
    """Lift + swing forward, plant, then push back through stance."""
    stride = CRAWL_STRIDE * direction
    # 1. lift + swing forward
    set_leg(leg_name, stride / 2, CRAWL_LIFT)
    time.sleep(CRAWL_PHASE_MS / 1000.0)
    # 2. plant at forward position
    set_leg(leg_name, stride / 2, 0)
    time.sleep(0.05)
    # 3. push back through stance (propulsion)
    set_leg(leg_name, -stride / 2, 0)
    time.sleep(CRAWL_PHASE_MS / 1000.0)

def crawl_cycle(direction=1):
    """One full crawl iteration: each leg steps once in CRAWL_ORDER."""
    for leg in CRAWL_ORDER:
        crawl_step(leg, direction)

def walk_and_measure(n=10):
    """Run n crawl iterations, time them, prompt for measured distance."""
    stand()
    input("Place tape at start, then press Enter to walk...")
    t0 = time.time()
    for i in range(n):
        crawl_cycle(direction=1)
        print(f"  iteration {i+1}/{n} done")
    elapsed = time.time() - t0
    print(f"\n=== {n} iterations in {elapsed:.2f} s ({elapsed/n:.2f} s/iter) ===")
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