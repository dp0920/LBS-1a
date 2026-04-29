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

    If XCONFIG is True, the standing pose AND the per-step offsets are
    flipped on the rear legs so they sit in the ANYmal X position
    (rear hips swept forward, rear knees bent the opposite way).
    """
    leg = LEGS[leg_name]
    d = leg["dir"]

    stand_hip = STANDING_HIP_OFFSET
    stand_knee = STANDING_KNEE_OFFSET
    h_off = hip_offset
    k_off = knee_offset
    if XCONFIG and leg_name in ("RL", "RR"):
        # ANYmal X: rear hip + knee both flipped, front mammalian.
        stand_hip = -stand_hip
        stand_knee = -stand_knee
        h_off = -h_off
        k_off = -k_off

    hip_target  = get_neutral(leg["hip"])  + (stand_hip  * d * -1) + (h_off * d * -1)
    knee_target = get_neutral(leg["knee"]) + (stand_knee * d)      + (k_off * d)

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
# Targets: FL=-70, FR=-65, RL=-65, RR=-65 (all roughly even)
# vs crawl_stance baseline (front=-100, rear=-50). trim = baseline - target.
KNEE_TRIM = {"FL": -33, "FR": +2, "RL": +10, "RR": +12}

# X-config (ANYmal X stance). When True, BOTH the rear hip and the rear
# knee are flipped (rear hip sweeps forward, rear knee bends opposite of
# front), while the front legs stay mammalian. Sign pattern derived from
# the user's manual kinesthetic capture:
#   FL: hip+47, knee-35    FR: hip+58, knee-71
#   RL: hip-47, knee+63    RR: hip-46, knee+53
XCONFIG = False

def leg_abs(name, hip_off, knee_off, duration=MOVE_DURATION):
    """Set one leg to absolute hip/knee offsets (stance.py convention).

    If XCONFIG is True, rear legs' hip_off and knee_off are negated so
    the rear hips swing forward instead of back and the rear knees bend
    opposite from the front — putting the rear legs into X-config while
    keeping the gait's swing/plant logic unchanged.
    """
    leg = LEGS_CRAWL[name]
    d = leg["dir"]
    if XCONFIG and name in ("RL", "RR"):
        hip_off = -hip_off
        knee_off = -knee_off
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
    """Front low, rear high — biases CoG backward so the body doesn't
    tip onto a leg as it swings forward."""
    leg_abs("FL", 35, -100)
    leg_abs("FR", 35, -100)
    leg_abs("RL", 35,  -50)
    leg_abs("RR", 35,  -50)
    time.sleep(0.8)

def full_stride_x(n_cycles=2, dt=0.4):
    """Simple diagonal trot for X-config (use with --xconfig).

    Pattern (each cycle, two beats):
      beat 1: lift+swing FL+RR forward, FR+RL plant
      beat 2: lift+swing FR+RL forward, FL+RR plant

    Hip/knee values are written in mammalian convention; --xconfig flips
    rear-leg values inside leg_abs so the actual hardware motion is
    correct for the X-pose.

    Tune dt for stability — too fast and the body falls during swing.
    """
    # Settle into a balanced X-stance
    for leg in ["FL", "FR", "RL", "RR"]:
        leg_abs(leg, 35, -80)
    time.sleep(0.8)

    for _ in range(n_cycles):
        # ---- Beat 1: FL+RR diagonal lifts and swings forward ----
        leg_abs("FL", 35, -100)   # lift (knee more bent)
        leg_abs("RR", 35, -100)
        leg_abs("FR", 50, -80)    # FR plants forward-shifted
        leg_abs("RL", 50, -80)    # RL plants forward-shifted
        time.sleep(dt)
        leg_abs("FL", 50, -100)   # swing FL hip forward (still lifted)
        leg_abs("RR", 50, -100)
        time.sleep(dt)
        leg_abs("FL", 50, -80)    # plant FL forward
        leg_abs("RR", 50, -80)
        time.sleep(dt)

        # ---- Beat 2: FR+RL diagonal lifts and swings forward ----
        leg_abs("FR", 35, -100)
        leg_abs("RL", 35, -100)
        leg_abs("FL", 20, -80)    # FL/RR push back to center (stance)
        leg_abs("RR", 20, -80)
        time.sleep(dt)
        leg_abs("FR", 50, -100)
        leg_abs("RL", 50, -100)
        time.sleep(dt)
        leg_abs("FR", 50, -80)
        leg_abs("RL", 50, -80)
        time.sleep(dt)

def full_stride():
    """One full crawl stride: RL+FR step, then RR+FL step."""
    crawl_stance()

    # === Side 1: RL + FR step ===
    # Raise FL + RR, drop FR to free RL diagonal
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
    # Raise FR + RL, drop FL to free RR diagonal
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

# ============================================================
# JSON GAIT PLAYBACK (from mujoco_gait.py optimization)
# ============================================================
# best_gait.json layout (from mujoco_gait.py):
#   params[0..103] = 13 phases × 8 joint angles (degrees)
#     per-phase order: FL_hip, FL_knee, FR_hip, FR_knee,
#                      RL_hip, RL_knee, RR_hip, RR_knee
#   params[104]    = phase_time (seconds per phase)
#
# Angles are in stance.py convention — same as leg_abs() takes.

PHASE_ORDER = [
    "start",
    "shift_FR", "swing_FR", "plant_FR",
    "shift_RL", "swing_RL", "plant_RL",
    "shift_FL", "swing_FL", "plant_FL",
    "shift_RR", "swing_RR", "plant_RR",
]

def decode_gait_json(path):
    """Load best_gait.json → (list of 13 phase dicts, phase_time seconds)."""
    with open(path) as f:
        d = json.load(f)
    params = d["params"]
    if len(params) != len(PHASE_ORDER) * 8 + 1:
        raise ValueError(
            f"Expected {len(PHASE_ORDER)*8 + 1} params, got {len(params)}")
    phases = []
    idx = 0
    for name in PHASE_ORDER:
        phases.append({
            "name": name,
            "FL": (params[idx],   params[idx+1]),
            "FR": (params[idx+2], params[idx+3]),
            "RL": (params[idx+4], params[idx+5]),
            "RR": (params[idx+6], params[idx+7]),
        })
        idx += 8
    phase_time = float(params[idx])
    return phases, phase_time, d.get("reward")

def apply_phase(phase, duration_ms):
    """Drive all 4 legs to this phase's pose."""
    for name in ("FL", "FR", "RL", "RR"):
        hip_off, knee_off = phase[name]
        leg_abs(name, hip_off, knee_off, duration=duration_ms)

def run_json_gait(path="best_gait.json", cycles=5, speed_scale=1.0):
    """Play a mujoco-optimized gait on the robot."""
    phases, phase_time, reward = decode_gait_json(path)
    phase_time *= speed_scale
    duration_ms = int(phase_time * 1000)
    rew_str = f"{reward:.2f}" if reward is not None else "n/a"
    print(f"Loaded {path}: reward={rew_str}, "
          f"phase_time={phase_time:.3f}s, duration={duration_ms}ms")

    # Start pose (phase 0) — slow, deliberate move so robot doesn't topple
    print("Moving to start pose...")
    apply_phase(phases[0], duration_ms=800)
    time.sleep(1.0)

    stride_phases = phases[1:]  # 12 swing/shift/plant phases per stride
    input("Place tape at start, then press Enter to walk...")
    t0 = time.time()
    for i in range(cycles):
        for p in stride_phases:
            apply_phase(p, duration_ms=duration_ms)
            time.sleep(phase_time)
        print(f"  stride {i+1}/{cycles} done")
    elapsed = time.time() - t0
    print(f"\n=== {cycles} strides in {elapsed:.2f} s "
          f"({elapsed/cycles:.2f} s/stride) ===")
    try:
        distance_cm = float(input("Measured distance traveled (cm): "))
        speed = (distance_cm / 100.0) / elapsed
        print(f"Distance: {distance_cm} cm | Speed: {speed:.3f} m/s")
    except ValueError:
        print("(no distance entered, skipping speed calc)")
    sit()

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
    """Find the Rii Mini i8 or first available keyboard.

    Skips HDMI / audio / GPIO pseudo-devices that expose EV_KEY for
    things like CEC hotplug or volume buttons but never produce W/S.
    """
    SKIP_SUBSTRINGS = ("hdmi", "vc4", "headphone", "internal", "gpio",
                       "vbus", "power", "lid", "cec")
    try:
        import evdev
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            name = dev.name.lower()
            if any(skip in name for skip in SKIP_SUBSTRINGS):
                continue
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

    if "--xconfig" in sys.argv:
        XCONFIG = True
        print("  XCONFIG enabled — rear legs flipped to ANYmal X-stance")

    if "--measure" in sys.argv:
        # Crawl gait benchmark mode: 10 iterations, manual distance entry
        n = 10
        for arg in sys.argv:
            if arg.startswith("--n="):
                n = int(arg.split("=", 1)[1])
        walk_and_measure(n=n)
        sys.exit(0)

    if "--gait" in sys.argv:
        # Play a mujoco-optimized gait from JSON.
        #   python gait_controller.py --gait best_gait.json [--n=5] [--slow]
        gait_path = "best_gait.json"
        cycles = 5
        speed_scale = 1.0
        i = sys.argv.index("--gait")
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            gait_path = sys.argv[i + 1]
        for arg in sys.argv:
            if arg.startswith("--n="):
                cycles = int(arg.split("=", 1)[1])
            elif arg == "--slow":
                speed_scale = 2.0
            elif arg.startswith("--scale="):
                speed_scale = float(arg.split("=", 1)[1])
        crawl_stance()  # settle into base pose before running the learned gait
        run_json_gait(gait_path, cycles=cycles, speed_scale=speed_scale)
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