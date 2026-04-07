"""
Boot test, Sweeps every joint a small arc and reports pass/fail
"""

import time
import sys
from math import sin, cos
from pylx16a.lx16a import LX16A, ServoTimeoutError

NEUTRAL = 120 # motor mid-point (0 - 240 degrees)
HIP_SWEEP = 20
KNEE_SWEEP = 30
SWEEP_MS = 600 # duration of each move in ms

# Leg name -> hip ID (knee = hip + 4)
LEGS = {"RL": 1, "FR": 2, "RR": 3, "FL": 4}

# Helpers
def wait(ms, eps=0.1):
    time.sleep(ms/1000 + eps)

def sep(char="-", w = 52):
    print(char * w)

def connect(sid):
    try:
        for i in range(8):
            s = LX16A(sid)
            s.get_physical_angle()
            return s
    except ServoTimeoutError: 
        return None
    except Exception as e:
        print(f"[!] ID {sid} unexpected error: {e}")

def torque_off(s):
    try:
        s.power_off()
    except Exception:
        pass

def main():
    sep("=")
    print("BOOT TEST")
    sep("=")
    


    try:
        LX16A.initialize("/dev/ttyUSB0", 0.1)
        print("BUS open")
    except Exception as e:
        print (f"  ]FATAL] Serial init failed: {e}")
        sys.exit(1)

    print("Scanning motors...")
    sep()

    found = {} # sid -> LX16A
    absent = [] # (legm joint, sid)

    for leg, hip in LEGS.items():
        for label, sid in [(f"{leg} hip", hip), (f"{leg} knee", hip + 4)]:
            s = connect(sid)
            if s:
                torque_off(s)
                found[sid] = s
                angle = s.get_physical_angle()
                print(f" {label:10s} ID={sid} {angle:6.1f} degrees")
            else:
                absent.append((label, sid))
                print(f" {label:10s} ID={sid} : No Response!")

    print()
    if absent:
        print(f"{len(absent)} motor(s) missing: {[s for _, s in absent]}")
        if input(" Continue to sweep test? (y/N): ").strip().lower() != "y":
            sys.exit(1)
        print()
    else:
        print("all 8 motors found")

### Sweep test
    print("Sweep test...")
    sep()

    passed = []
    failed = []

    for leg, hip in LEGS.items():
        for label, sid, sweep in [(f"{leg} hip", hip, HIP_SWEEP), (f"{leg} knee", hip + 4, KNEE_SWEEP)]:
            print(leg, hip)
            if sid not in found:
                print(f" {label:10s} ID={sid} : SKIP (not found)")
                failed.append(label)
                continue
            s = found[sid]
            try:
                s.move(NEUTRAL + sweep, SWEEP_MS);wait(SWEEP_MS)
                s.move(NEUTRAL - sweep, SWEEP_MS);wait(SWEEP_MS)
                s.move(NEUTRAL, SWEEP_MS);wait(SWEEP_MS)

                torque_off(s)
                actual = s.get_physical_angle()
                err = abs(actual - NEUTRAL)


                if err < 10: 
                    print(f" {label:10s} ID={sid} returned to {actual:.1f}")
                    passed.append(label)

                else:
                    print(f" {label:10s} ID={sid} returned to {actual:.1f}, drift detected")
                    passed.append(label)

            except Exception as e:
                torque_off(s)
                print(f" {label:10s} ID={sid} : Failed!")
                failed.append(label)



main()
