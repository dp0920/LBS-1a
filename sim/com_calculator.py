#!/usr/bin/env python3
"""
Center of Mass calculator for Optimus Primal body layout.

Coordinate system (looking down at the robot from above):
  +x = forward (toward front legs)
  -x = backward (toward rear legs)
  +y = left
  -y = right
  Origin = geometric center of the body frame

Usage:
  1. Weigh each component on a kitchen scale
  2. Measure where each component's center sits relative to the body center
  3. Update the numbers below
  4. Run: python com_calculator.py
  5. Move components until CoM_x and CoM_y are ~0
"""

# ============================================================
# COMPONENTS — update these with your measurements
# ============================================================
# Format: (name, mass_grams, x_mm, y_mm)
#   x_mm: positive = forward, negative = rearward
#   y_mm: positive = left, negative = right

components = [
    # Structural
    ("Frame (PETG-CF)",       80,     0,    0),   # roughly centered

    # Power
    ("Battery (2S LiPo)",    150,     0,    0),   # centered between hips

    # Electronics — these are what you're repositioning
    ("Raspberry Pi",          45,   -50,    0),   # currently ~50mm behind center
    ("Buck converter",        20,   -40,    0),   # near the Pi
    ("Motor controller",      15,   -30,    0),   # servo bus board

    # Servos (8 total, mounted at hip joints)
    # These are part of the legs but their mass matters for the body CoM
    # Hip positions from URDF: FL/FR at x=+90.8mm, RL/RR at x=-90.8mm
    ("Servo FL hip",          55,    91,   66),
    ("Servo FR hip",          55,    91,  -66),
    ("Servo RL hip",          55,   -91,   80),
    ("Servo RR hip",          55,   -91,  -80),
    ("Servo FL knee",         55,    91,   66),   # approximate — hangs below hip
    ("Servo FR knee",         55,    91,  -66),
    ("Servo RL knee",         55,   -91,   80),
    ("Servo RR knee",         55,   -91,  -80),

    # Wiring, standoffs, misc
    ("Wiring + misc",         30,   -10,    0),
]

# ============================================================
# CALCULATION
# ============================================================
total_mass = sum(m for _, m, _, _ in components)
com_x = sum(m * x for _, m, x, _ in components) / total_mass
com_y = sum(m * y for _, m, _, y in components) / total_mass

print(f"{'Component':<25s} {'Mass (g)':>8s} {'X (mm)':>8s} {'Y (mm)':>8s}")
print("-" * 53)
for name, m, x, y in components:
    print(f"{name:<25s} {m:8.0f} {x:8.1f} {y:8.1f}")
print("-" * 53)
print(f"{'TOTAL':<25s} {total_mass:8.0f}")
print()
print(f"  Center of Mass:  X = {com_x:+.1f} mm,  Y = {com_y:+.1f} mm")
print()

if abs(com_x) < 2 and abs(com_y) < 2:
    print("  --> Nicely centered!")
else:
    if abs(com_x) > 2:
        direction = "forward" if com_x > 0 else "rearward"
        print(f"  --> CoM is {abs(com_x):.1f} mm {direction} of center")
    if abs(com_y) > 2:
        direction = "left" if com_y > 0 else "right"
        print(f"  --> CoM is {abs(com_y):.1f} mm {direction} of center")

    # Suggest fix: how far to move the Pi to zero out X
    pi_mass = next(m for name, m, _, _ in components if "Pi" in name)
    if abs(com_x) > 2:
        shift_needed = -(com_x * total_mass) / pi_mass
        print(f"\n  To zero CoM_x by moving only the Pi:")
        pi_x = next(x for name, _, x, _ in components if "Pi" in name)
        print(f"    Move Pi from x={pi_x:+.0f} mm to x={pi_x + shift_needed:+.0f} mm "
              f"({shift_needed:+.1f} mm)")
