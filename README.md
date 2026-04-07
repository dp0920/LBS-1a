# LBS-1a

Quadruped robot ("Optimus Primal") — on-robot control code and PyBullet simulator.

## Layout

- `robot/` — code that runs on the physical robot (Raspberry Pi, LX-16A serial bus servos)
- `sim/` — PyBullet URDF simulator running on a desktop

## Robot

```
cd robot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python gait_controller.py            # WASD trot gait via keyboard
python gait_controller.py --measure  # 10-iteration crawl gait benchmark
```

> Raspberry Pi OS (Bookworm+) blocks system-wide pip (PEP 668), so the venv is required. If `python3 -m venv` fails, run `sudo apt install python3-venv python3-full` first.

`calibration.json` holds per-servo neutral angles for this specific robot.

## Sim (MuJoCo)

> Requires a display — runs on the desktop, not the headless Pi.


```
cd sim
pip install mujoco numpy
python mujoco_sim.py    # passive viewer, no control
python mujoco_trot.py   # trot gait demo with body position logging
```
