# LBS-1a

Quadruped robot ("Optimus Primal") — on-robot control code and PyBullet simulator.

## Layout

- `robot/` — code that runs on the physical robot (Raspberry Pi, LX-16A serial bus servos)
- `sim/` — PyBullet URDF simulator running on a desktop

## Robot

```
cd robot
pip install -r requirements.txt
python gait_controller.py            # WASD trot gait via keyboard
python gait_controller.py --measure  # 10-iteration crawl gait benchmark
```

`calibration.json` holds per-servo neutral angles for this specific robot.

## Sim (MuJoCo)

> Requires a display — runs on the desktop, not the headless Pi.


```
cd sim
pip install mujoco numpy
python mujoco_sim.py    # passive viewer, no control
python mujoco_trot.py   # trot gait demo with body position logging
```
