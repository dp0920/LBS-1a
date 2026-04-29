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

### Deploy a CMA gait to hardware
```bash
# mammalian
python3 gait_controller.py --gait best_gait.json --scale=2 --n=3
# ANYmal X-config (rear hip + rear knee flipped)
python3 gait_controller.py --gait cma_xconfig.json --xconfig --scale=2 --n=3
```
`--scale=N` slows phase time by Nx (good for first hardware test); `--n=N` is the number of stride cycles.

### Pose tools
```bash
python3 -i stance.py        # interactive stance tuner (xon() / xoff() / pos())
python3 pose_capture.py     # kinesthetic pose capture (per-leg or full-body)
python3 balance_playback.py --in=anymal_x_level.json --hold=3
```

## Sim (MuJoCo)

> Requires a display — runs on the desktop, not the headless Pi.

```
cd sim
pip install "mujoco>=3.2" cma scipy numpy matplotlib
mjpython mujoco_gait.py --replay best_gait.json    # replay CMA mammalian
mjpython mujoco_gait.py --replay cma_xconfig.json --xconfig   # X-config
mjpython view_xconfig.py                            # static X stance check
```

Full sim docs (training, replay flags, diagnostics) → [`sim/README.md`](sim/README.md).
Methodology + experimental record → [`sim/METHODS.md`](sim/METHODS.md).
Slide deck → [`sim/slides.pdf`](sim/slides.pdf).
