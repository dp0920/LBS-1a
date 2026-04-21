# Optimus Primal — Gait Optimization Slides

Slides 1 (Problem) and 2 (Digital Twin) are done. Content below covers slides 3–8.

---

## Slide 3: Search Space — 105 Parameters

- Gait structure is fixed: 13 phases in a crawl cycle
  - `start` then `{shift, swing, plant}` for each of `{FR, RL, FL, RR}`
  - Optimizer tunes the **angles at each phase**, not the sequencing or leg order
- **104 joint-angle parameters:** 13 phases x 4 legs x 2 joints (hip + knee), in degrees
- **+1 phase-time parameter:** seconds between phases — controls gait cadence
- Bounds enforced by the optimizer:
  - Hip: 5° to 55° (forward swing range)
  - Knee: -100° to -25° (more negative = more crouched)
  - Phase time: 0.15s to 1.2s
- **Seeded from the hand-tuned gait**, not random — generation 1 already walks, CMA-ES refines from there
- **Sim-to-real is a parameter copy** — the sim uses the exact same degree convention as the real robot's motor controller, so the winning JSON loads directly onto hardware with no conversion

---

## Slide 4: Algorithm — CMA-ES

**CMA-ES: Covariance Matrix Adaptation Evolution Strategy**

A gradient-free, population-based optimizer. The standard choice for robot gait tuning when you have a simulator but no differentiable model.

**Each generation:**
1. Sample a population of candidate gaits (48 per generation) from a multivariate Gaussian distribution
2. Evaluate every candidate in MuJoCo — in parallel across CPU cores
3. Rank by reward, keep the top half
4. Update the distribution:
   - **Mean** moves toward the winning region
   - **Covariance matrix** stretches along directions where good solutions cluster, shrinks across directions that don't matter — this is the key trick, it learns the *shape* of the search landscape
   - **Step size** grows or shrinks based on progress

**Why not a hill climber?**
- Hill climber: 1 candidate per step, no memory of directions, gets stuck on cliffs
- CMA-ES: 48 candidates per step, learns coordinate correlations, jumps over discontinuities
- Our reward has sharp cliffs (robot falls = rollout terminates instantly) — CMA-ES handles this well

**Engineering details:**
- Population size: 48, initial step size: 5.0, 150 generations default
- Auto-restart on stall: if best reward doesn't improve for 15 generations, reinitialize with larger step size and bigger population — escapes local minima
- Parallel evaluation across CPU cores (7 workers on 8-core Mac, ~7x speedup)
- Checkpoints every generation — `best_gait.json` always has the current best

---

## Slide 5: Reward Function

**What the optimizer is maximizing — shaped iteratively through failure.**

### The formula

| Component | Weight | Purpose |
|---|---|---|
| Forward distance | +100 per meter | Primary goal: walk forward |
| Phases completed | +0.5 per phase | Survival bonus — stay alive longer |
| Mean body height | +5 | Don't crouch or collapse |
| Backward distance | -200 per meter | Don't go the wrong way |
| Mean pitch squared | -30 | Stay level (front-to-back) |
| Mean roll squared | -40 | Stay level (side-to-side) |
| Peak pitch/roll squared | -8 each | No momentary big tilts |
| **Mean nose-down squared** | **-200** | **Don't flop forward** |
| **Peak nose-down squared** | **-60** | **Not even once** |
| Mean pitch-rate squared | -2 | No fast floppy rotations |
| Peak pitch-rate squared | -0.5 | No sudden snaps |

### Hard terminations (rollout ends, no more reward possible)
- Body height drops below 8 cm
- Any tilt exceeds 30 degrees
- Nose points down past ~27 degrees

### The iteration story — this wasn't designed, it was discovered

1. **v1: Distance only.** Robot learned to collapse onto its shins and scoot forward. Technically moved — not a gait.
2. **v2: Added symmetric tilt penalty.** Tightened fall threshold from 45 to 30 degrees. Robot balanced better but still toppled forward during swing phases.
3. **v3: Added directional nose-down penalty and pitch-rate penalty.** Targets the specific failure mode (forward flop) without penalizing backward lean, which is sometimes useful for weight shifting before a step.

Each version was shaped by watching what the optimizer exploited, then closing that loophole. The reward function is as much the product of learning as the gait itself.

---

## Slide 6: Results — Seed Comparison

**Does the starting point matter? Yes.**

Three CMA-ES runs, same reward function, different seeds:
- **Hand-tuned gait** (`--init gait`): seeded from the manually-tuned `stance.py` crawl
- **Random** (`--init random`): uniform random within bounds
- **Neutral standing** (`--init stand`): every phase pinned to the standing pose (no motion)

The hand-tuned seed starts from a working gait and refines. Random has to discover walking from scratch. Standing starts from a plausible pose but with no motion.

**Key takeaway:** seeding from a human-tuned gait gives a massive head start — generation 1 is already walking, so the optimizer spends its budget refining instead of discovering.

*(To run the missing standing seed:)*
```bash
python mujoco_gait.py --tune --init stand --generations 150
```

*(Generate the comparison convergence plot with:)*
```bash
python -c "
import json, matplotlib.pyplot as plt, os

runs = {
    'Hand-tuned seed': 'tune_gait.jsonl',
    'Random seed': 'tune_gait_random.jsonl',
    'Standing seed': 'tune_gait_stand.jsonl',
}

plt.figure(figsize=(10, 5))
for label, path in runs.items():
    if not os.path.exists(path):
        print(f'Skipping {label} — {path} not found')
        continue
    with open(path) as f:
        gens = [json.loads(l) for l in f]
    plt.plot([g['gen'] for g in gens], [g['best'] for g in gens], label=label)
plt.xlabel('Generation')
plt.ylabel('Best Reward')
plt.title('CMA-ES Convergence — Effect of Initialization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=150)
print('Saved convergence_comparison.png')
"
```

- Also include: side-by-side video of MuJoCo sim replay next to real robot running the same gait
- Key numbers from the best gait (distance per stride, survival rate, tilt metrics)

---

## Slide 7: Limitation — Open-Loop Control

**CMA-ES found the best choreography. But it's still a choreography.**

- The output is a **fixed lookup table**: 13 phases, 8 angles each, played back identically every stride
- The robot has **no sensors, no feedback, no ability to react**
- If it hits a bump, drifts off-balance, or the floor is slightly uneven — it doesn't know
- Works in sim (perfect flat floor, zero disturbances) but struggles on real hardware (servo slop, surface variation, weight shifts)

**The fundamental gap:**

| | CMA-ES (current) | What we actually need |
|---|---|---|
| Output | 105 fixed numbers | A function that reacts |
| Sensors at runtime | None — blind playback | IMU, joint positions |
| Adapts to disturbance | No | Yes |
| What it learned | "The best script" | "How to walk" |

---

## Slide 8: Next — Reinforcement Learning

**Close the loop: from choreography to skill.**

RL trains a neural network (a "policy") that takes in sensor readings every timestep and outputs motor commands:

```
observe → decide → act → observe → decide → act → ...
```

Instead of "at phase 3, set FL hip to 34 degrees," the policy says "given that I'm currently tilted 5 degrees forward and my left knee is lagging, here's what to do."

**What we reuse from this work:**
- The MuJoCo digital twin — same sim
- The reward function — same objectives, already well-shaped through iteration
- The CMA-ES gait — baseline to beat, and can seed the initial policy

**What's new:**
- Observation space: IMU (pitch, roll, angular velocity) + 8 joint positions + 8 joint velocities
- Algorithm: PPO (Proximal Policy Optimization) — the standard for legged locomotion
- Output: a neural network that deploys to the robot, not a JSON of fixed angles

**The punchline:**
> CMA-ES found the best choreography for a perfect floor. RL will learn to actually walk — adjusting every step based on what the robot feels.
