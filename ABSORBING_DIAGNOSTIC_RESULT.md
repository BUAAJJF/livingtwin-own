# Absorbing-failure Stage 1 diagnostic

**Status: smoke complete; stopped before formal training.**

The original representative return order was success > out-of-bounds > valid timeout under both undiscounted and gamma=0.99 discounted return. With the frozen -1.262 absorbing reward, it becomes success > valid timeout > out-of-bounds.

The absorbing state uses an all-zero 15-D observation, frozen MuJoCo state, ignored actions, no done mask through step 199, and done/zero bootstrap only at step 200.

| Initial yaw error | Correct rotation | Simultaneous improvement | Heuristic success |
|---|---:|---:|---:|
| 0-45° | 84.4% | 75.0% | 43.8% |
| 45-90° | 71.9% | 65.6% | 0.0% |
| 90-135° | 75.0% | 71.9% | 0.0% |
| 135-180° | 81.2% | 84.4% | 0.0% |

Smoke common: joint 0.0%, position-only 0.0%, yaw-only 12.5%, failure 0.0%, position error 0.102609 m, yaw error 25.25°, discounted return -22.610.

No out-of-bounds event occurred during stochastic smoke collection or deterministic evaluation. The old smoke also had no out-of-bounds event, so identical task metrics are expected; the shortcut appeared only in longer old formal runs.

Recommendation: a new preregistered three-seed absorbing-failure Stage 1 is justified to test the fix at the budget where the shortcut previously emerged. Do not modify any other task or PPO setting, and do not enter Stage 2 yet.
