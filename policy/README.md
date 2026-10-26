# Trained policies

| Folder | Date | Training |
|---|---|---|
| `air75_base/` | 2026-09-29 | RAPTOR recipe with the narrowed whoop range (300 teachers), perfect-state post-training |
| `air75_delay_dropout/` | 2026-10-22 | `air75_base` teachers, post-trained with the camera model: delay, on-board prediction, real accelerometer, dropouts |

Each folder holds:
- `checkpoint.h5`: rl-tools checkpoint. Load it with `foundation_policy.load(...)`, `sim/air75_demo.py` or
  `eval_external_pose`.
- `checkpoint.h`: the same weights as a C header. Copy it to Betaflight's `src/rl_tools/blob/policy.h` to build
  firmware.

`dynamics_parameters/` has the 300 sampled training drones (`<n>.json`). `6.json` is the closest to the Air75 II:
41.9 g, 38.8 mm arm, thrust-to-weight 4.9. `checkpoints_2026-09-28_17-48-14.txt` is the list of teachers that
post-training and `eval_external_pose` read. The teacher checkpoints themselves are not included.
