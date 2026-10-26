# Notes

What was measured and learned while getting RAPTOR to fly on the Air75 (September-October 2026). Each item has the
symptom, the cause, and what the code does about it now.

## Conventions

- **Room frame** (`room_layout.json`): origin on the floor under the middle marker column, x into the wall, y left, z up.
  The drone flies at negative x.
- **Heading frame**: what the laptop sends, x forward, y left, z up, with the drone's yaw removed. The firmware rotates
  it into the policy's target frame by the yaw of the attitude it sees. The FC's yaw estimate drifts, so the laptop
  never sends room-frame vectors.
- **Attitude**: Betaflight's `getQuaternion()` is already FLU (yaw + = nose left, same sign as gyro z). This was checked
  against the camera's attitude (logs 26-29) and with a shake test. No FRD->FLU conversion is needed.
- **Motor order**: the policy's actions are in Crazyflie order. Betaflight motor index = `{1, 0, 2, 3}[action index]`.
- **Props out**: the simulator spins props-in, while the Air75 flies with `yaw_motors_reversed = ON`. Without the mirror
  in `policy.cpp`, every yaw correction pushed the wrong way and the drone spun up on takeoff. This was the cause of all
  the yaw runaways before 2026-10-21.

## Position link

- **Channel mapping:** CH9-CH11 position error and CH12-CH14 velocity, full deflection = ±0.5 m / ±1.5 m/s. CH15 is
  valid + heartbeat: about 2012 and 1756 alternating per update, about 988 for "no fix". The ranges must match in
  `room_pose.py` (`SEND_RANGE_*`) and `policy.cpp` (`EXTERNAL_RANGE_*`, `MAX_*`).
- **Velocity clip:** the clip used to be ±0.3 m/s. The policy, trained without a clip, was then blind to the 0.8 m/s
  takeoff and overshot to 0.74 m.
- **Lua script throughput:** EdgeTX `serialRead()` without a size returns one line per call. The first `laptlm.lua`
  therefore handled about 22 lines a second, and commands queued up about 1 s late. It now drains everything that has
  arrived.
- **ELRS mode:** CH13-CH15 only reach the drone in a 16-channel full-resolution ELRS mode.
- **Short gaps:** gaps under `--hold-time` (0.2 s) send nothing. The radio keeps the last value and the firmware
  dead-reckons for up to 0.5 s. Sending "no fix" for every bad frame made the policy's input jump to zero and back.

## Camera and tracker

- **Delay:** about 112 ms from the camera to the laptop (`video_delay.py`, screen flashes). In flight the IMU vs camera
  height signals line up at 140-160 ms.
- **Marker size:** analog blur makes the detected squares about 8 % smaller than printed (14.7 vs 15.95 cm). The
  calibration fits this as `marker_size_eff`. Don't "correct" `room_layout.json` from images: that inflated the layout
  by up to 4.5 cm.
- **Decoding:** on the soft analog picture the squares were found but their bits not read. Reading only the cell
  centres (`tune_detector`) raised frames with at least 2 markers from 54 to 70 %.
- **Jump filter:** single-frame jumps of 15-55 cm, mostly from 2-marker poses, went to the policy as sudden errors.
  Now at least 3 markers may move up to 5 cm + 1.5 m/s × gap (max 35 cm), and 2 markers at most 15 cm. Anything else
  waits for two agreeing frames.
- **Wall optical flow** (`wall_flow.py`): LK tracking of wall features, back-projected onto the wall plane, then PnP.
  It is anchored on every ArUco pose. In-flight coverage rises from 30-58 % to 87-100 %, with about 1 cm median error
  against withheld ArUco poses.
  - Switching between flow and ArUco made 6.6 cm steps at every switch. `FusedPose` instead adds flow's
    frame-to-frame motion and pulls 15 % per frame towards each ArUco pose.
- **Distance:** beyond about 1.4 m from the wall the markers are too small to decode in flight (read rate 50 % at
  0.9-1.2 m, 25 % at 1.4-1.6 m).

## Reference and trim (`room_pose.py`, `Reference`)

- **Takeoff windup:** engaged on the floor with a target 0.3 m up, the recurrent policy wound up its thrust while the
  floor held it, then launched at about 1 m/s. The reference therefore starts at the drone and climbs at 0.15 m/s,
  moving sideways only after 10 cm.
- **Trim:** the policy has no integral term and hovered 8-17 cm off, mostly towards the wall. The trim is a laptop-side
  integral, shifting the sent reference against the mean error. It is saved after each hold (`last_trim.json`) and
  faded in during the next takeoff.
- **Landing:** the height trim must be off while landing. Otherwise the floor target can end up above the floor; the
  drone then winds up its thrust on the ground and bounces back up. Landing aims 12 cm below the floor, and once
  landed, 45 cm below, so it stays down.

## Firmware

- **Flashing:** the settings live inside the firmware image range (see `firmware/README.md`). Use `flash_air75.sh`.
- **Inference when off:** observation and inference ran in every loop even with RAPTOR off. In an A/B test against stock
  firmware, manual flight reset and flipped, and loop jitter disturbing the motor protocol was the suspected cause. It is
  now skipped when off.
- **Hover point:** `motor_output_center = 722` is the measured motor command in a stable manual hover. The output range
  is symmetric around it; giving the policy the full range upwards made it worse.
- **Runaway-takeoff check:** Betaflight's own check disarmed the drone 75 ms into a policy flight, because the
  bypassed PID saturated. It is paused only while RAPTOR is engaged.

## Simulator findings (`eval_external_pose`)

Hover sway in the last 2 s of 5 s episodes (spread around the episode's own mean, per axis), flashed policy:

| State the policy gets | Sway |
|---|---|
| perfect | 0.4 cm |
| 130 ms delay + on-board prediction with an ideal accelerometer | ~1 cm |
| + the Air75's accelerometer (noise 2.5 m/s², offset 0.5-2 m/s²) | 4.6 cm |
| + camera dropouts / bad fixes as measured in flight | ~9 cm |

- **Accelerometer noise:** most of the remaining hover sway comes from accelerometer noise integrated by the on-board
  prediction. In flight it measured 2.5 m/s² sideways and 4.2 m/s² vertical, mostly at 2-30 Hz, and the firmware takes
  one raw sample per 10 ms. Low-pass filtering it before integration is the obvious next step.
- **Prediction is needed:** without it, extrapolating only with the laptop's velocity, the policy crashed in every
  simulated episode.
- **Steady offset:** the policy settles about 14 cm from the target even with perfect state when a rotor is weak. That
  part is steady, and the laptop trim removes it.
- **Position scaling:** scaling the position input down (0.8 / 0.6 / 0.4) lowered hover speed (47 / 43 / 38 / 34 cm/s)
  but increased the distance (22 / 25 / 28 / 38 cm). In flight, 0.5 for x/y made the drone slower but let it wander
  further than 1.0.
- **Retraining:** the on-board prediction is what let the policy cope with the delay. With it, the base policy crashed in
  about 5 % of episodes, against nearly all without it. Retraining with the camera model (raw delay, then prediction
  with the real accelerometer, then dropouts) gave no clear further gain, e.g. with dropouts 23.4 vs 24.0 % crashes.
  `air75_delay_dropout` is flashed because it was at least as good.

Building and running the evaluation (it is not a CMake target):

```bash
# in rl-tools with patches/rl-tools.patch applied, after a CMake configure as in RAPTOR's training instructions
# (here: -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS=ON -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_EXPERIMENTAL=ON
#  -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_JSON=ON) in ./build
F=build/src/foundation_policy/CMakeFiles/foundation_policy_post_training.dir/flags.make
c++ $(grep CXX_DEFINES $F | cut -d= -f2-) $(grep CXX_INCLUDES $F | cut -d= -f2-) $(grep CXX_FLAGS $F | cut -d= -f2-) \
    -std=gnu++17 src/foundation_policy/post_training/eval_external_pose.cpp -o eval_external_pose -lhdf5 -lopenblas
# run in the folder with ./src/foundation_policy/checkpoints_<experiment>.txt and dynamics_parameters_<experiment>/
RAPTOR_TEACHER_EXPERIMENT=<experiment> EVAL_DROPOUTS=1 ./eval_external_pose checkpoint.h5 [more.h5 ...]
```

Environment variables:
- `EVAL_SKIP_BASIC`: skip the perfect-state and tracker runs.
- `EVAL_DROPOUTS`: real accelerometer, with and without dropouts.
- `EVAL_REAL_ACCEL`: offset-filter time constants, plus a velocity-only prediction.
- `EVAL_GAINS`: position scaling.
- `EVAL_IMBALANCE`: weak front-right rotor ×0.8 and battery ×0.93.
- `EVAL_WINDOW_START`: first step of the hover window (default 250 of 500).

`policy/dynamics_parameters/` and `policy/checkpoints_2026-09-28_17-48-14.txt` are the 300 training drones the
evaluation samples from. To use them, copy them to `src/foundation_policy/dynamics_parameters_2026-09-28_17-48-14/` and
`src/foundation_policy/checkpoints_2026-09-28_17-48-14.txt` under the folder you run in, and set
`RAPTOR_TEACHER_EXPERIMENT=2026-09-28_17-48-14`. The teacher checkpoints themselves are not included (78 MB).

## Hardware observations

- **Weak motor:** the front-right motor (Betaflight motor index 1) needs 1.13-1.3× the average command in every flight
  since 2026-10-21, a weak motor or bent prop. It saturates in hard manoeuvres.
- **Battery sag:** 1S packs sag to 2.6-3.0 V under the policy's thrust bursts. The analog video breaks up at the same
  moments, and those torn frames cost ArUco fixes.
- **Mechanical shocks:** single-sample gyro jumps of 200-500 °/s with 3-8 g accelerometer spikes appear 2-14 times per
  flight. They look like prop or duct strikes, not motor control.
