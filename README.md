# RAPTOR on a BetaFPV Air75 with wall-ArUco positioning

The [RAPTOR](https://arxiv.org/abs/2509.11481) foundation policy for quadrotor control ([rl-tools/raptor](https://github.com/rl-tools/raptor)),
retrained for a 75 mm 1S whoop (BetaFPV Air75 II) and flown inside Betaflight 4.6 on the drone itself. The policy needs
position and velocity, which a whoop has no sensor for. Here they come from ArUco markers on a wall, seen by the drone's
own analog FPV camera, computed on a laptop and sent up the normal RC link.

```
 FPV camera ──5.8 GHz video──> receiver ──USB capture──> laptop: position/room_pose.py
                                                           │ ArUco pose of the wall + optical flow on the wall
                                                           │ reference (takeoff / hold / circle / land) + trim
                                                           ▼
 Air75: Betaflight + policy.cpp <──ELRS CH9-15── Radiomaster Zorro: radio/laptlm.lua <──USB serial──┘
   │ position error, velocity (heading frame), heartbeat
   │ on-board delay compensation: accelerometer integrated forward over the ~130 ms the values are old
   ▼
 RAPTOR policy (GRU, 100 Hz)  ──>  4 motor commands
```

## Status

Takeoff, hover, slow circles and landing have been flown, 30-60 s per battery (October 2026). Honest numbers:

| | |
|---|---|
| Hover, horizontal error over a whole hold | 12-17 cm RMS |
| Hover after the trim has settled | 5-7 cm RMS per axis |
| Circle, radius 0.3-0.4 m, 15 s per lap | 9-16 cm RMS |
| ArUco pose in flight | 30-58% of frames (analog blur); with wall optical flow 87-100% |
| Video delay (camera -> laptop) | ~112 ms; position about 140-160 ms old when the policy uses it |

Most of the remaining hover sway is not the policy: in simulation the same policy sways 0.4 cm with perfect state,
about 1 cm with the real delay and an ideal on-board prediction, 4.6 cm once the prediction uses an accelerometer as noisy
as the Air75's (2.5 m/s² in flight), and about 9 cm with the camera's dropped frames on top. See [docs/NOTES.md](docs/NOTES.md).

This is experimental. Fly it in a net or an empty room, props off for every bench test, and keep your thumb on the
RAPTOR switch.

## Repository layout

| Path | What |
|---|---|
| `position/` | laptop side: `room_pose.py` (tracker, reference, sender), `wall_flow.py`, camera calibration, video-delay test, `room_layout.json`, `fpv_calib.npz`, `radio/laptlm.lua`, marker sheets |
| `firmware/` | `policy.cpp` (the Betaflight integration), ready-built `.hex` files, `flash_air75.sh`, settings backup + restore |
| `patches/` | everything changed in rl-tools, its Betaflight platform folder and Betaflight itself, against the upstream commits below |
| `policy/` | the two trained policies (`.h5` checkpoint + `.h` C header), the 300 training drones' dynamics |
| `sim/air75_demo.py` | fly a policy in the L2F simulator |
| `analysis/` | blackbox summary, wall-flow evaluation, tracker replay comparison |
| `examples/` | one recorded flight (raw FPV video + `room_pose` CSV) to replay the tracker without hardware |
| `docs/NOTES.md` | what was learned along the way: conventions, measurements, failures and their fixes |

## Hardware

- BetaFPV Air75 II (BETAFPVG473_V2_BMI270 flight controller, 1S, built-in ELRS receiver and VTX), props **out**
  (`yaw_motors_reversed = ON`).
- Radiomaster Zorro (EdgeTX 2.12) with an external ELRS module (3.6), in a 16-channel full-resolution mode so CH9-CH15
  reach the drone.
- 5.8 GHz video receiver with a USB (UVC) capture device.
- Six 16 cm ArUco markers (`DICT_4X4_50`, IDs 10-15) on one wall, printed from `position/markers/`.
- A Linux laptop (Python 3, OpenCV with `aruco`, `numpy`, `pyserial`).

## Upstream versions

| Repository | Commit |
|---|---|
| [rl-tools/raptor](https://github.com/rl-tools/raptor) | `4ef7566` |
| [rl-tools/rl-tools](https://github.com/rl-tools/rl-tools) (submodule `rl-tools`) | `3dea1bc8` |
| `rl-tools/embedded_platforms/betaflight` (submodule) | `b82378f` |
| `.../betaflight/betaflight` (Betaflight fork, submodule) | `cc2179f6f` |

## Setup

### 1. Markers and room frame

Print `position/markers/room_markers_A4.pdf` at 100 % (check the 100 mm bar) and put IDs 10-15 on a wall in two rows of
three. Measure the marker centres with a tape and write them into `position/room_layout.json`. The room frame: origin on
the floor under the middle column, x into the wall, y to the left when facing the wall, z up. The drone flies at
negative x.

### 2. Laptop

```bash
pip install -r requirements.txt
cd position
python3 aruco_view.py                 # live feed with marker detection: is the capture card found?
python3 calibrate_fpv_wall.py         # props off: carry the drone around in front of the wall for a minute
python3 video_delay.py                # optional: camera -> laptop delay
python3 room_pose.py                  # live pose in the room frame; t = measure the camera tilt (drone level)
```

`fpv_calib.npz` is the calibration of the camera this was developed with; redo it for yours.

### 3. Radio

Copy `position/radio/laptlm.lua` to `SCRIPTS/FUNCTIONS/` on the Zorro's SD card and add it as a Special Function (ON,
Lua script). Set SYS > Hardware > USB-VCP = LUA and plug the radio into the laptop as USB Serial (VCP). The script
publishes telemetry sensors `LPx LPy LPz LVx LVy LVz LOk`; use them as the sources of Inputs/mixes for CH9-CH15 in that
order (CH15 = valid + heartbeat). The RAPTOR engage switch goes on CH6 (AUX2), above 1750 µs = on.

### 4. Firmware

Either flash a ready-built hex from `firmware/hex/` or build it (see [firmware/README.md](firmware/README.md)), then:

```bash
cd firmware
./flash_air75.sh hex/betaflight_4.6.0_STM32G47X_BETAFPVG473_V2_BMI270_air75_dropout999_posgain07.hex
```

`flash_air75.sh` keeps the drone's settings; Betaflight Configurator's flasher wiped them on this board. Set
`debug_mode = RAPTOR` to get the policy's inputs in the blackbox.

### 5. Fly

```bash
cd position
python3 room_pose.py --send --record --pos-gain-xy 1.43 --pos-gain-z 1.43 --trim-tau 2
```

Arm in angle mode, then flip the RAPTOR switch: the drone takes off from where it sits to `--target`
(default `-1.2,0,0.3`, 1.2 m from the wall, 30 cm up). Keys in the window: `n` land, `g` take off again, `c` circle,
`i/k j/l u/o` move the target 5 cm, `q` quit. When it prints `LANDED`, switch RAPTOR off and disarm.
`--pos-gain-xy/z 1.43` undoes the 0.7 position scaling built into the `posgain07` firmware.

Replay a recording without hardware:

```bash
python3 room_pose.py --video ../examples/room_pose_20261022_194554.avi
```

## Training

The policy was retrained from scratch with RAPTOR's two-stage recipe (300 teacher policies, then distillation into one
recurrent student). Apply the patch to rl-tools at `3dea1bc8` inside a RAPTOR checkout at `4ef7566`:

```bash
git clone https://github.com/rl-tools/raptor.git && cd raptor && git checkout 4ef7566
git submodule update --init rl-tools data
cd rl-tools && git checkout 3dea1bc8
git apply /path/to/air75_raptor/patches/rl-tools.patch
```

then follow the "Training" section of the RAPTOR README. What the patch changes:

- **Pre-training** (`pre_training/sample_dynamics_parameters.cpp`): 300 drones instead of 1000, narrowed to 75 mm 1S
  whoops: mass 20-70 g, thrust-to-weight 3-8, torque-to-inertia 300-2000, arm length 36-44 mm, faster motors, smaller
  rotor torque constant (the original ranges are in the comments).
- **Post-training** reads `RAPTOR_TEACHER_EXPERIMENT` (your pre-training experiment name) to use your own teachers.
- **A camera model in the simulator** (`StateExternalPose` in the L2F environment, options in
  `post_training/config.h`, constants in `post_training/environment.h`). It covers the delay (90-170 ms), 30 fps camera
  and 20 Hz radio updates, noise, the firmware's on-board prediction with a realistic accelerometer, camera dropouts and
  bad fixes, and the firmware's position scaling.
- **Motor imbalance** (`post_training/main.cpp`, `MOTOR_IMBALANCE`): a random weak rotor and battery sag.
- `eval_external_pose.cpp` compares checkpoints in all these simulator variants (build command in [docs/NOTES.md](docs/NOTES.md)).

Policies in `policy/`:

| | Trained with | Use |
|---|---|---|
| `air75_base` (2026-09-29) | narrowed pre-training, original post-training (perfect state) | the simulator demo; was flown first |
| `air75_delay_dropout` (2026-10-22) | + camera model with prediction, real accelerometer and dropouts (`MOTOR_IMBALANCE = false`) | in the `dropout999*.hex` firmware, flown |

A third run with `MOTOR_IMBALANCE = true` scored the same as `air75_delay_dropout` in simulation and was not flown. The
patch has `MOTOR_IMBALANCE = true`; set it to `false` to reproduce `air75_delay_dropout`.

## Credits and third-party licences

This repository has no licence of its own. It builds on other people's work, whose licences are in
[`third_party_licenses/`](third_party_licenses):

- **RAPTOR** and **rl-tools** by Jonas Eschmann et al., MIT licence (`raptor-MIT.txt`, `rl-tools-MIT.txt`). The
  patches, `firmware/policy.cpp` and the policy files contain or are derived from their code. Please cite the RAPTOR
  paper if you use this: *RAPTOR: A Foundation Policy for Quadrotor Control*, arXiv:2509.11481.
- **Betaflight**, GPL-3.0 (`betaflight-GPL-3.0.txt`). `patches/betaflight.patch` and the firmware built with it
  (`firmware/hex/`, which includes `firmware/policy.cpp`) fall under the GPL-3.0. The complete source for those files is
  the upstream commits listed above plus the patches in this repository.
- **OpenCV** for the ArUco marker detection.
