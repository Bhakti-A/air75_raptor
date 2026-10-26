# Firmware: RAPTOR inside Betaflight 4.6 on the Air75

`policy.cpp` runs the policy in Betaflight's loop (rl-tools' `embedded_platforms/betaflight`, which hooks
`rl_tools_control()` into the mixer). On top of the upstream integration it adds:

- **Position link:** reads the laptop's position error and velocity from CH9-CH14 and the valid + heartbeat flag from
  CH15 (`update_external`). The values arrive in the drone's heading frame and are rotated into the target frame.
- **On-board delay compensation** (`predictor_update`): each laptop value is taken as 13 ticks (130 ms) old and
  integrated forward with the accelerometer. A running-mean offset filter (`predictor_offset_update`) is applied, and
  the estimate dead-reckons through camera gaps of up to 0.5 s. The constants must match
  `post_training/environment.h`.
- **Props-out yaw fix** (`PROPS_OUT`): the simulator's rotors spin props-in, so the yaw component of the command is
  mirrored for `yaw_motors_reversed = ON`.
- **Safety:** soft start on engage, yaw authority clamp and a yaw-rate cutoff. Runaway-takeoff detection is paused
  while the policy flies.
- **Position scaling:** `POSITION_OBSERVATION_GAIN = 0.7` (the `posgain07` hex) makes the hover calmer.
- **Engage switch:** AUX2 (CH6) above 1750 µs. The heading at engage is held.
- **Blackbox:** `debug_mode = RAPTOR` logs the attitude the policy sees in debug 0-3, the laptop position in mm in
  debug 4-6, and the predicted x in debug 7.

## Ready-built

| `hex/` | Policy | Position gain |
|---|---|---|
| `..._air75_dropout999_posgain07.hex` | `policy/air75_delay_dropout` | 0.7 (fly with `room_pose.py --pos-gain-xy 1.43 --pos-gain-z 1.43`) |
| `..._air75_dropout999.hex` | `policy/air75_delay_dropout` | 1.0 |

Target `BETAFPVG473_V2_BMI270` (STM32G473) only.

## Build

```bash
# in a RAPTOR checkout with rl-tools at 3dea1bc8 and patches/rl-tools.patch applied (policy.cpp needs its ARM matrix ops)
cd rl-tools
git submodule update --init embedded_platforms/betaflight
cd embedded_platforms/betaflight
git submodule update --init betaflight src/rl_tools/blob
git apply /path/to/air75_raptor/patches/rl-tools-betaflight-platform.patch
(cd betaflight && git apply /path/to/air75_raptor/patches/betaflight.patch)
cp /path/to/air75_raptor/policy/air75_delay_dropout/checkpoint.h src/rl_tools/blob/policy.h

cd betaflight
make arm_sdk_install
make CONFIG=BETAFPVG473_V2_BMI270 RL_TOOLS_ROOT=$(cd ../../.. && pwd) -j$(nproc)
# -> obj/betaflight_4.6.0_STM32G47X_BETAFPVG473_V2_BMI270.hex
```

After swapping `blob/policy.h`, delete `obj/main/STM32G47X_BETAFPVG473_V2_BMI270/rl_tools/policy.o`. The build does not
track the header, so it would otherwise say "up to date" and keep the old weights.

## Flash

```bash
./flash_air75.sh <file.hex>      # drone in DFU mode: Betaflight CLI `bl`, or hold the boot button while plugging in
```

The G47x keeps its settings in flash at 0x08004000-0x08006000, inside the firmware image. Writing the image in one
piece, as `dfu-util` with a plain binary or a failed Betaflight Configurator attempt does, wipes them. This script
writes the two parts around that region instead.

Set `AIR75_DFU_SERIAL` to your drone's DFU serial (`dfu-util -l`). The script refuses to flash any other DFU device:
a Zorro plugged in while switched off also shows up as an STM32 bootloader.

If the settings are gone, use `python3 restore_air75_settings.py [check] [backup.txt]`. `air75_cli_backup.txt` holds
the settings of the drone this was developed on, including its accelerometer calibration and board alignment. Use your
own `diff all` for yours.
