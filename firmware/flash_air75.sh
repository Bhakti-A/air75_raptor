#!/bin/bash
# Flash a Betaflight .hex to the Air75 (STM32G47x) over DFU WITHOUT touching its settings.
# The settings live in flash at 0x08004000-0x08006000 (FLASH_CONFIG in stm32_flash_g474.ld), inside the
# range a plain "objcopy -O binary + dfu-util" image covers -- writing that image wipes them (2026-10-21).
# This writes the two firmware parts around that region separately, like Betaflight Configurator does.
#
# usage: flash_air75.sh firmware.hex      (drone in DFU mode; radio unplugged or switched ON, never off+plugged)
#   AIR75_DFU_SERIAL=<serial>  your drone's DFU serial number (see `dfu-util -l` with the drone in DFU mode)
#   OBJCOPY=<path>             arm-none-eabi-objcopy, if it is not on PATH
set -e
HEX="$1"
[ -f "$HEX" ] || { echo "usage: $0 firmware.hex"; exit 1; }
OBJCOPY=${OBJCOPY:-$(command -v arm-none-eabi-objcopy || echo /opt/gcc-arm-none-eabi-10-2020-q4-major/bin/arm-none-eabi-objcopy)}
SERIAL=${AIR75_DFU_SERIAL:-366435593335}   # refuses to flash any other device (e.g. the Zorro's bootloader)
TMP=$(mktemp -d)
dfu-util -l 2>/dev/null | grep -q "serial=\"$SERIAL\"" || { echo "Air75 not in DFU mode (serial $SERIAL)"; exit 1; }
$OBJCOPY -I ihex -O binary --gap-fill 0xFF "$HEX" "$TMP/all.bin"
SIZE=$(stat -c %s "$TMP/all.bin")
head -c $((0x4000)) "$TMP/all.bin" > "$TMP/part0.bin"                         # 0x08000000 .. 0x08004000
tail -c +$((0x6000 + 1)) "$TMP/all.bin" > "$TMP/part1.bin"                    # 0x08006000 .. end
# the settings region must not carry firmware data in the hex
python3 - "$TMP/all.bin" <<'EOF'
import sys
b = open(sys.argv[1], 'rb').read()[0x4000:0x6000]
assert set(b) <= {0xFF}, "hex has data in the settings region 0x08004000-0x08006000; not flashing"
EOF
echo "firmware $SIZE bytes: part 0 $(stat -c %s $TMP/part0.bin) B @0x08000000, part 1 $(stat -c %s $TMP/part1.bin) B @0x08006000"
dfu-util -d 0483:df11 -S $SERIAL -a 0 -s 0x08006000 -D "$TMP/part1.bin" 2>&1 | grep -E "done|rror|successfully" || true
dfu-util -d 0483:df11 -S $SERIAL -a 0 -s 0x08000000:leave -D "$TMP/part0.bin" 2>&1 | grep -E "done|rror|successfully" || true
rm -rf "$TMP"
echo "flashed; settings region untouched. ('Error during download get_status' after the last part is normal: the drone rebooted.)"
