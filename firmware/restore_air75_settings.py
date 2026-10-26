"""Restore the Air75's Betaflight settings from a CLI backup (+ debug_mode RAPTOR) and print the diff.

usage: python3 restore_air75_settings.py [check] [backup.txt]   ('check' only prints the current diff)
The default backup is air75_cli_backup.txt next to this script: the settings of the drone this was developed on
(accelerometer calibration included) -- make your own with Betaflight's `diff all` and use that instead.
"""
import os, serial, time, glob, subprocess, sys
BACKUP = next((a for a in sys.argv[1:] if a.endswith('.txt')),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'air75_cli_backup.txt'))
def find_port():
    for d in sorted(glob.glob('/dev/ttyACM*')):
        if 'Betaflight' in subprocess.run(['udevadm', 'info', '-q', 'property', '-n', d], capture_output=True, text=True).stdout:
            return d
    sys.exit('drone not found on USB')
def cli(port, cmds, wait=0.4):
    s = serial.Serial(port, 115200, timeout=0.2); time.sleep(0.5); s.reset_input_buffer()
    s.write(b'#'); time.sleep(1.5); s.read(5000)
    out = b''
    for c in cmds:
        s.write((c + '\n').encode())
        t0 = time.time()
        while time.time() - t0 < (wait if c not in ('save', 'diff') else 3):
            try: out += s.read(5000)
            except serial.SerialException: return out.decode(errors='replace')  # rebooted after save
    try: s.write(b'exit\n'); s.close()
    except Exception: pass
    return out.decode(errors='replace')
if __name__ == '__main__':
    lines = [l.rstrip('\n') for l in open(BACKUP) if l.strip() and not l.startswith('#')]
    lines.insert(lines.index('save'), 'set debug_mode = RAPTOR')
    if 'check' not in sys.argv:
        out = cli(find_port(), lines)
        print('errors:', [l for l in out.splitlines() if 'ERROR' in l] or 'none')
        time.sleep(6)
    out = cli(find_port(), ['diff'])
    print('\n'.join(l for l in out.splitlines() if l.startswith(('aux', 'set ', 'feature'))))
