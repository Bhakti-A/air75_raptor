"""Laptop -> Radiomaster Zorro (running radio/laptlm.lua as a Special Function) over USB serial.

Used by room_pose.py to send the position error / velocity, which laptlm.lua puts on CH9..CH15 for the drone.
It can also send stick commands (sticks=True), which only take effect while the laptop-control switch (SA) is on.
The Zorro must be in USB Serial (VCP) mode with SYS > Hardware > USB-VCP = LUA.
Arming always stays on the pilot's own switch.
"""
import argparse
import glob
import math
import subprocess
import threading
import time

import serial


def find_zorro_port():
    """Return the /dev/ttyACM* device of the Zorro (its number changes on every replug)."""
    for dev in sorted(glob.glob("/dev/ttyACM*")):
        props = subprocess.run(["udevadm", "info", "-q", "property", "-n", dev],
                               capture_output=True, text=True).stdout
        if "Zorro" in props:
            return dev
    raise SystemExit("Zorro not found - is it plugged in as USB Serial (VCP)?")


class ZorroLink:
    """Streams roll/pitch/throttle/yaw (each -1.0..1.0) and/or room positions to the radio at a fixed rate.

    The old radio script handled only one line per run, about 22 lines per second (measured 2026-10-02),
    and extra lines queued up so every command arrived about 1 s late. laptlm.lua now reads everything
    that has arrived (2026-10-17); sending still pauses if the radio falls behind anyway.

    sticks=False sends no stick lines at all, so the radio keeps the pilot's sticks on CH1..CH4.
    Positions (send_position) go out once per new value, never repeated: the radio's heartbeat on
    CH15 only flips when new data arrives, so a stalled camera stops looking valid to the drone.
    """

    def __init__(self, port=None, rate_hz=15, sticks=True):
        self.ser = serial.Serial(port or find_zorro_port(), 115200, timeout=0)
        self.period = 1.0 / rate_hz
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0, -1.0, 0.0)   # centred sticks, throttle low
        self.sticks = sticks
        self.pos = None         # (px, py, pz, vx, vy, vz, ok) in -1024..1024, waiting to be sent
        self.running = True
        self.rx_buf = b""
        self.status = None      # latest status line from laptlm.lua, see _parse_status
        self.status_time = 0.0
        self.sent = 0           # bytes written to the radio
        self.ref = None         # (sent, radio rx_bytes) at the first status line
        self.backlog = 0        # bytes the radio has not read yet (estimate, updated with each status)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set(self, roll=0.0, pitch=0.0, throttle=-1.0, yaw=0.0):
        with self.lock:
            self.cmd = (roll, pitch, throttle, yaw)

    def send_position(self, values):
        """values: position error x, y, z and velocity x, y, z in -1..1 of full range, or None = no fix."""
        if values is None:
            line = (0, 0, 0, 0, 0, 0, -1024)
        else:
            line = tuple(max(-1024, min(1024, int(round(v * 1024)))) for v in values) + (1024,)
        with self.lock:
            self.pos = line

    def get_status(self, max_age=1.0):
        """Latest radio status dict, or None if nothing arrived within max_age seconds."""
        with self.lock:
            if self.status and time.time() - self.status_time < max_age:
                return dict(self.status)
        return None

    def _parse_status(self, line):
        # "S<bytes received>,<ch5 arm>,<ch7 mode>,<ch8 laptop switch>,<FM telemetry>[,<ch3 throttle out>,<throttle stick>,<ch1 out>,<ch2 out>,<ch4 out>[,<ch6 RAPTOR switch>,<ch9 out>,<ch15 out>]]"
        if not line.startswith("S"):
            return
        parts = line[1:].split(",")
        if len(parts) not in (5, 10, 13):
            return
        try:
            rx, ch5, ch7, ch8 = (int(p) for p in parts[:4])
            ch3, thr, ch1, ch2, ch4 = (int(p) for p in parts[5:10]) if len(parts) >= 10 else (None,) * 5
            ch6, ch9, ch15 = (int(p) for p in parts[10:13]) if len(parts) == 13 else (None,) * 3
        except ValueError:
            return
        if self.ref is None or rx < self.ref[1]:        # first status, or the radio script restarted
            self.ref = (self.sent, rx)
        self.backlog = (self.sent - self.ref[0]) - (rx - self.ref[1])
        with self.lock:
            # ch1..ch4 are what the radio really transmits (after its mixer), -1024..1024
            self.status = {"rx_bytes": rx, "ch5": ch5, "ch7": ch7, "ch8": ch8, "fm": parts[4].strip(),
                           "ch3": ch3, "thr": thr, "ch1": ch1, "ch2": ch2, "ch4": ch4,
                           "ch6": ch6, "ch9": ch9, "ch15": ch15}
            self.status_time = time.time()

    def _loop(self):
        while self.running:
            with self.lock:
                vals = [max(-1024, min(1024, int(v * 1024))) for v in self.cmd]
                pos, self.pos = self.pos, None
            out = ""
            if self.sticks:
                out += "C%d,%d,%d,%d\n" % tuple(vals)
            if pos is not None:
                out += "P%d,%d,%d,%d,%d,%d,%d\n" % pos
            try:
                if out and self.backlog < 150:      # radio keeping up (normal: under ~60 bytes in flight)
                    self.sent += self.ser.write(out.encode())
                self.rx_buf += self.ser.read(512)
                *lines, self.rx_buf = self.rx_buf.split(b"\n")
                for line in lines:
                    self._parse_status(line.decode(errors="replace").strip())
                self.rx_buf = self.rx_buf[-256:]
            except (serial.SerialException, OSError):
                self._reconnect()
            time.sleep(self.period)

    def _reconnect(self):
        # The radio re-enumerates (often as a new ttyACM number) after a restart or replug.
        print("Zorro disconnected, waiting for it to come back...", flush=True)
        try:
            self.ser.close()
        except (serial.SerialException, OSError):
            pass
        while self.running:
            time.sleep(1)
            try:
                self.ser = serial.Serial(find_zorro_port(), 115200, timeout=0)
                self.ref, self.backlog = None, 0
                print("Zorro reconnected on", self.ser.port, flush=True)
                return
            except (SystemExit, serial.SerialException, OSError):
                continue

    def close(self):
        # Stop sending; after 300 ms the radio falls back to the pilot's sticks.
        self.running = False
        self.thread.join()
        self.ser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bench test: slow roll sweep, throttle low. PROPS OFF.")
    ap.add_argument("--port", default=None, help="default: auto-detect Zorro")
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()

    link = ZorroLink(args.port)
    print("Sending test sweep for %.0fs (Ctrl-C to stop). Flip SA to see it take effect." % args.seconds)
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            link.set(roll=0.5 * math.sin(time.time() - t0), throttle=-1.0)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        print("Stopped; radio is back on your sticks.")
