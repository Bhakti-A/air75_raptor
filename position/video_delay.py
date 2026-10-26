"""Video delay test: laptop screen flashes black/white, the FPV camera watches it, the delay is timed.

Measures screen -> FPV camera -> video transmitter -> receiver -> capture card -> OpenCV frame, i.e. how
old a frame is when the tracker gets it. Includes the laptop screen's own lag (about 10-30 ms), so the
real camera-side delay is a little lower than reported.

Props off, drone powered, video on. Turn the screen brightness up, hold the drone so the FPV camera is
10-30 cm from the screen and the screen fills most of the picture, keep it still, then run:
    python3 video_delay.py
The window goes fullscreen and flashes --flashes times (about 30 s). Esc stops early.
Prints the delay (time until the average brightness change reaches half its full size) and saves the
raw frame brightness and flash times as video_delay_<time>_frames.csv / _flips.csv.
"""
import argparse
import csv
import random
import threading
import time

import cv2
import numpy as np

from aruco_view import find_capture_device


class Grabber(threading.Thread):
    """Reads frames as fast as they come; keeps (time read, mean brightness) of each one."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.samples, self.running = cap, [], True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.samples.append((time.monotonic(), float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())))


def step_response(flips, samples, taus):
    """Average brightness change after a flash, signed so both directions add up, at each delay in `taus`.

    The brightness at time t is that of the last frame read by then: what the tracker would have.
    Returns (both directions, to white only, to black only), each relative to just before the flash.
    """
    t = np.array([s[0] for s in samples])
    b = np.array([s[1] for s in samples])
    at = lambda q: b[np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(b) - 1)]
    curves = {True: [], False: []}
    for tf, white in flips:
        if tf - 0.05 < t[0] or tf + taus[-1] > t[-1]:
            continue
        sign = 1.0 if white else -1.0
        curves[white].append(sign * (at(tf + taus) - at(tf - 0.05)))
    mean = lambda c: np.mean(c, axis=0) if c else np.zeros(len(taus))
    return mean(curves[True] + curves[False]), mean(curves[True]), mean(curves[False]), len(curves[True]) + len(curves[False])


def half_rise(curve, taus):
    """First delay at which the response reaches half of its peak, or None if there is no clear response."""
    peak = curve.max()
    if peak < 5:
        return None
    return float(taus[np.argmax(curve >= peak / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--flashes", type=int, default=40)
    args = ap.parse_args()

    dev = args.device or find_capture_device()
    if not dev:
        raise SystemExit("No capture card found")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    grab = Grabber(cap)
    grab.start()

    win = "video delay"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black, white = np.zeros((720, 1280, 3), np.uint8), np.full((720, 1280, 3), 255, np.uint8)
    cv2.imshow(win, black)
    cv2.waitKey(1500)                                 # let the camera's exposure settle on the black screen

    flips, state = [], False
    for _ in range(args.flashes):
        state = not state
        cv2.imshow(win, white if state else black)
        cv2.waitKey(1)                                # the frame goes to the screen here
        flips.append((time.monotonic(), state))
        if cv2.waitKey(int(random.uniform(400, 900))) & 0xFF == 27:   # random gap: no lock-in with the 25 fps video
            break
    time.sleep(0.5)
    grab.running = False
    grab.join(1.0)
    cap.release()
    cv2.destroyAllWindows()

    taus = np.arange(0.0, 0.8, 0.002)
    both, to_white, to_black, n = step_response(flips, grab.samples, taus)
    name = time.strftime("video_delay_%Y%m%d_%H%M%S")
    with open(name + "_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "brightness"])
        w.writerows([["%.4f" % t, "%.1f" % v] for t, v in grab.samples])
    with open(name + "_flips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "to_white"])
        w.writerows([["%.4f" % t, int(v)] for t, v in flips])
    fps = (len(grab.samples) - 1) / (grab.samples[-1][0] - grab.samples[0][0]) if len(grab.samples) > 1 else 0
    print("frames %d at %.1f fps, %d flashes used" % (len(grab.samples), fps, n))
    print("brightness change after a flash: to white %+.0f, to black %+.0f (out of 255)" % (to_white.max(), -to_black.max()))
    for label, curve in (("both", both), ("to white", to_white), ("to black", to_black)):
        d = half_rise(curve, taus)
        print("  delay %-8s %s" % (label, "no clear response" if d is None else "%.0f ms" % (1000 * d)))
    if both.max() < 15:
        print("the flashes barely show in the video: get the camera closer so the screen fills the picture")
    print("saved %s_frames.csv and %s_flips.csv" % (name, name))


if __name__ == "__main__":
    main()
