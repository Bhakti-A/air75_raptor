"""Live FPV feed from the USB capture card with ArUco marker detection.

Keys in the window: q = quit, s = save snapshot.
--check N grabs N frames without a window, reports detections and saves check.jpg.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np


def find_capture_device():
    """Return /dev/videoN of the USB capture card (not the laptop webcam)."""
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name = open(os.path.join(path, "name")).read().strip()
        if "Integrated" in name:
            continue
        dev = "/dev/" + os.path.basename(path)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return dev
    return None


CLAHE = cv2.createCLAHE(2.0, (8, 8))


def make_detector(dict_name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def annotate(frame, detector):
    """Detect markers, draw them, return list of (id, cx_err, cy_err, size) with errors in -1..1."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # small, far-away markers: boost local contrast and detect on a 2x enlarged image (about 3x more hits
    # at 25 px marker size in recorded flights), then scale the corners back to frame coordinates
    big = cv2.resize(CLAHE.apply(gray), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = detector.detectMarkers(big)
    corners = [c / 2.0 for c in corners]
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    found = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            size = float(np.linalg.norm(pts[0] - pts[2]))
            ex, ey = (cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)
            found.append((int(i), ex, ey, size))
            cv2.line(frame, (w // 2, h // 2), (int(cx), int(cy)), (0, 255, 255), 2)
            cv2.putText(frame, "id %d  ex %+.2f ey %+.2f  size %.0fpx" % (i, ex, ey, size),
                        (10, 60 + 25 * len(found)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="default: auto-detect capture card")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--check", type=int, default=0, help="headless: grab N frames and report")
    args = ap.parse_args()

    dev = args.device or find_capture_device()
    if not dev:
        raise SystemExit("No capture card found (is the video receiver's USB capture plugged in?)")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    detector = make_detector(args.dict)
    print("capture device:", dev, "| dictionary:", args.dict)

    if args.check:
        hits, t0, frame = 0, time.time(), None
        for _ in range(args.check):
            ok, frame = cap.read()
            if not ok:
                continue
            found = annotate(frame, detector)
            hits += bool(found)
            if found:
                print("markers:", [(i, round(ex, 2), round(ey, 2)) for i, ex, ey, _ in found])
        fps = args.check / (time.time() - t0)
        print("frames %d, with markers %d, %.1f fps, frame %s, mean brightness %.0f"
              % (args.check, hits, fps, None if frame is None else frame.shape[1::-1],
                 0 if frame is None else frame.mean()))
        if frame is not None:
            cv2.imwrite("check.jpg", frame)
        return

    t_last, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            time.sleep(0.1)
            continue
        found = annotate(frame, detector)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-3)
        t_last = now
        cv2.putText(frame, "%.0f fps  markers: %d" % (fps, len(found)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("FPV ArUco", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = time.strftime("snap_%H%M%S.jpg")
            cv2.imwrite(name, frame)
            print("saved", name)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
