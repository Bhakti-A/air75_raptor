"""Calibrate the FPV camera (fisheye model) using the marker wall itself as the target -> fpv_calib.npz.

Props off, battery in, video on. Carry the drone by hand in front of the wall for about a minute:
near (0.5 m) and far (2 m), left and right of the wall, low and high, and tilt/turn it so the markers
pass through the centre, the edges and the corners of the picture. Move slowly - the analog picture
smears when the camera moves fast. Views are picked automatically whenever at least --min-markers are
seen and the picture has changed enough; stops at --views or on q.

Your tape measurements in room_layout.json are used as they are. What is fitted besides the lens is
the size the detector sees: on the analog picture the black squares come out a few % smaller than the
printed ones (edge blur), and that would otherwise make every distance too long. It is saved in
fpv_calib.npz as marker_size_eff and room_pose.py uses it.

--video calibrates from a recorded raw FPV video; --from-views redoes the fit on the views of the last run.
"""
import argparse
import os
import time

import cv2
import numpy as np

from aruco_view import find_capture_device, make_detector
from room_pose import HERE, detect, load_layout

def plane_points(ids, centres, half):
    """Marker corners (TL TR BR BL) of `ids` in wall-plane coordinates (right, down, 0), metres."""
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    pts = np.concatenate([centres[i] + square for i in ids])
    return np.column_stack([pts, np.zeros(len(pts))])


def calibrate(views, centres, half, size):
    """views: list of (ids, 4n x 2 pixels). Returns (rms, K, D, rvecs, tvecs)."""
    obj = [plane_points(ids, centres, half).reshape(1, -1, 3) for ids, _ in views]
    img = [px.reshape(1, -1, 2).astype(np.float64) for _, px in views]
    f = 0.447 * size[0]
    K = np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1.0]])
    D = np.zeros((4, 1))
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS   # OpenCV 5 names
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    return cv2.fisheye.calibrate(obj, img, tuple(size), K, D, flags=flags, criteria=crit)


def fit_marker_size(views, centres, half, size):
    """Calibrate for a range of apparent marker sizes, keep the best. Returns (half size, rms, K, D)."""
    def fit(h):
        try:
            return calibrate(views, centres, h, size)
        except cv2.error:                            # fisheye initialisation can fail for a bad size guess
            return None
    tried = {}
    for scale in np.arange(0.84, 1.021, 0.02):
        tried[round(scale, 3)] = fit(half * scale)
    best = min((k for k in tried if tried[k]), key=lambda k: tried[k][0])
    for scale in np.arange(best - 0.015, best + 0.0151, 0.005):
        if round(scale, 3) not in tried:
            tried[round(scale, 3)] = fit(half * scale)
    best = min((k for k in tried if tried[k]), key=lambda k: tried[k][0])
    rms, K, D = tried[best][:3]
    return half * best, rms, K, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--from-views", action="store_true", help="reuse fpv_calib_views.npz from the last run")
    ap.add_argument("--layout", default=os.path.join(HERE, "room_layout.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "fpv_calib.npz"))
    ap.add_argument("--views", type=int, default=80)
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--min-shift", type=float, default=25.0, help="mean corner movement (px) before the next view is taken")
    args = ap.parse_args()

    dict_name, layout = load_layout(args.layout)
    half = np.linalg.norm(next(iter(layout.values()))[0] - next(iter(layout.values()))[1]) / 2.0
    centres = {i: np.array([-c[:, 1].mean(), -c[:, 2].mean()]) for i, c in layout.items()}   # (right, down)
    views_file = os.path.join(HERE, "fpv_calib_views.npz")

    if args.from_views:
        d = np.load(views_file)
        size = tuple(int(v) for v in d["image_size"])
        views = [(list(d["ids"][d["view"] == v][::4]), d["px"][d["view"] == v]) for v in np.unique(d["view"])]
    else:
        detector = make_detector(dict_name)
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            dev = args.device or find_capture_device()
            if not dev:
                raise SystemExit("No capture card found")
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        views, last, size, t_last = [], {}, None, 0.0
        while len(views) < args.views:
            ok, frame = cap.read()
            if not ok:
                break
            size = frame.shape[1::-1]
            found = {i: c for i, c in detect(frame, detector).items() if i in layout}
            common = [i for i in found if i in last]
            moved = not common or np.mean([np.linalg.norm(found[i] - last[i], axis=1).mean() for i in common]) > args.min_shift
            if len(found) >= args.min_markers and moved and (args.video or time.time() - t_last > 0.3):
                ids = sorted(found)
                views.append((ids, np.concatenate([found[i] for i in ids])))
                last, t_last = found, time.time()
            if not args.video:
                for _, px in views[-15:]:
                    for p in px:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
                cv2.putText(frame, "views %d/%d   markers now: %d" % (len(views), args.views, len(found)), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if len(found) >= args.min_markers else (0, 0, 255), 2)
                cv2.imshow("FPV calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
        if views:
            np.savez(views_file, image_size=np.array(size), px=np.concatenate([px for _, px in views]),
                     view=np.concatenate([[v] * len(px) for v, (_, px) in enumerate(views)]),
                     ids=np.concatenate([np.repeat(ids, 4) for ids, _ in views]))
    if len(views) < 15:
        raise SystemExit("only %d views - need at least 15, with the markers spread over the whole picture" % len(views))

    pts = np.concatenate([px for _, px in views])
    cover = (pts.max(axis=0) - pts.min(axis=0)) / np.array(size)
    print("views %d, corners cover %.0f%% x %.0f%% of the picture" % (len(views), 100 * cover[0], 100 * cover[1]))
    rms = calibrate(views, centres, half, size)[0]
    print("printed marker size %.1f cm: rms %.2f px" % (200 * half, rms))
    half_eff, rms, K, D = fit_marker_size(views, centres, half, size)
    print("detector sees %.1f cm (%.0f%% of printed): rms %.2f px" % (200 * half_eff, 100 * half_eff / half, rms))
    print("K =\n", K.round(2), "\nD =", D.ravel().round(4))
    np.savez(args.out, K=K, D=D, image_size=np.array(size), rms=rms, marker_size_eff=2 * half_eff)
    print("saved", args.out)
    if rms > 1.5:
        print("rms is still high: redo it moving more slowly, with the markers reaching the edges of the picture")


if __name__ == "__main__":
    main()
