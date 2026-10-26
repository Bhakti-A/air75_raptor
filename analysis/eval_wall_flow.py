"""Offline check of wall_flow.WallFlow on recorded raw FPV videos (room_pose.py --record writes them).

python3 eval_wall_flow.py video.avi [...]
For each video: frames with an ArUco pose, frames covered by ArUco-or-flow, and the flow pose's error against the
ArUco pose in artificial blackouts (ArUco is withheld from the tracker for BLACKOUT s every PERIOD s, and the flow pose
in those frames is compared with the ArUco pose that was withheld), by age since the last anchor.
"""
import os
import sys
import cv2
import numpy as np
POS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "position")
sys.path.insert(0, POS)
from room_pose import load_layout, load_camera, tune_detector, detect, solve_pose
from aruco_view import make_detector
from wall_flow import WallFlow

BLACKOUT, PERIOD = 0.5, 1.0
for path in sys.argv[1:]:
    dict_name, layout = load_layout(os.path.join(POS, "room_layout.json"))
    det = tune_detector(make_detector(dict_name))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    K, D, _, size_eff = load_camera(os.path.join(POS, "fpv_calib.npz"), frame.shape[1::-1])
    layout = load_layout(os.path.join(POS, "room_layout.json"), size_eff)[1]
    flow_live, flow_test = WallFlow(K, D), WallFlow(K, D)
    n = n_aruco = n_cover = 0
    series = []  # per frame: 0 none, 1 ArUco, 2 flow
    errs = {}  # age bucket (0.1 s) -> list of (pos err, yaw err deg)
    prev_R = None
    while ok:
        now = n / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose = solve_pose(detect(frame, det), layout, K, D, prev_R)
        if pose and (len(pose[3]) < 2 or pose[2] > 2.0):
            pose = None
        # live: anchor on every ArUco pose, flow otherwise
        if pose:
            prev_R = pose[0]
            flow_live.anchor(gray, pose[0], pose[1], now)
            n_aruco += 1
            n_cover += 1
            series.append(1)
        else:
            prev_R = None
            if flow_live.track(gray, now):
                n_cover += 1
                series.append(2)
            else:
                series.append(0)
        # test: ArUco withheld during blackouts
        blackout = (now % PERIOD) < BLACKOUT
        if pose and not blackout:
            flow_test.anchor(gray, pose[0], pose[1], now)
        else:
            r = flow_test.track(gray, now)
            if r and pose and blackout:
                age = now - flow_test.t_anchor
                e = np.linalg.norm(r[1] - pose[1])
                dR = r[0].T @ pose[0]
                ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
                errs.setdefault(min(int(age / 0.1), 9), []).append((e, ang))
        ok, frame = cap.read()
        n += 1
    print("%s: %d frames, ArUco %.0f%%, ArUco+flow %.0f%%" % (path, n, 100 * n_aruco / n, 100 * n_cover / n))
    s = np.array(series)
    a = np.flatnonzero(s == 1)
    # flight window: from the end of the first long ArUco run (sitting on the ground) to the last ArUco frame
    runs = np.split(np.arange(len(s)), np.flatnonzero(np.diff(s == 1)) + 1)
    start = next((r[-1] + 1 for r in runs if s[r[0]] == 1 and len(r) > 60), a[0])
    w = s[start:a[-1] + 1]
    print("   in flight (frames %d-%d): ArUco %.0f%%, ArUco+flow %.0f%%" % (start, a[-1], 100 * np.mean(w == 1), 100 * np.mean(w > 0)))
    for b in sorted(errs):
        a = np.array(errs[b])
        print("   age %.1f-%.1f s: n %3d  pos err median %4.1f cm  90%% %4.1f cm   angle median %.1f deg"
              % (b / 10, b / 10 + 0.1, len(a), 100 * np.median(a[:, 0]), 100 * np.percentile(a[:, 0], 90), np.median(a[:, 1])))
