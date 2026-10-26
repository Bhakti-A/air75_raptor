"""Live drone position from the wall markers: FPV capture -> ArUco -> pose in the room frame.

Room frame (room_layout.json; same as the policy's FLU frame while the drone faces the wall):
origin on the floor under the middle marker column, x into the wall, y to the left, z up.
The drone is at negative x (x = -1.5 is 1.5 m from the wall); yaw 0 = facing the wall square on, + = turned left.

Needs the FPV camera calibration fpv_calib.npz (calibrate_fpv_wall.py). Without it a nominal wide-angle
lens is assumed and the numbers are only rough - the window says UNCALIBRATED.

Keys in the window: q = quit, t = drone is sitting level right now -> measure the camera tilt from this frame.
--record writes room_pose_<time>.csv (one row per frame: pose, velocity, what was sent, reference, trim, and why a
frame has no pose) and the raw FPV feed as room_pose_<time>.avi (replay it with --video).

--send streams the position to the drone for the RAPTOR policy (radio: laptlm.lua, CH9..CH15; sticks are
never touched): position error (drone - target) and velocity, turned into the drone's own heading frame
(x forward, y left, z up), +-SEND_RANGE_POSITION m / +-SEND_RANGE_VELOCITY m/s = full channel. Gaps shorter than
--hold-time send nothing (the drone keeps the last value and dead-reckons); longer ones are sent as "no fix".
Target point: --target x,y,z (room frame), then in the window: i / k = 5 cm toward / away from the wall,
j / l = 5 cm left / right, u / o = 5 cm up / down, b = 10 cm up, h = target := where the drone is now.
The drone is never sent the target itself but a reference that moves to it at a limited speed (Reference):
RAPTOR switched on = take off slowly from where the drone sits, n = land (down to the floor), g = take off again,
c = fly a horizontal circle around the target at its height (c again = back to holding the target).
Trim: an integral term on the laptop removes the policy's steady offset; it is saved to last_trim.json after each
hold and the next flight starts from it (--no-saved-trim to start at 0).

Between ArUco poses the pose is carried on by optical flow on the wall (wall_flow.py, off with --no-flow).
"""
import argparse
import csv
import json
import math
import os
import time

import cv2
import numpy as np

from aruco_view import find_capture_device, make_detector
from wall_flow import FusedPose, WallFlow

HERE = os.path.dirname(os.path.abspath(__file__))
# full channel deflection; must match EXTERNAL_RANGE_* (and the clip) in the firmware's policy.cpp. Were 0.3 / 0.3 until
# 2026-10-21: the policy couldn't see faster than 0.3 m/s and overshot the takeoff
SEND_RANGE_POSITION = 0.5                            # m
SEND_RANGE_VELOCITY = 1.5                            # m/s
SEND_RANGE = np.array([SEND_RANGE_POSITION] * 3 + [SEND_RANGE_VELOCITY] * 3)


def heading_frame(vec, yaw_deg):
    """Room-frame vector -> drone heading frame (x forward, y left, z up; drone yaw removed)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1], vec[2]])


def load_layout(path, marker_size=None):
    """Return (dict name, {id: 4x3 corners in the room frame, ArUco order TL TR BR BL as seen facing the wall}).

    marker_size overrides the printed size (the calibration's marker_size_eff: the size the detector sees).
    """
    cfg = json.load(open(path))
    h = (marker_size or cfg["marker_size"]) / 2.0
    corners = {}
    for mid, (x, y, z) in cfg["markers"].items():
        corners[int(mid)] = np.array([[x, y + h, z + h], [x, y - h, z + h], [x, y - h, z - h], [x, y + h, z - h]])
    return cfg["dict"], corners


def load_camera(path, size):
    """Fisheye intrinsics (K, D, calibrated?, apparent marker size or None). Falls back to a nominal ~160 deg lens."""
    if os.path.exists(path):
        d = np.load(path)
        K, D = d["K"].astype(float), d["D"].astype(float).reshape(4, 1)
        cw, ch = d["image_size"]
        if (cw, ch) != tuple(size):                  # calibration made at another capture resolution
            K = K * np.array([[size[0] / cw], [size[1] / ch], [1.0]])
        return K, D, True, float(d["marker_size_eff"]) if "marker_size_eff" in d.files else None
    f = 0.447 * size[0]                              # equidistant lens, 160 deg across the diagonal of a 4:3 frame
    return np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1.0]]), np.zeros((4, 1)), False, None


def tune_detector(detector):
    """2026-10-21 flight videos: in flight the squares were found but their bits not read (soft analog video); reading
    only the cell centres (margin 0.3, 10 px per cell) and no CLAHE: frames with >= 2 markers 54 -> 70% and 31 -> 49%."""
    params = detector.getDetectorParameters()
    params.perspectiveRemoveIgnoredMarginPerCell = 0.3
    params.perspectiveRemovePixelPerCell = 10
    detector.setDetectorParameters(params)
    return detector


def detect(frame, detector):
    """{id: 4x2 corner pixels}, detected on a 2x enlarged image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = detector.detectMarkers(big)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) / 2.0 for c, i in zip(corners, ids.flatten())}


def solve_pose(found, layout, K, D, prev_R=None):
    """Camera pose from every visible layout marker at once.

    Returns (R_wc, cam_pos, err_px, ids) or None. R_wc has the camera axes (x right, y down, z forward)
    as columns in the room frame. A planar target has two candidate poses; with a previous pose the
    closer one is taken, otherwise the one with the lower reprojection error.
    """
    ids = sorted(i for i in found if i in layout)
    if not ids:
        return None
    obj = np.concatenate([layout[i] for i in ids]).astype(np.float64)
    img = np.concatenate([found[i] for i in ids]).astype(np.float64)
    norm = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    if not np.all(np.isfinite(norm)):
        return None
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
        R_wc = cv2.Rodrigues(rvec)[0].T
        pos = (-R_wc @ tvec).ravel()
        if pos[0] >= 0:                              # behind the wall
            continue
        if prev_R is not None:                       # rotation angle from the previous frame's pose
            score = math.acos(clamp((np.trace(prev_R.T @ R_wc) - 1) / 2, -1, 1))
        else:
            score = float(err)
        if best is None or score < best[0]:
            best = (score, R_wc, pos, float(err) * K[0, 0])
    if best is None:
        return None
    return best[1], best[2], best[3], ids


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def body_attitude(R_wc, tilt_deg):
    """(yaw, pitch, roll) of the drone in degrees: yaw + = left, pitch + = nose up, roll + = right side down."""
    t = math.radians(tilt_deg)
    # camera axes in the body frame (x forward, y left, z up) for a camera tilted up by `tilt`
    R_bc = np.array([[0, math.sin(t), math.cos(t)], [-1, 0, 0], [0, -math.cos(t), math.sin(t)]])
    R_wb = R_wc @ R_bc.T
    yaw = math.atan2(R_wb[1, 0], R_wb[0, 0])
    pitch = math.asin(clamp(R_wb[2, 0], -1, 1))
    roll = math.atan2(R_wb[2, 1], R_wb[2, 2])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class Velocity:
    """Low-passed finite difference of the position; restarts after a gap in the detections."""

    def __init__(self, tau=0.08, max_gap=0.3):  # tau must match VELOCITY_TAU in the post-training environment.h
        self.tau, self.max_gap = tau, max_gap
        self.t = self.p = None
        self.v = np.zeros(3)

    def update(self, t, p):
        if self.t is None or t - self.t > self.max_gap:
            self.v = np.zeros(3)
        elif t > self.t:
            dt = t - self.t
            a = dt / (self.tau + dt)
            self.v = (1 - a) * self.v + a * (p - self.p) / dt
        self.t, self.p = t, p.copy()
        return self.v


def step_towards(p, goal, speed_xy, speed_z, dt):
    """p moved towards goal by at most speed_xy * dt sideways and speed_z * dt in height."""
    p = p.copy()
    d = goal[:2] - p[:2]
    dist = np.linalg.norm(d)
    p[:2] = goal[:2] if dist <= speed_xy * dt else p[:2] + d / dist * speed_xy * dt
    p[2] = p[2] + clamp(goal[2] - p[2], -speed_z * dt, speed_z * dt)
    return p


class Reference:
    """Where the drone is told to be. It follows the target at a limited speed instead of jumping to it.

    2026-10-22 (f16, f17, d1-d3): with RAPTOR engaged on the floor and the target 0.2-0.5 m up, the recurrent policy
    kept winding its thrust up while the floor held the drone, launched at ~1 m/s, overshot and lost the markers.
    idle (RAPTOR off): the reference is the drone itself, so the error is ~0 the moment RAPTOR engages.
    takeoff: up at --climb-speed from where the drone sat; sideways to the target only once 10 cm up.
    hold: on the target (target changes are followed at the same limited speeds).
    land: straight down to LAND_Z (below the floor, so the policy keeps pushing down) -> 'landed'.
    circle: around the target (centred on it) in the horizontal plane at the target's height; the radius grows
    from 0 over the first lap. (Was centred one radius away from the wall: with 0.3 m the drone went out to 1.8 m,
    where the markers were too small to decode and it flew on flow alone, drifted 0.7 m sideways and was lost.)
    trim: the policy hovered 8-17 cm closer to the wall than the reference in every flight of 2026-10-22 (and never
    corrected it), so in hold / circle the reference sent to the drone is shifted against the mean horizontal error
    (time constant TRIM_TAU, at most TRIM_MAX) -- the integral term the policy does not have.
    Without a fresh pose the reference stands still (the drone is dead-reckoning).
    2026-10-23 (23_2): with the trim starting at 0 every flight, the first ~6 s of hold were 20-30 cm off (the
    whole-hold error 12 cm vs 7 cm once settled). The trim of the last hold is now saved and the next flight starts
    from it, faded in over the climb so the drone does not get pushed sideways while it still sits on the floor.
    Height is trimmed too (at most TRIM_MAX_Z): its offset was 2-5 cm in d5 / 23_2.
    2026-10-23 (23_3): the height trim (-9 cm) stayed on while landing, so the floor target was 4 cm above the floor:
    on the floor the policy saw itself too low, wound its thrust up (motors at hover power for 8 s) and bounced up
    four times, the last to 0.7 m. Landing now uses no height trim, aims LAND_Z = 12 cm below the floor, and once it
    has sat below TOUCH_Z for TOUCH_S it is 'landed' and aimed LANDED_Z below the floor, so it stays down."""
    LAND_Z = -0.12
    TOUCH_Z = 0.06                                   # m: on the floor (the pose at rest reads 1-5 cm)
    TOUCH_S = 0.4                                    # s below TOUCH_Z -> landed
    LANDED_Z = -0.45                                 # aim once landed: the position error is at the top of its range
    LIFT_FIRST = 0.10                                # m of climb before moving sideways
    TRIM_TAU = 4.0                                   # s (default of --trim-tau)
    TRIM_MAX = 0.25                                  # m
    TRIM_MAX_Z = 0.10                                # m

    def __init__(self, climb_speed, move_speed, circle_radius=0.15, circle_period=10.0, trim_tau=TRIM_TAU,
                 trim0=None):
        self.climb, self.move = climb_speed, move_speed
        self.radius, self.period, self.angle, self.t_circle = circle_radius, circle_period, 0.0, 0.0
        self.mode, self.p, self.v = "idle", None, np.zeros(3)
        self.z0, self.t, self.t_down = 0.0, None, None
        self.trim_tau = trim_tau
        self.trim0 = np.zeros(3) if trim0 is None else np.asarray(trim0, float)
        self.trim = self.trim0.copy()
        self.trim_fade = 0.0                         # 0 on the floor -> 1 at the target height
        self.t_hold = 0.0                            # s in hold / circle this flight (the trim is saved after 10 s)

    @property
    def aim(self):
        """The reference as sent to the drone (with the trim; none in height while landing)."""
        if self.mode in ("land", "landed"):
            aim = self.p - self.trim_fade * self.trim * [1.0, 1.0, 0.0]
            if self.mode == "landed":
                aim[2] = self.LANDED_Z
            return aim
        return self.p - self.trim_fade * self.trim

    def start(self, mode):
        if mode == "circle" and self.mode != "hold":
            print("reference: circle only from hold (now %s)" % self.mode)
            return
        if mode == "circle":
            self.angle, self.t_circle = 0.0, 0.0
        if mode == "takeoff":
            self.trim, self.trim_fade, self.t_hold = self.trim0.copy(), 0.0, 0.0
        if mode == "land" and self.p is not None and self.mode not in ("land", "landed"):
            self.p[2] -= self.trim_fade * self.trim[2]   # start down from where the drone was aimed (no step)
        if self.p is not None:
            self.mode, self.z0, self.t_down = mode, self.p[2], None
            print("reference:", mode, "from", self.p.round(2))

    def update(self, now, pos, target):
        dt = 0.0 if self.t is None else min(now - self.t, 0.1)
        self.t = now
        old = None if self.p is None else self.p.copy()
        if self.mode == "idle" or self.p is None:
            if pos is not None:
                self.p = pos.copy()
            self.v = np.zeros(3)
            return
        if pos is None:                              # no fresh pose: wait for the drone
            self.v = np.zeros(3)
            return
        if self.mode == "takeoff":
            goal = target if self.p[2] >= min(self.z0 + self.LIFT_FIRST, target[2]) else \
                np.array([self.p[0], self.p[1], target[2]])
            self.p = step_towards(self.p, goal, self.move, self.climb, dt)
            self.trim_fade = clamp((self.p[2] - self.z0) / max(target[2] - self.z0, 0.05), 0.0, 1.0)
            if np.allclose(self.p, target):
                self.mode = "hold"
                print("reference: hold at", target.round(2))
        elif self.mode == "hold":
            self.p = step_towards(self.p, target, self.move, self.climb, dt)
        elif self.mode == "circle":
            # the angle and the radius ramp advance only while poses come in (frozen above otherwise)
            self.t_circle += dt
            self.angle += 2 * math.pi * dt / self.period
            r = self.radius * min(1.0, self.t_circle / self.period)
            goal = target + r * np.array([math.cos(self.angle), math.sin(self.angle), 0.0])
            self.p = step_towards(self.p, goal, 2 * self.move, self.climb, dt)
        elif self.mode == "land":
            self.p = step_towards(self.p, np.array([self.p[0], self.p[1], self.LAND_Z]), self.move, self.climb, dt)
            if pos[2] < self.TOUCH_Z and self.p[2] <= 0.02:
                self.t_down = self.t_down or now
                if now - self.t_down > self.TOUCH_S:
                    self.mode = "landed"
                    print("reference: LANDED -- switch RAPTOR off / disarm")
            else:
                self.t_down = None
        if self.mode in ("hold", "circle"):
            self.trim_fade, self.t_hold = 1.0, self.t_hold + dt
            self.trim += (pos - self.p) * dt / self.trim_tau
            n = np.linalg.norm(self.trim[:2])
            if n > self.TRIM_MAX:
                self.trim[:2] *= self.TRIM_MAX / n
            self.trim[2] = clamp(self.trim[2], -self.TRIM_MAX_Z, self.TRIM_MAX_Z)
        self.v = (self.p - old) / dt if dt > 0 else np.zeros(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="default: auto-detect capture card")
    ap.add_argument("--video", default=None, help="run on a recorded raw FPV video instead of the capture card")
    ap.add_argument("--layout", default=os.path.join(HERE, "room_layout.json"))
    ap.add_argument("--calib", default=os.path.join(HERE, "fpv_calib.npz"))
    ap.add_argument("--cam-tilt", type=float, default=21.0, help="camera up-tilt in degrees (key t measures it)")
    ap.add_argument("--min-markers", type=int, default=2, help="ignore poses from fewer markers (one marker jumps)")
    ap.add_argument("--max-err", type=float, default=2.0, help="ignore poses with a larger reprojection error (px)")
    ap.add_argument("--max-jump-speed", type=float, default=1.5,
                    help="reject a pose that moved faster than this (m/s, + 5 cm) since the last accepted one")
    ap.add_argument("--record", action="store_true",
                    help="save room_pose_<time>.csv (every frame, with why a frame has no pose) and the raw FPV feed "
                         "as room_pose_<time>.avi (frame n = csv column 'frame'; replay with --video)")
    ap.add_argument("--headless", action="store_true", help="no window, print only")
    ap.add_argument("--send", action="store_true", help="stream the position error to the drone over the Zorro")
    # 0.5 -> 0.3 (2026-10-22): above ~0.5 m the camera loses the markers
    ap.add_argument("--target", default="-1.2,0,0.3", help="target point x,y,z in the room frame (m)")
    ap.add_argument("--climb-speed", type=float, default=0.15, help="reference speed up / down (m/s)")
    ap.add_argument("--move-speed", type=float, default=0.10, help="reference speed sideways (m/s)")
    ap.add_argument("--circle-radius", type=float, default=0.15, help="key c: circle radius (m)")
    ap.add_argument("--circle-period", type=float, default=10.0, help="key c: seconds per lap")
    ap.add_argument("--no-flow", action="store_true", help="ArUco poses only (no optical flow between them)")
    ap.add_argument("--hold-time", type=float, default=0.2,
                    help="pose lost this long -> send 'no fix' (shorter gaps: the drone keeps the last value)")
    # 2026-10-23: the firmware scales the position error by 0.7 on every axis (POSITION_OBSERVATION_GAIN). In 23_1
    # the drone sank from 0.3 m to ~0.05 m and was slow to stop it, then pumped back up to 0.65 m and crashed -- the
    # wobble to calm is sideways, so the height is put back to full strength here and only x/y are turned down.
    ap.add_argument("--pos-gain-xy", type=float, default=1.0, help="extra scale on the sent x/y position error")
    ap.add_argument("--pos-gain-z", type=float, default=1.0,
                    help="extra scale on the sent height error (1.43 undoes the firmware's 0.7)")
    ap.add_argument("--trim-tau", type=float, default=Reference.TRIM_TAU, help="trim (integral) time constant (s)")
    ap.add_argument("--trim-file", default=os.path.join(HERE, "last_trim.json"),
                    help="trim saved after each hold of > 10 s; the next flight starts from it")
    ap.add_argument("--no-saved-trim", action="store_true", help="start the trim at 0")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="send the position this many s ahead (pos + v * lead): camera delay beyond the firmware's 130 ms")
    args = ap.parse_args()

    dict_name, layout = load_layout(args.layout)
    detector = tune_detector(make_detector(dict_name))
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        dev = args.device or find_capture_device()
        if not dev:
            raise SystemExit("No capture card found (is the video receiver's USB capture plugged in?)")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames from the video source")
    K, D, calibrated, size_eff = load_camera(args.calib, frame.shape[1::-1])
    if size_eff:
        layout = load_layout(args.layout, size_eff)[1]
    print("layout markers:", sorted(layout), "| camera:", "calibrated" if calibrated else "UNCALIBRATED (nominal lens)")

    log = video = None
    if args.record:
        name = time.strftime("room_pose_%Y%m%d_%H%M%S")
        log = csv.writer(open(name + ".csv", "w", newline=""))
        log.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "yaw", "pitch", "roll", "markers", "err_px",
                      "unix_time", "tx", "ty", "tz", "send_px", "send_py", "send_pz", "send_vx", "send_vy", "send_vz",
                      "frame", "status", "rx", "ry", "rz", "ref_mode", "source", "trim_x", "trim_y"])
        video = cv2.VideoWriter(name + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS) or 30.0,
                                frame.shape[1::-1])
        print("recording", name + ".csv", "and", name + ".avi")

    target = np.array([float(v) for v in args.target.split(",")])
    link = None
    if args.send:
        from zorro_link import ZorroLink
        link = ZorroLink(rate_hz=50, sticks=False)
        print("sending position to the drone, target", target, " position gain xy %.2f z %.2f (x firmware 0.7)"
              % (args.pos_gain_xy, args.pos_gain_z))
    trim0 = None
    if not args.no_saved_trim and os.path.exists(args.trim_file):
        with open(args.trim_file) as fh:
            trim0 = json.load(fh)["trim"]
        print("starting from the saved trim", np.round(trim0, 3), "(%s)" % args.trim_file)
    ref = Reference(args.climb_speed, args.move_speed, args.circle_radius, args.circle_period, args.trim_tau, trim0)
    trim_saved_at = None
    fused = None if args.no_flow else FusedPose(WallFlow(K, D))
    raptor_was_on = False
    vel, prev_R, t_seen, t_print, tilt, n = Velocity(), None, 0.0, 0.0, args.cam_tilt, 0
    t_sent = -1e9
    t_accepted, p_accepted = -1e9, None
    p_candidate, t_candidate, n_candidate = None, 0.0, 0  # a new position seen in agreeing frames (the drone really moved)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    t0 = time.time()
    while ok:
        now = n / fps if args.video else time.time() - t0
        if video is not None:
            video.write(frame)                       # raw, before the overlays are drawn
        found = detect(frame, detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if now - t_seen > 0.3:
            prev_R = None                            # too long without a pose to trust it as a hint
        pose = solve_pose(found, layout, K, D, prev_R)
        text, colour = "no layout marker in view", (0, 0, 255)
        if pose and (len(pose[3]) < args.min_markers or pose[2] > args.max_err):
            text = "rejected: %d marker(s), %.1f px" % (len(pose[3]), pose[2])
            pose = None
        if pose:
            # 2026-10-21 flights: 15-34 times per flight the pose jumped > 15 cm in one frame (33 ms), mostly from
            # 2 markers -- impossible for the drone, and each jump went to the policy as a sudden big error.
            # 2026-10-22 (f14/f15): a looser first version still let 55 cm jumps through after short gaps and with
            # the wider velocity range the policy cut all motors on them.
            # 2026-10-22 (d2/d3): capping every jump at 15 cm then threw away real 3-6 marker poses while the drone
            # climbed at 0.8 m/s after a short gap. The bad jumps are almost all from 2 markers, so: >= 3 markers
            # may move 5 cm + speed * gap (at most 35 cm), 2 markers at most 15 cm. Anything else -- including the
            # first pose after > 1 s without one -- is only taken once 2 frames in a row with >= 3 markers agree
            # (within 5 cm + speed * time between them).
            fresh = p_accepted is None or now - t_accepted > 1.0
            jump = 0.0 if fresh else np.linalg.norm(pose[1] - p_accepted)
            limit = min(0.05 + args.max_jump_speed * (now - t_accepted), 0.35 if len(pose[3]) >= 3 else 0.15)
            if fresh or jump > limit:
                if (len(pose[3]) >= 3 and p_candidate is not None
                        and np.linalg.norm(pose[1] - p_candidate) < 0.05 + args.max_jump_speed * (now - t_candidate)):
                    n_candidate += 1
                else:
                    n_candidate = 1 if len(pose[3]) >= 3 else 0
                p_candidate, t_candidate = pose[1].copy(), now
                if n_candidate < 2:
                    text = ("waiting for a steady pose (%d markers)" % len(pose[3]) if fresh
                            else "rejected: jump %.0f cm (%d markers)" % (100 * jump, len(pose[3])))
                    pose = None
                else:
                    vel = Velocity()                 # restart the velocity: the step is not a speed
        source = "aruco" if pose else ""
        if fused is not None:
            # 2026-10-22 replay of the d/f flight videos: in flight ArUco alone gave a pose in 50-82% of the
            # frames, with the flow 94-100%; flow vs the withheld ArUco pose: ~1 cm median, 90% within 2-19 cm.
            # Flow motion and ArUco position are blended (FusedPose), not switched between.
            r = fused.update(gray, now, (pose[0], pose[1]) if pose else None)
            if r is None:
                pose, source = None, ""
            else:
                pose, source = (r[0], r[1], pose[2] if pose else 0.0, pose[3] if pose else []), r[2]
        if pose:
            t_accepted, p_accepted, p_candidate, n_candidate = now, pose[1].copy(), None, 0
        if link:
            st = link.get_status()
            raptor_on = bool(st and st.get("ch6") is not None and st["ch6"] > 500)  # firmware engages above 1750 us
            if raptor_on and not raptor_was_on:
                ref.start("takeoff")
            elif raptor_was_on and not raptor_on:
                ref.mode = "idle"
            raptor_was_on = raptor_on
        ref.update(now, pose[1] if pose else None, target)
        if ref.mode in ("land", "landed") and ref.t_hold > 10.0 and trim_saved_at != ref.t_hold:
            trim_saved_at = ref.t_hold
            with open(args.trim_file, "w") as fh:
                json.dump({"trim": [round(float(a), 4) for a in ref.trim], "hold_s": round(ref.t_hold, 1),
                           "unix_time": time.time()}, fh)
            print("trim saved", ref.trim.round(3), "->", args.trim_file)
        sent = None
        if pose:
            R_wc, pos, err, ids = pose
            prev_R, t_seen = R_wc, now
            v = vel.update(now, pos)
            yaw, pitch, roll = body_attitude(R_wc, tilt)
            sent = np.concatenate([heading_frame(pos + v * args.lead - ref.aim, yaw)
                                   * [args.pos_gain_xy, args.pos_gain_xy, args.pos_gain_z],
                                   heading_frame(v - ref.v, yaw)]) / SEND_RANGE
        if link:
            # A single bad frame (blur, a marker at the edge) used to go out as "no fix", which made the
            # drone's position input jump to zero for that update. Short gaps now send nothing: the radio
            # keeps the last value and the drone holds it. Only a gap over --hold-time is "no fix".
            if sent is not None:
                link.send_position(np.clip(sent, -1, 1))
                t_sent = now
            elif now - t_sent > args.hold_time:
                link.send_position(None)
        if log and not pose:
            log.writerow(["%.3f" % now] + [""] * 11 + ["%.3f" % time.time()] + ["%.3f" % a for a in target]
                         + [""] * 6 + [n, text]
                         + (["%.3f" % a for a in ref.p] if ref.p is not None else [""] * 3) + [ref.mode, ""] + ["%.3f" % a for a in ref.trim[:2]])
        if pose:
            text = "x %+.2f y %+.2f z %.2f m  yaw %+.0f  [%d markers, %.1f px]" % (*pos, yaw, len(ids), err)
            colour = (255, 255, 0) if source == "flow" else (0, 255, 0) if len(ids) > 1 else (0, 255, 255)
            if source == "flow":
                text = "x %+.2f y %+.2f z %.2f m  yaw %+.0f  [flow]" % (*pos, yaw)
            if log:
                log.writerow(["%.3f" % now] + ["%.4f" % a for a in (*pos, *v)]
                             + ["%.2f" % a for a in (yaw, pitch, roll)] + [len(ids), "%.2f" % err]
                             + ["%.3f" % time.time()] + ["%.3f" % a for a in target]
                             + ["%.4f" % a for a in np.clip(sent, -1, 1) * SEND_RANGE] + [n, "ok"]
                             + ["%.3f" % a for a in ref.p] + [ref.mode, source] + ["%.3f" % a for a in ref.trim[:2]])
            if link:
                e = pos - ref.p
                text = "%s  err x %+.2f y %+.2f z %+.2f m  ref z %.2f  trim %+.2f %+.2f  [%s]" % (
                    ref.mode.upper(), *e, ref.p[2], *ref.trim[:2], "flow" if source == "flow" else "%d mk" % len(ids))
            if now - t_print > 0.2:
                t_print = now
                print("%7.2f  pos %+.3f %+.3f %.3f  vel %+.2f %+.2f %+.2f  yaw %+6.1f pitch %+5.1f roll %+5.1f  ids %s  err %.1f px"
                      % (now, *pos, *v, yaw, pitch, roll, ids, err))
        if not args.headless:
            if found:
                cv2.aruco.drawDetectedMarkers(frame, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                              np.array(list(found)).reshape(-1, 1))
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if not calibrated:
                cv2.putText(frame, "UNCALIBRATED", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if link:
                st = link.get_status()
                line = "radio: no status" if st is None or st.get("ch15") is None else \
                    "radio CH9 %+5d  CH15 %+5d  RAPTOR %s" % (st["ch9"], st["ch15"], "ON" if st["ch6"] > 500 else "off")
                cv2.putText(frame, line, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Room pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            step = {ord("i"): (0.05, 0, 0), ord("k"): (-0.05, 0, 0), ord("j"): (0, 0.05, 0),
                    ord("l"): (0, -0.05, 0), ord("u"): (0, 0, 0.05), ord("o"): (0, 0, -0.05),
                    ord("b"): (0, 0, 0.10)}.get(key)
            if step is not None:
                target = target + np.array(step)
                print("target", target.round(2))
            if key == ord("g"):
                ref.start("takeoff")
            if key == ord("n"):
                ref.start("land")
            if key == ord("c"):
                ref.start("hold" if ref.mode == "circle" else "circle")
            if key == ord("h") and pose:
                target = pose[1].copy()
                print("target := current position", target.round(2))
            if key == ord("t") and pose:
                tilt = math.degrees(math.asin(clamp(pose[0][2, 2], -1, 1)))   # elevation of the optical axis
                print("camera tilt set to %.1f deg (use --cam-tilt %.1f next time)" % (tilt, tilt))
        ok, frame = cap.read()
        n += 1
    if link:
        link.send_position(None)
        time.sleep(0.1)
        link.close()
    cap.release()
    if video is not None:
        video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
