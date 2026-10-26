"""Camera pose between ArUco fixes from optical flow on the wall, 2026-10-22.

In flight the markers decode in only 20-55% of the frames (blur, a marker at the edge), but the wall around them --
the marker squares, the paper edges, any texture -- is still visible and still moves with the camera. Every ArUco pose
anchors the tracker: features are found in that frame and their 3D points are where their rays hit the wall plane
(x = 0 in the room frame). In the following frames the features are followed with pyramidal Lucas-Kanade and the
camera pose comes from PnP on those 2D-3D pairs (RANSAC: features on the floor or a moving object are not on the
plane and drop out). Between anchors the error grows (each re-seed of lost features inherits the current pose), so
the pose is only trusted for MAX_AGE seconds after the last anchor.
"""
import cv2
import numpy as np

WALL_X = 0.0               # wall plane in the room frame (room_layout.json: all markers at x = 0)
WALL_Z = (0.0, 2.5)        # a feature whose ray hits the wall outside this height band is not on the wall
MAX_AGE = 1.0              # s since the last ArUco anchor
MIN_INLIERS = 12
MAX_FEATURES = 150
RESEED_BELOW = 60          # find new features when fewer than this many are still tracked
RANSAC_PX = 3.0            # reprojection threshold in pixels
FB_PX = 1.0                # forward-backward flow check in pixels
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class WallFlow:
    def __init__(self, K, D):
        self.K, self.D = K, D
        self.gray = None           # previous frame
        self.pts = None            # Nx2 pixels in the previous frame
        self.obj = None            # Nx3 room-frame points on the wall
        self.R_wc = self.pos = None
        self.t_anchor = -1e9

    def _wall_points(self, px, R_wc, pos):
        """Pixels -> (kept mask, room-frame points where their rays hit the wall)."""
        if len(px) == 0:
            return np.zeros(0, bool), np.zeros((0, 3))
        norm = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), self.K, self.D).reshape(-1, 2)
        rays = (R_wc @ np.column_stack([norm, np.ones(len(norm))]).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (WALL_X - pos[0]) / rays[:, 0]
        obj = pos + s[:, None] * rays
        ok = np.isfinite(s) & (s > 0) & (obj[:, 2] > WALL_Z[0]) & (obj[:, 2] < WALL_Z[1]) & np.all(np.isfinite(norm), 1)
        return ok, obj

    def _seed(self, gray, R_wc, pos, keep=None):
        """Add features (away from the ones in keep) and give them wall points from the pose R_wc, pos."""
        mask = np.full(gray.shape, 255, np.uint8)
        if keep is not None:
            for x, y in keep.astype(int):
                cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
        n = MAX_FEATURES - (0 if keep is None else len(keep))
        new = cv2.goodFeaturesToTrack(gray, n, 0.01, 10, mask=mask, blockSize=7) if n > 0 else None
        new = np.zeros((0, 2)) if new is None else new.reshape(-1, 2)
        ok, obj = self._wall_points(new, R_wc, pos)
        return new[ok], obj[ok]

    def anchor(self, gray, R_wc, pos, now):
        """A trusted (ArUco) pose for this frame: re-seed every feature from it."""
        self.pts, self.obj = self._seed(gray, R_wc, pos)
        self.gray, self.R_wc, self.pos, self.t_anchor = gray, R_wc, pos.copy(), now

    def track(self, gray, now, max_age=MAX_AGE):
        """Follow the features into this frame. Returns (R_wc, pos, inliers) or None (and the tracker stops)."""
        if self.gray is None or now - self.t_anchor > max_age or self.pts is None or len(self.pts) < MIN_INLIERS:
            self.gray = None
            return None
        p0 = self.pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, p0, None, **LK)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, p1, None, **LK)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1) < FB_PX)
        pts, obj = p1.reshape(-1, 2)[good], self.obj[good]
        if len(pts) < MIN_INLIERS:
            self.gray = None
            return None
        norm = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.D).reshape(-1, 2)
        rvec = cv2.Rodrigues(self.R_wc.T)[0]
        tvec = (-self.R_wc.T @ self.pos).reshape(3, 1)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, norm, np.eye(3), None, rvec.copy(), tvec.copy(),
                                                 useExtrinsicGuess=True, iterationsCount=50,
                                                 reprojectionError=RANSAC_PX / self.K[0, 0], confidence=0.99,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < MIN_INLIERS:
            self.gray = None
            return None
        inl = inl.ravel()
        R_wc = cv2.Rodrigues(rvec)[0].T
        pos = (-R_wc @ tvec).ravel()
        if pos[0] >= 0:
            self.gray = None
            return None
        self.pts, self.obj = pts[inl], obj[inl]
        if len(self.pts) < RESEED_BELOW:
            new, new_obj = self._seed(gray, R_wc, pos, keep=self.pts)
            self.pts, self.obj = np.vstack([self.pts, new]), np.vstack([self.obj, new_obj])
        self.gray, self.R_wc, self.pos = gray, R_wc, pos
        return R_wc, pos, len(inl)


class FusedPose:
    """Flow for the motion, ArUco for where the drone is, 2026-10-22.

    The first version switched between the two: an ArUco pose re-anchored the flow, flow filled the gaps. In the
    195413 flight the flow had drifted by the time ArUco came back, so the position stepped by 6.6 cm median (up to
    15 cm) at every flow -> ArUco switch, against 2 cm between flow frames -- ~50 times in 15 s, each a burst of
    ~0.5 m/s in the velocity the drone was sent. Now the flow runs on every frame and only its frame-to-frame motion
    is used; each ArUco pose pulls the position GAIN of the way towards itself (~0.2 s time constant at 30 fps), so
    the drift is taken out smoothly instead of in one step."""
    GAIN = 0.15
    GAIN_NO_FLOW = 0.5     # ArUco without flow motion this frame: follow it faster
    STALE = 1.0            # s without an ArUco pose -> no estimate
    REANCHOR = 1.0         # s between re-seeds of the flow from an ArUco pose (keeps its depth / scale right)
    MAX_STEP = 0.15        # m of flow motion in one frame: lost, not moved

    def __init__(self, flow):
        self.flow = flow
        self.p = self.R = None
        self.t_fix = -1e9
        self.flow_prev = None

    def update(self, gray, now, aruco):
        """aruco: (R_wc, pos) accepted this frame, or None. Returns (R_wc, pos, source) or None."""
        r = self.flow.track(gray, now, max_age=float("inf")) if self.flow.gray is not None else None
        moved = None
        if r is not None and self.flow_prev is not None:
            moved = r[1] - self.flow_prev
            if np.linalg.norm(moved) > self.MAX_STEP:
                r = moved = None
                self.flow.gray = None
        self.flow_prev = r[1].copy() if r is not None else None
        if aruco is not None and (r is None or now - self.flow.t_anchor > self.REANCHOR):
            self.flow.anchor(gray, aruco[0], aruco[1], now)
            self.flow_prev = aruco[1].copy()
        if self.p is None or now - self.t_fix > self.STALE:
            self.p = None
            if aruco is None:
                return None
            self.R, self.p, self.t_fix = aruco[0], aruco[1].copy(), now
            return self.R, self.p.copy(), "aruco"
        if moved is not None:
            self.p = self.p + moved
            self.R = r[0]
        if aruco is not None:
            self.p = self.p + (self.GAIN if moved is not None else self.GAIN_NO_FLOW) * (aruco[1] - self.p)
            self.R, self.t_fix = aruco[0], now
        if moved is None and aruco is None:
            return None
        return self.R, self.p.copy(), "aruco" if aruco is not None and moved is None else "flow+aruco" if aruco is not None else "flow"
