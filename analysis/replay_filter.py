"""Replay recorded FPV videos through two room_pose.py versions; compare accepted frames while airborne and suspicious steps.

python3 replay_filter.py old_room_pose.py new_room_pose.py video.avi [...]   (scripts run from ../position)
"""
import csv, subprocess, sys, os, glob, numpy as np
POS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'position')
def run(script, video):
    before = set(glob.glob(os.path.join(POS, 'room_pose_2*.csv')))
    subprocess.run(['python3', os.path.abspath(script), '--video', os.path.abspath(video), '--headless', '--record', '--target', '-1.2,0,0.2'], cwd=POS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    new = sorted(set(glob.glob(os.path.join(POS, 'room_pose_2*.csv'))) - before)[-1]
    rows = [r for r in csv.reader(open(new, newline=''))][1:]
    os.remove(new); os.remove(new[:-4] + '.avi')
    return rows
def stats(rows, window):
    t = np.array([float(r[0]) for r in rows]); ok = np.array([r[-1].strip() == 'ok' for r in rows])
    m = (t >= window[0]) & (t <= window[1])
    P = np.array([[float(r[k]) for k in (1, 2, 3)] for r, o in zip(rows, ok) if o])
    tt = t[ok]; mm = (tt >= window[0]) & (tt <= window[1])
    step = np.linalg.norm(np.diff(P[mm], axis=0), axis=1); dt = np.diff(tt[mm])
    return 100 * ok[m].mean(), int((step > 0.05 + 1.5 * dt).sum())
for video in sys.argv[3:]:
    a = run(sys.argv[1], video)
    t = np.array([float(r[0]) for r in a]); ok = [r[-1].strip() == 'ok' for r in a]
    z = np.array([float(r[3]) if o else 0 for r, o in zip(a, ok)])
    air = t[z > 0.12]
    if len(air) == 0: print(os.path.basename(video), 'never above 12 cm'); continue
    w = (air[0] - 0.3, air[-1] + 0.5)
    b = run(sys.argv[2], video)
    print('%s  flight %.1f-%.1f s   old: %.0f%% frames used, %d impossible steps   new: %.0f%%, %d' % (os.path.basename(video), *w, *stats(a, w), *stats(b, w)))
