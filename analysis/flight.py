"""Summarise an Air75 RAPTOR blackbox CSV (exported with blackbox_decode / Blackbox Explorer, debug_mode = RAPTOR).

python3 flight.py log.csv [step_s]      prints one line per step: attitude, laptop position (debug 4-6), predicted x
(debug 7), motors, battery, peak gyro. Caches the parsed log as log.npy / log_h.txt next to the CSV.
"""
import csv, numpy as np, sys, os
def load(path):
    base = os.path.splitext(path)[0]
    if os.path.exists(base + '.npy') and os.path.getmtime(base + '.npy') > os.path.getmtime(path):
        return np.load(base + '.npy'), open(base + '_h.txt').read().split('\n')
    rows = list(csv.reader(open(path)))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip() == 'loopIteration')
    h = [c.strip() for c in rows[hi]]
    A = np.array([[float(x) if x.strip() != '' else np.nan for x in r[:len(h)]] for r in rows[hi + 1:] if len(r) >= len(h)], dtype=float)
    np.save(base + '.npy', A); open(base + '_h.txt', 'w').write('\n'.join(h))
    return A, h
if __name__ == '__main__':
    A, h = load(sys.argv[1]); step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    c = lambda n: A[:, h.index(n)]
    t = (c('time') - c('time')[0]) / 1e6
    q = np.stack([c('debug[%d]' % i) for i in range(4)], 1) / 1e4; w, x, y, z = q.T
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))); pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    on = np.abs(np.diff(c('debug[0]'), prepend=c('debug[0]')[0])) + np.abs(np.diff(c('debug[3]'), prepend=c('debug[3]')[0])) > 0
    print('%s: %.1f s' % (sys.argv[1], t[-1]))
    print('    t act  roll pitch  yaw  rawx  rawy  rawz predx   m0   m1   m2   m3 vbat gyroR gyroP gyroY')
    for s in np.arange(0, t[-1], step):
        m = (t >= s) & (t < s + step)
        if not m.any(): continue
        f = lambda v: np.nanmean(v[m]); g = lambda n: np.nanmax(np.abs(c(n)[m]))
        print('%5.1f %3s %5.0f %5.0f %4.0f %5.0f %5.0f %5.0f %5.0f %4.0f %4.0f %4.0f %4.0f %4.0f %5.0f %5.0f %5.0f' % (s, 'ON' if on[m].mean() > 0.2 else '', f(roll), f(pitch), f(yaw),
              *[f(c('debug[%d]' % k)) for k in (4, 5, 6, 7)], *[f(c('motor[%d]' % k)) for k in range(4)], f(c('vbatLatest')), g('gyroADC[0]'), g('gyroADC[1]'), g('gyroADC[2]')))
