"""Find tooth centre lines from the slots, track along them, save for rendering."""
import os

import numpy as np
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

import pyidi

OUT = os.path.dirname(os.path.abspath(__file__))
ref = np.load(os.path.join(OUT, "musicbox.npz"))["reference"]
H, W = ref.shape
PROFILES = {}


def profile(x, half=3):
    if x not in PROFILES:
        PROFILES[x] = gaussian_filter1d(ref[:, x - half:x + half + 1].mean(axis=1), 1.0)
    return PROFILES[x]


def dip(x, y):
    """Depth of the dark slot at (x, y): local median minus the minimum within +-2 px."""
    p = profile(x)
    yi = int(round(y))
    if yi - 10 < 0 or yi + 11 > H:
        return 0.0
    return float(np.median(p[yi - 10:yi + 11]) - p[yi - 2:yi + 3].min())


# 1. slot positions at x = 520, and one tilt shared by all of them
seeds, _ = find_peaks(-profile(520)[70:480], distance=15, prominence=8)
seeds = seeds + 70
B0 = -0.094
xs_fit = np.arange(380, 541, 8)
slopes, fits = [], []
for y0 in seeds:
    xs, ys = [], []
    for x in xs_fit:
        yp = y0 + B0 * (x - 520)
        lo = int(round(yp)) - 4
        seg = profile(x)[lo:lo + 9]
        k = int(np.argmin(seg))
        if 0 < k < 8 and dip(x, lo + k) > 6:
            xs.append(x); ys.append(lo + k)
    fits.append((np.array(xs), np.array(ys)))
    if len(xs) >= 12:
        slopes.append(np.polyfit(xs, ys, 1)[0])
B = float(np.median(slopes))
print(f"shared tilt: {B:.4f} ({np.degrees(np.arctan(-B)):.1f} deg), from {len(slopes)} slots")

slots = []
for y0, (xs, ys) in zip(seeds, fits):
    a = float(np.median(ys - B * xs)) if len(xs) else y0 - B * 520
    # 2. the slot ends in a small bright hole, or at the screw head: the rightmost bright
    #    spot along the line with dark slot to its right. Fall back to where the dip fades.
    xline = np.arange(520, 79, -2)
    depth = np.array([dip(x, a + B * x) for x in xline])
    ref_depth = np.median(depth[:40])
    good = uniform_filter1d((depth > 0.12 * ref_depth).astype(float), 15) > 0.5
    fade = int(xline[np.argmin(good)]) if not good.all() else int(xline[-1])
    xs_dot = np.arange(90, 330)
    bright = np.array([ref[max(0, int(round(a + B * x)) - 3):int(round(a + B * x)) + 4, x].max()
                       for x in xs_dot])
    thr = np.percentile(ref, 99.0)
    cand = [k for k in range(len(xs_dot) - 25)
            if bright[k] > thr and bright[k + 6:k + 25].max() < 0.5 * thr]
    root = int(xs_dot[cand[-1]]) + 5 if cand else fade
    slots.append(dict(a=a, root=root, y500=a + B * 500, depth=ref_depth))
    print(f"slot y500={a + B * 500:6.1f}: root x = {root} "
          f"({'bright end' if cand else 'dip fade'}; fade at {fade})")

# the roots form a staircase; a slot whose fade point disagrees with its neighbours (lost in
# a reflection, or the comb edge) takes the value of a line fitted through the others
from scipy.ndimage import median_filter  # noqa: E402

yr = np.array([s["y500"] for s in slots])
r = np.array([s["root"] for s in slots], float)
inlier = (yr > 105) & (yr < 440) & (np.abs(r - median_filter(r, size=3, mode="nearest")) < 30)
c1, c0 = np.polyfit(yr[inlier], r[inlier], 1)
for s, ok in zip(slots, inlier):
    if not ok:
        s["root"] = int(round(c0 + c1 * s["y500"]))
print(f"root fit: x = {c0:.0f} + {c1:.3f} * y500; replaced slots at y500 =", [round(v) for v in yr[~inlier]])

# 3. teeth between consecutive slots, only the 15 of the comb
teeth = []
for s0, s1 in zip(slots[:-1], slots[1:]):
    gap = s1["y500"] - s0["y500"]
    mid = (s0["y500"] + s1["y500"]) / 2
    if 15 < gap < 30 and 100 < mid < 425:
        teeth.append(dict(a=(s0["a"] + s1["a"]) / 2, root=max(s0["root"], s1["root"]), y500=mid))
print(f"{len(teeth)} teeth, y at x=500:", [round(t["y500"]) for t in teeth])

TIP, STEP = 530, 12
tooth_points, owner = [], []
for i, t in enumerate(teeth):
    for x in range(t["root"] + 8, TIP + 1, STEP):
        tooth_points.append([t["a"] + B * x, x])
        owner.append(i)
tooth_points = np.array(tooth_points)
owner = np.array(owner)

video = pyidi.datasets.load_music_box()
fps = video.fps
lk = pyidi.LucasKanade(video)
lk.set_points(np.round(tooth_points).astype(int))
lk.configure(roi_size=(21, 25), pad=3, int_order=3, reference_image=0, show_pbar=False)
d = lk.get_displacements(autosave=False).copy()
failed = np.where(np.isnan(d).any(axis=(1, 2)))[0]
print("failed points:", failed, tooth_points[failed].round() if len(failed) else "")

# fill a lost point from its neighbours along the same tooth
for i in range(len(teeth)):
    idx = np.where(owner == i)[0]
    xs = tooth_points[idx, 1]
    for comp in (0, 1):
        block = d[idx, :, comp]
        for f in np.where(np.isnan(block).any(axis=0))[0]:
            ok = ~np.isnan(block[:, f])
            block[~ok, f] = np.interp(xs[~ok], xs[ok], block[ok, f])
        d[idx, :, comp] = block

sos = butter(4, 250, "highpass", fs=fps, output="sos")
yhp = sosfiltfilt(sos, d[:, :, 0] - d[:, :, 0].mean(axis=1, keepdims=True), axis=1)

NOTES = "C C# D D# E F F# G G# A A# B".split()
freq, amp, notes = [], [], []
for i, t in enumerate(teeth):
    s = yhp[np.where(owner == i)[0][-1]]
    win = np.hanning(len(s))
    f = np.fft.rfftfreq(len(s), 1 / fps)
    A = np.abs(np.fft.rfft(s * win)) * 2 / win.sum()
    k0 = np.searchsorted(f, 300)
    k = k0 + np.argmax(A[k0:])
    left, centre, right = np.log(A[k - 1:k + 2])
    fpk = (k + 0.5 * (left - right) / (left - 2 * centre + right)) * (f[1] - f[0])
    m = int(round(12 * np.log2(fpk / 440) + 69))
    freq.append(fpk); amp.append(A[k]); notes.append(f"{NOTES[m % 12]}{m // 12 - 1}")
    print(f"tooth y500={t['y500']:5.0f} root={t['root']:3d} pts={len(np.where(owner == i)[0]):2d}: "
          f"{fpk:7.1f} Hz {notes[-1]:>4}  peak amp {A[k]:.3f} px, tip std {s.std():.3f} px")

np.savez_compressed(os.path.join(OUT, "teeth.npz"), points=tooth_points, owner=owner, y=yhp,
                    fps=fps, reference=ref, slope=B,
                    teeth=np.array([[t["a"], t["root"], t["y500"]] for t in teeth]),
                    freq=np.array(freq), amp=np.array(amp), notes=np.array(notes))

# overlay for checking the geometry
lo, hi = np.percentile(ref, [1, 99.5])
img = (np.stack([np.clip((ref - lo) / (hi - lo), 0, 1)] * 3, -1) * 255).astype(np.uint8)
for s in slots:
    for x in range(s["root"], 560):
        yy = int(round(s["a"] + B * x))
        if 0 <= yy < H:
            img[yy, x] = (255, 60, 60)
for py, px in tooth_points:
    img[int(round(py)) - 1:int(round(py)) + 2, int(px) - 1:int(px) + 2] = (60, 255, 255)
iio.imwrite(os.path.join(OUT, "geometry_check.png"), img)
print("saved")
