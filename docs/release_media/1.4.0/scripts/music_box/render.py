"""Render the pyIDI 1.4.0 release GIF (and an MP4) from teeth.npz."""
import os
import shutil
import subprocess

import numpy as np
import matplotlib
from scipy.signal import butter, resample, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(OUT, "teeth.npz"))
pts, owner, y, fps, ref = z["points"], z["owner"], z["y"], float(z["fps"]), z["reference"]
teeth, freq, amp, notes, slope = z["teeth"], z["freq"], z["amp"], z["notes"], float(z["slope"])

START = int(os.environ.get("START", 180))   # mid-record, clear of the band-pass start-up transient
N = int(os.environ.get("N", 120))           # 120 frames: D#5, A#5 and D#6 all close their cycles
FPS_GIF = 25
W, H, DPI = 1200, 675, 100
GAIN_PX = float(os.environ.get("GAIN_PX", 18.0))   # tip swing on screen, in image pixels
BAND_HZ = 80
UP = 6                                      # band-limited upsampling of the traces, display only

BG, FG, MUTED = "#0b0f14", "#e6edf3", "#8b949e"
PALETTE = ["#4cc9f0", "#f72585", "#ffd166", "#7bf1a8"]

avail = {f.name for f in font_manager.fontManager.ttflist}
for name in ("Inter", "IBM Plex Sans", "Source Sans 3", "Noto Sans", "DejaVu Sans"):
    if name in avail:
        plt.rcParams["font.family"] = name
        break
MONO = next((n for n in ("JetBrains Mono", "Fira Code", "Noto Sans Mono", "DejaVu Sans Mono") if n in avail),
            "monospace")

# the four loudest teeth that ring at different notes, ordered by pitch
chosen, seen = [], set()
for i in np.argsort(amp)[::-1]:
    if notes[i] not in seen and (owner == i).sum() >= 10:
        chosen.append(int(i))
        seen.add(notes[i])
    if len(chosen) == 4:
        break
chosen.sort(key=lambda i: freq[i])

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
fig.text(0.035, 0.915, os.environ.get("TITLE", "pyIDI 1.4.0"), color=FG, fontsize=30, fontweight="bold",
         va="center")
fig.text(0.035, 0.852, "A music-box comb ringing, measured from 7500 fps video",
         color=MUTED, fontsize=15, va="center")
clock = fig.text(0.97, 0.915, "", color=MUTED, fontsize=14, family=MONO, ha="right", va="center")

# left: the comb, with the highlighted teeth drawn from their measured displacement
X0, X1, Y0, Y1 = 90, 600, 72, 470
axL = fig.add_axes([0.015, 0.075, 0.585, 0.72])
axL.set_facecolor(BG)
axL.axis("off")
lo, hi = np.percentile(ref, [1, 99.7])
axL.imshow(np.clip((ref - lo) / (hi - lo), 0, 1) ** 0.9, cmap="gray", vmin=0, vmax=1, alpha=0.55,
           interpolation="bilinear")
axL.set_xlim(X0, X1)
axL.set_ylim(Y1, Y0)
axL.set_aspect("equal")
axL.set_anchor("W")

n_fine = (N - 1) * UP + 1
t_fine = np.arange(n_fine) / (fps * UP) * 1000
tooth_art, traces = [], []
TOP, BOTTOM, XL, XR = 0.775, 0.155, 0.635, 0.97
h = (TOP - BOTTOM) / 4
for k, i in enumerate(chosen):
    c = PALETTE[k]
    idx = np.where(owner == i)[0]
    a, root = teeth[i][0], teeth[i][1]
    xs = pts[idx, 1]
    sos = butter(4, [freq[i] - BAND_HZ, freq[i] + BAND_HZ], "bandpass", fs=fps, output="sos")
    yb_full = sosfiltfilt(sos, y[idx], axis=1)                    # (points, all frames), px
    yb = yb_full[:, START:START + N]
    tip = yb[-1]
    tip_fine = resample(yb_full[-1], yb_full.shape[1] * UP)[START * UP:START * UP + n_fine]
    gain = GAIN_PX / np.abs(tip).max()

    # clamped cantilever: no displacement and no slope at the root; points that do not follow
    # the fitted shape (a reflection, a scratch) are left out of the fit and not drawn
    xi = (xs - root) / (xs[-1] - root)
    basis = np.stack([xi ** 2, xi ** 3, xi ** 4], axis=1)
    keep = np.ones(len(xs), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(basis[keep], yb[keep], rcond=None)
        resid = np.sqrt(((basis @ coef - yb) ** 2).mean(axis=1))
        keep = resid < max(3 * np.median(resid[keep]), 0.04 * np.abs(tip).max())
    coef, *_ = np.linalg.lstsq(basis[keep], yb[keep], rcond=None)
    x_dense = np.linspace(root, xs[-1], 90)
    xi_d = (x_dense - root) / (xs[-1] - root)
    shape = np.stack([xi_d ** 2, xi_d ** 3, xi_d ** 4], axis=1) @ coef
    print(f"{notes[i]} {freq[i]:.0f} Hz, tooth y500={teeth[i][2]:.0f}: tip ±{np.abs(tip).max():.2f} px, "
          f"left out x = {xs[~keep].astype(int).tolist()}")

    glow = [axL.plot([], [], color=c, lw=lw, alpha=al, solid_capstyle="round", solid_joinstyle="round")[0]
            for lw, al in ((13, 0.06), (7, 0.16), (2.6, 1.0))]
    dots = axL.plot([], [], "o", color="white", ms=2.3, alpha=0.85)[0]
    axL.text(xs[-1] + 16, a + slope * xs[-1], notes[i], color=c, fontsize=13, fontweight="bold", va="center")
    tooth_art.append((x_dense, a + slope * x_dense, shape * gain,
                      xs[keep], a + slope * xs[keep], yb[keep] * gain, glow, dots))

    peak = np.abs(tip).max()
    ax = fig.add_axes([XL, TOP - (k + 1) * h + 0.01, XR - XL, h - 0.045])
    ax.set_facecolor(BG)
    ax.set_xlim(0, t_fine[-1])
    ax.set_ylim(-1.2 * peak, 1.2 * peak)
    ax.axis("off")
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.25)
    ax.plot(t_fine, tip_fine, color=c, lw=1.0, alpha=0.16)
    live = ax.plot([], [], color=c, lw=1.9, solid_joinstyle="round")[0]
    head = ax.plot([], [], "o", color=c, ms=6)[0]
    label_y = TOP - k * h - 0.022
    fig.text(XL, label_y, notes[i], color=c, fontsize=15, fontweight="bold", va="center")
    fig.text(XL + 0.052, label_y, f"{freq[i]:.0f} Hz", color=FG, fontsize=13, va="center")
    fig.text(XR, label_y, f"± {peak:.2f} px" if peak < 1 else f"± {peak:.1f} px", color=MUTED,
             fontsize=12, family=MONO, ha="right", va="center")
    traces.append((tip_fine, live, head))

fig.text(XL, BOTTOM - 0.012, "tooth-tip displacement at its note", color=MUTED, fontsize=10.5, va="center")
fig.text(XR, BOTTOM - 0.012, f"{t_fine[-1]:.0f} ms", color=MUTED, fontsize=10.5, ha="right", va="center")
fig.text(0.035, 0.05, "Tooth motion amplified for display", color=MUTED, fontsize=10.5, va="center")
fig.text(0.97, 0.05, os.environ.get("PIP", "pip install -U pyidi"), color=PALETTE[0], fontsize=15, family=MONO,
         ha="right", va="center")

frames_dir = os.path.join(OUT, "frames")
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir)
for f in range(N):
    for x_d, yc_d, shape, xs_k, yc_k, yb_k, glow, dots in tooth_art:
        for line in glow:
            line.set_data(x_d, yc_d + shape[:, f])
        dots.set_data(xs_k, yc_k + yb_k[:, f])
    j = f * UP
    for tip_fine, live, head in traces:
        live.set_data(t_fine[:j + 1], tip_fine[:j + 1])
        head.set_data([t_fine[j]], [tip_fine[j]])
    clock.set_text(f"{t_fine[j]:5.2f} ms")
    fig.savefig(os.path.join(frames_dir, f"f{f:04d}.png"), facecolor=BG)

gif = os.path.join(OUT, "pyidi_1.4.0.gif")
mp4 = os.path.join(OUT, "pyidi_1.4.0.mp4")
pattern = os.path.join(frames_dir, "f%04d.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS_GIF), "-i", pattern, "-vf",
                "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
                "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0", gif], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS_GIF), "-i", pattern, "-vf",
                f"loop=loop=3:size={N}:start=0,pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color={BG}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4],
               check=True)
for p in (gif, mp4):
    print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.2f} MB")
