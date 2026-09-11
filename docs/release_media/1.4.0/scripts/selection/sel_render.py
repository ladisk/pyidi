"""Render the automatic feature selection scene for the pyIDI 1.4.0 release post."""
import os
import shutil
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "selection.npz"))
ref, score, mask, polygon, whole = z["reference"], z["score"], z["mask"], z["polygon"], z["whole"]
t_eval = float(z["t_eval_ms"])
sel = {int(k[4:]): z[k] for k in z.files if k.startswith("sel_")}

FPS = 20
W, H, DPI = 1200, 675, 100
BG, FG, MUTED = "#0b0f14", "#e6edf3", "#8b949e"
CYAN, MAGENTA, AMBER, MINT = "#4cc9f0", "#f72585", "#ffd166", "#7bf1a8"
SWEEP = [11, 16, 30, 16, 9, 6, 9, 12]
SEP_MIN, SEP_MAX = 6, 30

avail = {f.name for f in font_manager.fontManager.ttflist}
for name in ("Inter", "IBM Plex Sans", "Source Sans 3", "Noto Sans", "DejaVu Sans"):
    if name in avail:
        plt.rcParams["font.family"] = name
        break
MONO = next((n for n in ("JetBrains Mono", "Fira Code", "Noto Sans Mono", "DejaVu Sans Mono") if n in avail),
            "monospace")

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
fig.text(0.035, 0.915, "pyIDI 1.4.0", color=FG, fontsize=30, fontweight="bold", va="center")
fig.text(0.035, 0.852, "Automatic feature selection: score the whole frame, keep the best features",
         color=MUTED, fontsize=15, va="center")
fig.text(0.035, 0.05, "In SelectionGUI, or headless from pyidi.selection", color=MUTED, fontsize=10.5,
         va="center")
fig.text(0.97, 0.05, "pip install -U pyidi", color=CYAN, fontsize=15, family=MONO, ha="right", va="center")

# ---- left: the frame, the score, the region and the points
X0, X1, Y0, Y1 = 90, 600, 72, 470
axL = fig.add_axes([0.015, 0.075, 0.585, 0.72])
axL.set_facecolor(BG)
axL.axis("off")
lo, hi = np.percentile(ref, [1, 99.7])
axL.imshow(np.clip((ref - lo) / (hi - lo), 0, 1) ** 0.9, cmap="gray", vmin=0, vmax=1, alpha=0.55,
           interpolation="bilinear")

valid = np.isfinite(score)
logs = np.log10(np.where(valid, np.maximum(score, 1e-12), np.nan))
s_lo, s_hi = np.nanpercentile(logs, [55, 99.8])
norm = np.nan_to_num(np.clip((logs - s_lo) / (s_hi - s_lo), 0, 1))
heat_rgba = plt.get_cmap("magma")(norm)
heat_base_alpha = np.where(valid, 0.15 + 0.8 * norm, 0.0)
heat_im = axL.imshow(heat_rgba, interpolation="bilinear")
row_idx = np.arange(ref.shape[0])[:, None]
scan_line = axL.plot([], [], color=AMBER, lw=2, alpha=0.9)[0]

poly_xy = np.vstack([polygon[:, ::-1], polygon[:1, ::-1]]).astype(float)   # (x, y), closed
N_EDGES = len(polygon)
poly_fill = axL.fill(poly_xy[:, 0], poly_xy[:, 1], color=CYAN, alpha=0.0, lw=0)[0]
poly_glow = axL.plot([], [], color=CYAN, lw=8, alpha=0.18, solid_capstyle="round")[0]
poly_line = axL.plot([], [], color=CYAN, lw=2.2, solid_capstyle="round")[0]
poly_vert = axL.plot([], [], "o", color=BG, mec=CYAN, mew=2, ms=7)[0]

whole_in_art = axL.plot([], [], "o", color=MINT, ms=2.6, mew=0)[0]
whole_out_art = axL.plot([], [], "o", color=MINT, ms=2.6, mew=0)[0]
sel_art = axL.plot([], [], "o", color=MINT, ms=3.0, mew=0)[0]
axL.set_xlim(X0, X1)
axL.set_ylim(Y1, Y0)
axL.set_aspect("equal")
axL.set_anchor("W")
inside_whole = mask[whole[:, 0], whole[:, 1]]

# ---- right: the three steps, the separation control and the count
axR = fig.add_axes([0.635, 0.1, 0.335, 0.7])
axR.set_xlim(0, 1)
axR.set_ylim(0, 1)
axR.axis("off")
STEPS = [("Evaluate", "Shi-Tomasi score of every pixel", AMBER),
         ("Mask", "draw the region you want points in", CYAN),
         ("Select", "best features, no two closer than the separation", MINT)]
step_art = []
for k, (title, desc, accent) in enumerate(STEPS):
    yk = 0.93 - k * 0.15
    ring = axR.plot([0.035], [yk], "o", ms=24, color=BG, mec=MUTED, mew=1.6, clip_on=False)[0]
    num = axR.text(0.035, yk, str(k + 1), ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")
    ttl = axR.text(0.1, yk + 0.012, title, va="center", fontsize=16, color=MUTED, fontweight="bold")
    dsc = axR.text(0.1, yk - 0.052, desc, va="center", fontsize=11.5, color=MUTED)
    step_art.append((ring, num, ttl, dsc, accent))
eval_value = axR.text(1.0, 0.93 + 0.012, "", ha="right", va="center", fontsize=13, color=AMBER, family=MONO)

axR.plot([0, 1], [0.5, 0.5], color=MUTED, lw=0.6, alpha=0.3)
axR.text(0.0, 0.42, "separation", va="center", fontsize=12.5, color=MUTED)
sep_value = axR.text(1.0, 0.42, "", ha="right", va="center", fontsize=14, color=FG, family=MONO)
SLIDER_Y = 0.35
axR.plot([0, 1], [SLIDER_Y, SLIDER_Y], color=MUTED, lw=4, alpha=0.25, solid_capstyle="round")
slider_fill = axR.plot([], [], color=MINT, lw=4, solid_capstyle="round")[0]
slider_knob = axR.plot([], [], "o", color=MINT, ms=13, mec=BG, mew=2)[0]
count_text = axR.text(0.0, 0.21, "", va="center", fontsize=36, color=MINT, fontweight="bold")
axR.text(0.0, 0.115, "points selected", va="center", fontsize=12.5, color=MUTED)
code_text = axR.text(0.0, 0.015, "", va="center", fontsize=11, color=FG, family=MONO)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def mix(c0, c1, t):
    a, b = np.array(matplotlib.colors.to_rgb(c0)), np.array(matplotlib.colors.to_rgb(c1))
    return tuple(a + (b - a) * t)


# ---- timeline: a list of states, one per frame
states = []
base = dict(step=0, done=(), scan=0.0, heat_out=1.0, heat_all=1.0, n_whole=0, out_dim=0.0, whole_in=True,
            poly=0.0, sep=11.0, sel=None, count=0, eval_val=False)


def add(n, **anim):
    for k in range(n):
        t = k / max(n - 1, 1)
        s = dict(base)
        s.update({key: (v(t) if callable(v) else v) for key, v in anim.items()})
        states.append(s)


n_in = int(inside_whole.sum())
add(16)                                                            # the frame alone
base.update(step=1)
add(24, scan=ease)                                                 # score sweeps down the frame
base.update(scan=1.0, eval_val=True)
add(8)
add(18, n_whole=lambda t: int(round(ease(t) * len(whole))), count=lambda t: int(round(ease(t) * len(whole))))
base.update(n_whole=len(whole), count=len(whole))
add(10)
base.update(step=2, done=(1,))
add(36, poly=lambda t: N_EDGES * t)                                # the region, corner by corner
base.update(poly=float(N_EDGES))
add(16, heat_out=lambda t: 1 - 0.88 * ease(t), out_dim=ease,
    count=lambda t: int(round(len(whole) + (n_in - len(whole)) * ease(t))))
base.update(heat_out=0.12, out_dim=1.0, count=n_in)
add(6)
base.update(step=3, done=(1, 2), whole_in=False, sel=11, count=len(sel[11]))
add(10, heat_all=lambda t: 1 - 0.5 * ease(t))
base.update(heat_all=0.5)
for prev, target in zip(SWEEP[:-1], SWEEP[1:]):                    # sweep the separation
    add(4, sep=lambda t, p=prev, q=target: float(np.exp(np.log(p) + (np.log(q) - np.log(p)) * ease(t))),
        sel=target, count=len(sel[target]))
    base.update(sep=float(target), sel=target, count=len(sel[target]))
    add(9)
base.update(done=(1, 2, 3))
add(44)                                                            # hold
print(f"{len(states)} frames, {len(states) / FPS:.1f} s")


def draw(s):
    scan_row = Y0 - 10 + s["scan"] * (Y1 - Y0 + 20)
    alpha = heat_base_alpha * (row_idx < scan_row) * np.where(mask, 1.0, s["heat_out"]) * s["heat_all"]
    heat_rgba[..., 3] = alpha
    heat_im.set_data(heat_rgba)
    if 0 < s["scan"] < 1:
        scan_line.set_data([X0, X1], [scan_row, scan_row])
    else:
        scan_line.set_data([], [])

    shown = whole[:s["n_whole"]]
    inside = inside_whole[:s["n_whole"]]
    if s["whole_in"]:
        whole_in_art.set_data(shown[inside, 1], shown[inside, 0])
    else:
        whole_in_art.set_data([], [])
    whole_out_art.set_data(shown[~inside, 1], shown[~inside, 0])
    whole_out_art.set_color(mix(MINT, MUTED, s["out_dim"]))
    whole_out_art.set_alpha(1 - 0.7 * s["out_dim"])
    if s["sel"] is not None:
        p = sel[s["sel"]]
        sel_art.set_data(p[:, 1], p[:, 0])
    else:
        sel_art.set_data([], [])

    k = int(np.floor(min(s["poly"], N_EDGES)))
    frac = s["poly"] - k
    if s["poly"] > 0:
        pts = poly_xy[:k + 1]
        if k < N_EDGES:
            pts = np.vstack([pts, poly_xy[k] + frac * (poly_xy[k + 1] - poly_xy[k])])
        for line in (poly_line, poly_glow):
            line.set_data(pts[:, 0], pts[:, 1])
        poly_vert.set_data(poly_xy[:min(k + 1, N_EDGES), 0], poly_xy[:min(k + 1, N_EDGES), 1])
    poly_fill.set_alpha(0.07 * s["out_dim"])

    for i, (ring, num, ttl, dsc, accent) in enumerate(step_art, start=1):
        active, done = s["step"] == i, i in s["done"]
        ring.set_markeredgecolor(accent if (active or done) else MUTED)
        ring.set_markerfacecolor(accent if done and not active else BG)
        num.set_color(BG if done and not active else (accent if active else MUTED))
        ttl.set_color(FG if (active or done) else MUTED)
        dsc.set_color(FG if active else MUTED)
    eval_value.set_text(f"{t_eval:.0f} ms" if s["eval_val"] else "")

    frac_s = (np.log(s["sep"]) - np.log(SEP_MIN)) / (np.log(SEP_MAX) - np.log(SEP_MIN))
    slider_fill.set_data([0, frac_s], [SLIDER_Y, SLIDER_Y])
    slider_knob.set_data([frac_s], [SLIDER_Y])
    sep_value.set_text(f"{int(round(s['sep']))} px")
    count_text.set_text(f"{s['count']:,}")
    code_text.set_text(f"select_points(frame, [region], separation={int(round(s['sep']))})")


frames_dir = os.path.join(HERE, "frames")
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir)
for f, s in enumerate(states):
    draw(s)
    fig.savefig(os.path.join(frames_dir, f"f{f:04d}.png"), facecolor=BG)

gif = os.path.join(HERE, "pyidi_1.4.0_selection.gif")
mp4 = os.path.join(HERE, "pyidi_1.4.0_selection.mp4")
pattern = os.path.join(frames_dir, "f%04d.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                "split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
                "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0", gif], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                f"pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color={BG}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-movflags", "+faststart", mp4], check=True)
for p in (gif, mp4):
    print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.2f} MB")
