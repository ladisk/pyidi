"""Composite the SelectionGUI grabs (gui_capture.py) into the release scene.

Places the real window on the dark canvas of the other scenes and adds what a headless grab
cannot show: the pointer, click ripples, and the step list with the point count.
"""
import json
import os
import shutil
import subprocess

import imageio.v3 as iio
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRABS = os.path.join(HERE, "gui_frames")
timeline = json.load(open(os.path.join(GRABS, "timeline.json")))
records = timeline["records"]
spin_x, spin_y, spin_w, spin_h = timeline["spin_rect"]
grabs = [iio.imread(os.path.join(GRABS, r["file"])) for r in records]
WIN_H, WIN_W = grabs[0].shape[:2]

FPS = 20
W, H, DPI = 1200, 675, 100
BG, FG, MUTED, EDGE = "#0b0f14", "#e6edf3", "#8b949e", "#30363d"
CYAN, MAGENTA, AMBER, MINT = "#4cc9f0", "#f72585", "#ffd166", "#7bf1a8"

# timing, in frames
OPEN_HOLD, TAB_MOVE, TAB_HOLD = 16, 10, 6
VERTEX_MOVE, VERTEX_HOLD, POLY_HOLD = 7, 3, 10
SPIN_MOVE, SWEEP_STEP, END_HOLD, RIPPLE = 9, 2, 30, 7

avail = {f.name for f in font_manager.fontManager.ttflist}
for name in ("Inter", "IBM Plex Sans", "Source Sans 3", "Noto Sans", "DejaVu Sans"):
    if name in avail:
        plt.rcParams["font.family"] = name
        break
MONO = next((n for n in ("JetBrains Mono", "Fira Code", "Noto Sans Mono", "DejaVu Sans Mono") if n in avail),
            "monospace")

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
fig.text(0.035, 0.915, "pyIDI 1.4.0", color=FG, fontsize=30, fontweight="bold", va="center")
fig.text(0.035, 0.852, "The same steps in the SelectionGUI window", color=MUTED, fontsize=15, va="center")
fig.text(0.035, 0.035, "gui = SelectionGUI(video, subset_size=21)", color=MUTED, fontsize=11, family=MONO,
         va="center")
fig.text(0.97, 0.035, 'pip install -U "pyidi[qt]"', color=CYAN, fontsize=15, family=MONO, ha="right",
         va="center")

# ---- the window, as a card
CARD_LEFT, CARD_TOP, CARD_H = 24, 124, 498
scale = CARD_H / WIN_H
card_w = WIN_W * scale
axC = fig.add_axes([CARD_LEFT / W, 1 - (CARD_TOP + CARD_H) / H, card_w / W, CARD_H / H])
axC.set_xticks([])
axC.set_yticks([])
for spine in axC.spines.values():
    spine.set_color(EDGE)
    spine.set_linewidth(1.5)
card = axC.imshow(grabs[0], extent=(0, WIN_W, WIN_H, 0), interpolation="antialiased")
axC.set_xlim(0, WIN_W)
axC.set_ylim(WIN_H, 0)

ARROW = np.array([(0, 0), (0, 17), (4.2, 13.2), (7, 19.6), (9.6, 18.5), (6.9, 12.2), (12.3, 12.2)]) * 1.35 / scale
pointer = Polygon(ARROW, closed=True, facecolor="white", edgecolor="black", lw=1.3, zorder=10)
axC.add_patch(pointer)
ripples = [Circle((0, 0), 1, fill=False, lw=2.2, edgecolor=CYAN, alpha=0, zorder=9) for _ in range(3)]
for r in ripples:
    axC.add_patch(r)

# ---- right: the steps and the count, as in the animated scene
x0 = CARD_LEFT + card_w + 30
axR = fig.add_axes([x0 / W, 0.1, (W - 24 - x0) / W, 0.7])
axR.set_xlim(0, 1)
axR.set_ylim(0, 1)
axR.axis("off")
STEPS = [("Open", "opens with the whole frame scored", AMBER),
         ("Mask", "click the corners of a polygon", CYAN),
         ("Evaluate + select", "separation sets how many points", MINT)]
step_art = []
for k, (title, desc, accent) in enumerate(STEPS):
    yk = 0.93 - k * 0.15
    ring = axR.plot([0.045], [yk], "o", ms=24, color=BG, mec=MUTED, mew=1.6, clip_on=False)[0]
    num = axR.text(0.045, yk, str(k + 1), ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")
    ttl = axR.text(0.14, yk + 0.012, title, va="center", fontsize=16, color=MUTED, fontweight="bold")
    dsc = axR.text(0.14, yk - 0.052, desc, va="center", fontsize=11.5, color=MUTED)
    step_art.append((ring, num, ttl, dsc, accent))
axR.plot([0, 1], [0.5, 0.5], color=MUTED, lw=0.6, alpha=0.3)
count_text = axR.text(0.0, 0.36, "", va="center", fontsize=36, color=MINT, fontweight="bold")
axR.text(0.0, 0.265, "points selected", va="center", fontsize=12.5, color=MUTED)
sep_text = axR.text(0.0, 0.16, "", va="center", fontsize=13, color=FG, family=MONO)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


# ---- timeline
states, clicks = [], []
cursor = np.array([WIN_W * 0.45, WIN_H * 0.8])
meta = dict(grab=0, step=1, done=(), sep=11)


def hold(n):
    for _ in range(n):
        states.append(dict(meta, cursor=cursor.copy()))


def move(target, n):
    global cursor
    start = cursor.copy()
    for k in range(n):
        states.append(dict(meta, cursor=start + (np.asarray(target, float) - start) * ease((k + 1) / n)))
    cursor = np.asarray(target, float)


def click(big=True):
    clicks.append((len(states), cursor.copy(), big))


def sep_of(record, default):
    return int(record["label"].split()[-1]) if record["label"].startswith("separation") else default


hold(OPEN_HOLD)
i = 1
move(records[i]["cursor"], TAB_MOVE)                        # to the Mask tab
click()
meta.update(grab=i, step=2, done=(1,))
hold(TAB_HOLD)
i += 1
while records[i]["label"] == "polygon vertex":             # the polygon
    move(records[i]["cursor"], VERTEX_MOVE)
    click()
    meta.update(grab=i)
    hold(VERTEX_HOLD)
    i += 1
hold(POLY_HOLD)
move(records[i]["cursor"], TAB_MOVE)                        # back to Evaluate + select
click()
meta.update(grab=i, step=3, done=(1, 2))
hold(TAB_HOLD)
i += 1
up = (spin_x + spin_w - 9, spin_y + spin_h * 0.28)
down = (spin_x + spin_w - 9, spin_y + spin_h * 0.72)
first = sep_of(records[i], 11)
move(up if first > meta["sep"] else down, SPIN_MOVE)
for record in records[i:]:                                  # the separation, click by click
    sep = sep_of(record, meta["sep"])
    cursor = np.asarray(up if sep > meta["sep"] else down, float)
    click(big=False)
    meta.update(grab=records.index(record), sep=sep)
    hold(SWEEP_STEP)
meta.update(done=(1, 2, 3))
hold(END_HOLD)
print(f"{len(states)} frames, {len(states) / FPS:.1f} s")


def draw(f, s):
    card.set_data(grabs[s["grab"]])
    pointer.set_xy(ARROW + s["cursor"])
    active = [(f - f0, pos, big) for f0, pos, big in clicks if 0 <= f - f0 < RIPPLE][-len(ripples):]
    for r in ripples:
        r.set_alpha(0)
    for r, (age, pos, big) in zip(ripples, active):
        t = age / (RIPPLE - 1)
        r.center = tuple(pos)
        r.set_radius(((6 + 22 * t) if big else (4 + 10 * t)) / scale)
        r.set_alpha(0.95 * (1 - t))
    for k, (ring, num, ttl, dsc, accent) in enumerate(step_art, start=1):
        is_active, is_done = s["step"] == k, k in s["done"]
        ring.set_markeredgecolor(accent if (is_active or is_done) else MUTED)
        ring.set_markerfacecolor(accent if is_done and not is_active else BG)
        num.set_color(BG if is_done and not is_active else (accent if is_active else MUTED))
        ttl.set_color(FG if (is_active or is_done) else MUTED)
        dsc.set_color(FG if is_active else MUTED)
    count_text.set_text(f"{records[s['grab']]['points']:,}")
    sep_text.set_text(f"separation  {s['sep']} px")


frames_dir = os.path.join(HERE, "frames_gui")
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir)
for f, s in enumerate(states):
    draw(f, s)
    fig.savefig(os.path.join(frames_dir, f"f{f:04d}.png"), facecolor=BG)

gif = os.path.join(HERE, "pyidi_1.4.0_selection_gui.gif")
mp4 = os.path.join(HERE, "pyidi_1.4.0_selection_gui.mp4")
pattern = os.path.join(frames_dir, "f%04d.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                "split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];"
                "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0", gif], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                f"pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color={BG}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-movflags", "+faststart", mp4], check=True)
for p in (gif, mp4):
    print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.2f} MB")
