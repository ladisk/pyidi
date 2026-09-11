"""Render the pyIDI 1.4.0 speed-race GIF (and a single-play MP4).

All timings are the measured values in CHANGELOG.md, 1.4.0 section; nothing is re-measured.
"""
import os
import shutil
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import pyidi  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "..", "..", "..", "..", "..", "data", "data_synthetic.cih")

W, H, DPI, FPS = 1200, 675, 100, 25
BG, FG, MUTED = "#0b0f14", "#e6edf3", "#8b949e"
PALETTE = ["#4cc9f0", "#f72585", "#ffd166", "#7bf1a8"]
OLD_C, NEW_C = PALETTE[2], PALETTE[0]

T_OLD, T_NEW = 7.94, 0.10      # CHANGELOG: data_synthetic.cih, 200 points, 101 frames
N_POINTS, BATCHES = 200, 8
INTRO, BOTH_DONE, END = 0.5, 0.8, 2.6   # seconds of hold

avail = {f.name for f in font_manager.fontManager.ttflist}
for name in ("Inter", "IBM Plex Sans", "Source Sans 3", "Noto Sans", "DejaVu Sans"):
    if name in avail:
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        break
MONO = next((n for n in ("JetBrains Mono", "Fira Code", "Noto Sans Mono", "DejaVu Sans Mono") if n in avail),
            "monospace")

frame = np.asarray(pyidi.VideoReader(DATA).get_frame(0)).astype(float)
fh, fw = frame.shape
rows, cols = 10, 20
gy, gx = np.meshgrid(np.linspace(0.14 * fh, 0.86 * fh, rows), np.linspace(0.05 * fw, 0.95 * fw, cols),
                     indexing="ij")
gy, gx = gy.ravel(), gx.ravel()

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
fig.text(0.035, 0.915, "pyIDI 1.4.0", color=FG, fontsize=30, fontweight="bold", va="center")
fig.text(0.035, 0.852, "Lucas-Kanade, compiled with numba and parallel over points",
         color=MUTED, fontsize=15, va="center")
clock = fig.text(0.965, 0.915, "", color=MUTED, fontsize=14, family=MONO, ha="right", va="center")
fig.text(0.965, 0.05, "pip install -U pyidi", color=PALETTE[0], fontsize=15, family=MONO,
         ha="right", va="center")
case = fig.text(0.035, 0.05, "data_synthetic.cih · 200 points · 101 frames", color=MUTED,
                fontsize=10.5, va="center")

race = [case]


def panel(x0, label, sub, accent):
    """One side of the race: header, thumbnail with the point grid, progress bar, elapsed time."""
    pw = 0.43
    head = fig.text(x0, 0.745, label, color=accent, fontsize=24, fontweight="bold", va="center")
    subt = fig.text(x0 + 0.085, 0.745, sub, color=MUTED, fontsize=13, va="center")
    ax = fig.add_axes([x0, 0.3, pw, 0.4])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() / 0.32, interpolation="bilinear",
              extent=(0, fw, fh, 0))
    ax.set_xlim(0, fw)
    ax.set_ylim(fh, 0)
    glow = ax.scatter(gx, gy, s=70, c=accent, alpha=0.0, linewidths=0)
    dots = ax.scatter(gx, gy, s=16, c=[MUTED], alpha=0.9, linewidths=0)
    bar_ax = fig.add_axes([x0, 0.225, pw, 0.022])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.5",
                                    mutation_aspect=0.04, color=MUTED, alpha=0.18, lw=0))
    fill = bar_ax.add_patch(FancyBboxPatch((0, 0), 0, 1, boxstyle="round,pad=0,rounding_size=0.5",
                                           mutation_aspect=0.04, color=accent, lw=0))
    count = fig.text(x0, 0.165, "", color=MUTED, fontsize=13, family=MONO, va="center")
    elapsed = fig.text(x0 + pw, 0.165, "", color=FG, fontsize=22, family=MONO, ha="right", va="center")
    race.extend([head, subt, ax, bar_ax, count, elapsed])
    return dict(glow=glow, dots=dots, fill=fill, count=count, elapsed=elapsed, accent=accent)


old = panel(0.035, "1.3.3", "previous release", OLD_C)
new = panel(0.535, "1.4.0", "this release", NEW_C)

# end card
end = []
end.append(fig.text(0.5, 0.64, "77× faster", color=NEW_C, fontsize=66, fontweight="bold",
                    ha="center", va="center"))
end.append(fig.text(0.5, 0.525, "Lucas-Kanade, same machine, identical results", color=MUTED,
                    fontsize=15, ha="center", va="center"))
TX = (0.16, 0.66, 0.76, 0.86)
table = [("case", "1.3.3", "1.4.0", "speedup"),
         ("data_synthetic.cih, 200 points, 101 frames", "7.94 s", "0.10 s", "77×"),
         ("synthetic 512×512, 400 points, 150 frames", "21.04 s", "0.24 s", "89×"),
         ("data_synthetic.mp4, 60 points, 10 frames", "2.16 s", "0.06 s", "36×")]
for r, row in enumerate(table):
    yy = 0.42 - r * 0.06
    for c, (x, cell) in enumerate(zip(TX, row)):
        header = r == 0
        color = MUTED if header else (NEW_C if c == 3 else FG)
        end.append(fig.text(x, yy, cell, color=color, fontsize=11 if header else 13.5,
                            family=None if (header or c == 0) else MONO,
                            ha="left" if c == 0 else "right", va="center"))
end.append(fig.text(0.16, 0.155, "Directional Lucas-Kanade: 38–65× faster, same results", color=MUTED,
                    fontsize=13, va="center"))
for a in end:
    a.set_visible(False)

lit_old_t = (np.arange(N_POINTS) + 1) / N_POINTS * T_OLD
batch = np.arange(N_POINTS) * BATCHES // N_POINTS
lit_new_t = (batch + 1) / BATCHES * T_NEW


def update(p, lit_t, t_end, t):
    lit = lit_t <= t + 1e-9
    n = int(lit.sum())
    colors = np.tile(matplotlib.colors.to_rgba(MUTED, 0.9), (N_POINTS, 1))
    colors[lit] = matplotlib.colors.to_rgba(p["accent"])
    p["dots"].set_facecolors(colors)
    p["glow"].set_alpha(None)
    glow = np.zeros((N_POINTS, 4))
    glow[:, :3] = matplotlib.colors.to_rgb(p["accent"])
    glow[lit, 3] = 0.22
    p["glow"].set_facecolors(glow)
    p["fill"].set_width(n / N_POINTS)
    done = t >= t_end - 1e-9
    p["count"].set_text(f"{n} / {N_POINTS} points" + ("   done" if done else ""))
    p["count"].set_color(p["accent"] if done else MUTED)
    p["elapsed"].set_text(f"{min(t, t_end):.2f} s")
    p["elapsed"].set_color(p["accent"] if done else FG)


frames_dir = os.path.join(OUT, "frames")
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir)
n_race = int(np.ceil(T_OLD * FPS))
timeline = ([0.0] * int(INTRO * FPS) + [min(i / FPS, T_OLD) for i in range(n_race + 1)]
            + [T_OLD] * int(BOTH_DONE * FPS))
k = 0
for t in timeline:
    update(old, lit_old_t, T_OLD, t)
    update(new, lit_new_t, T_NEW, t)
    clock.set_text(f"t = {t:4.2f} s")
    fig.savefig(os.path.join(frames_dir, f"f{k:04d}.png"), facecolor=BG)
    k += 1

for a in race:
    a.set_visible(False)
clock.set_visible(False)
for a in end:
    a.set_visible(True)
fig.savefig(os.path.join(frames_dir, f"f{k:04d}.png"), facecolor=BG)
for _ in range(int(END * FPS) - 1):
    k += 1
    shutil.copy(os.path.join(frames_dir, f"f{k - 1:04d}.png"), os.path.join(frames_dir, f"f{k:04d}.png"))
n_frames = k + 1
print(f"{n_frames} frames, {n_frames / FPS:.1f} s; race frames end at {len(timeline) - 1}")

gif = os.path.join(OUT, "pyidi_1.4.0_speed.gif")
mp4 = os.path.join(OUT, "pyidi_1.4.0_speed.mp4")
pattern = os.path.join(frames_dir, "f%04d.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
                "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0", gif], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf",
                f"pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color={BG}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4],
               check=True)
for p in (gif, mp4):
    print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.2f} MB")
