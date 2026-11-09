"""Generate ``feature_selection.gif`` for the point-selection docs page.

Records ``SelectionGUI`` (``pyidi/GUIs/feature_selection.py``) on the first frame of the
music-box example recording (``pyidi.datasets.load_music_box``, downloaded and cached on
first use), in the order the interface is meant to be used:

1. the window as it opens -- already scored, with points over the whole frame,
   because the ``Whole image`` mask row is seeded on startup;
2. the ``Mask`` tab, with a polygon clicked corner by corner around the teeth of the
   comb; the points outside it drop to the dim tier as soon as it has area;
3. back on ``Evaluate + select``, the separation stepped down and up, which is the
   control that decides how many points there are.

That order is the pitch: the score comes first and the region trims it, rather
than a grid being placed and then filtered. ``make_selection_animation.py``
does the same job for the deprecated ``SelectionGUIOld``.

A grab of the window has no mouse in it, so the pointer and a ripple for every click
are drawn onto the grabs afterwards, at the positions the clicks were made.
``Show subsets`` is unchecked, because at small separations the subset rectangles
merge into a single block and hide the points.

Why the headless setup is needed
--------------------------------
``SelectionGUI`` is a full Qt application whose constructor calls ``show()`` and
then enters the event loop, ending in ``sys.exit(app.exec())`` unless ``sys.ps1``
is set. To build the window, drive it and grab pixels from it in a plain script:

* ``QT_QPA_PLATFORM=offscreen`` must be set *before* Qt is imported, so Qt
  renders into its software framebuffer instead of opening a display;
* ``sys.ps1`` is set before construction, so the constructor takes the
  interactive branch rather than ``sys.exit(...)``;
* ``QtWidgets.QApplication.exec`` is monkeypatched to a no-op, because the
  interactive branch still calls ``app.exec()``, which would block with nothing
  driving it.

The window is then driven through the same calls a real click makes -- see
``on_mouse_click``: ``add_vertex``, ``_retire_whole_image``, ``refresh``.

Run with:

    QT_QPA_PLATFORM=offscreen python docs/source/quick_start/make_feature_selection_animation.py

This (re)writes ``docs/source/quick_start/feature_selection.gif`` in place. Encoding
needs ``ffmpeg`` on the ``PATH``.
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

import pyidi  # noqa: E402
from pyidi.GUIs.feature_selection import STEP_FIND, STEP_MASK, SelectionGUI  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "feature_selection.gif")

#: Polygon corners as ``(row, col)``, around the teeth of the comb.
POLYGON_VERTICES = [(134, 128), (93, 552), (427, 552), (455, 240), (300, 168)]

#: Separation values on the ``Evaluate + select`` tab, one spin-box click each.
SEPARATIONS = [11, 12, 13, 14, 16, 18, 20, 22, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12]

#: Fixed rather than merely resized: the status bar changes length as the window is
#: driven, and a resize between two grabs would give frames of different shapes.
WINDOW_SIZE = (1150, 720)
OUTPUT_WIDTH = 960
FPS = 20

#: How many frames each part of the animation lasts.
OPEN_HOLD, TAB_MOVE, TAB_HOLD = 16, 10, 6
VERTEX_MOVE, VERTEX_HOLD, POLYGON_HOLD = 7, 3, 10
SPIN_MOVE, SEPARATION_HOLD, END_HOLD, RIPPLE = 9, 2, 30, 7

#: Mouse pointer outline in window pixels, tip at the origin.
POINTER = np.array([(0, 0), (0, 17), (4.2, 13.2), (7, 19.6), (9.6, 18.5), (6.9, 12.2), (12.3, 12.2)]) * 1.3
RIPPLE_COLOUR = (76, 201, 240)


def grab_frame(window):
    """Process pending Qt events and grab the window as an (H, W, 3) uint8 array."""
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))[..., :3].copy()


def image_to_window(window, row, col):
    """Window pixel at which the image position ``(row, col)`` is drawn."""
    scene = window.view.mapViewToScene(QtCore.QPointF(col, row))
    point = window.pg_widget.mapTo(window, window.pg_widget.mapFromScene(scene))
    return point.x(), point.y()


def widget_centre(window, widget):
    point = widget.mapTo(window, widget.rect().center())
    return point.x(), point.y()


def capture():
    """Drive the window through the three steps: one grab, pointer target and kind per step."""
    sys.ps1 = ">>> "  # Make SelectionGUI think it's running interactively.
    QtWidgets.QApplication.exec = lambda self=None: 0  # Neutralise the blocking event loop.

    window = SelectionGUI(pyidi.datasets.load_music_box(), subset_size=21)
    window.setFixedSize(*WINDOW_SIZE)
    window.show_subsets.setChecked(False)
    QtWidgets.QApplication.processEvents()

    # 1. As opened: the whole frame is masked, so there are points before anything is drawn.
    window.select_step(STEP_FIND)
    window.refresh()
    steps = [(grab_frame(window), None, "open")]

    # 2. The Mask tab, with no row selected yet -- otherwise the tab rings every point of
    # the seeded "Whole image" row -- then the polygon corner by corner, as
    # on_mouse_click does it for a left click with the polygon tool active.
    target = widget_centre(window, window.step_buttons[STEP_MASK])
    window.active_index = None
    window.select_step(STEP_MASK)
    window.select_tool("polygon")
    window.refresh()
    steps.append((grab_frame(window), target, "tab"))

    for vertex in POLYGON_VERTICES:
        window.add_vertex(vertex)
        window._retire_whole_image()
        window.refresh()
        steps.append((grab_frame(window), image_to_window(window, *vertex), "vertex"))

    # 3. Back to Evaluate + select, clicking the separation up and down.
    target = widget_centre(window, window.step_buttons[STEP_FIND])
    window.select_step(STEP_FIND)
    window.refresh()
    steps.append((grab_frame(window), target, "tab"))

    spin = window.separation_spin
    corner = spin.mapTo(window, QtCore.QPoint(0, 0))
    up = (corner.x() + spin.width() - 9, corner.y() + spin.height() * 0.28)
    down = (corner.x() + spin.width() - 9, corner.y() + spin.height() * 0.72)
    for previous, separation in zip(SEPARATIONS[:-1], SEPARATIONS[1:]):
        spin.setValue(separation)
        window.refresh()
        print(f"separation {separation:>3} px -> {len(window.get_points()):>4} points")
        steps.append((grab_frame(window), up if separation > previous else down, "spin"))
    return steps


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def timeline(steps):
    """Per output frame the grab to show and the pointer position, and the clicks as (frame, position, big)."""
    frames, clicks = [], []
    height, width = steps[0][0].shape[:2]
    pointer = np.array([width * 0.45, height * 0.8])
    shown = 0

    def hold(n):
        frames.extend([(shown, pointer.copy())] * n)

    def move(target, n):
        nonlocal pointer
        start, target = pointer.copy(), np.asarray(target, float)
        frames.extend((shown, start + (target - start) * ease((k + 1) / n)) for k in range(n))
        pointer = target

    hold(OPEN_HOLD)
    for index, (_, target, kind) in enumerate(steps[1:], start=1):
        if kind != "spin":
            move(target, TAB_MOVE if kind == "tab" else VERTEX_MOVE)
        elif steps[index - 1][2] != "spin":
            move(target, SPIN_MOVE)
        pointer = np.asarray(target, float)
        clicks.append((len(frames), pointer.copy(), kind != "spin"))
        shown = index
        hold({"tab": TAB_HOLD, "vertex": VERTEX_HOLD, "spin": SEPARATION_HOLD}[kind])
        if kind == "vertex" and steps[index + 1][2] != "vertex":
            hold(POLYGON_HOLD)
    hold(END_HOLD)
    return frames, clicks


def with_pointer(grab, pointer, clicks, frame):
    """The grab with the pointer, and a fading ripple for every recent click, drawn on top."""
    image = Image.fromarray(grab).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for start, (x, y), big in clicks:
        age = frame - start
        if 0 <= age < RIPPLE:
            t = age / (RIPPLE - 1)
            radius = 6 + 22 * t if big else 4 + 10 * t
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         outline=RIPPLE_COLOUR + (int(240 * (1 - t)),), width=3)
    draw.polygon([tuple(p) for p in POINTER + pointer], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
    return Image.alpha_composite(image, overlay).convert("RGB")


def main():
    steps = capture()
    frames, clicks = timeline(steps)
    with tempfile.TemporaryDirectory() as tmp:
        for index, (shown, pointer) in enumerate(frames):
            with_pointer(steps[shown][0], pointer, clicks, index).save(os.path.join(tmp, f"{index:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
             "-vf", f"scale={OUTPUT_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];"
                    "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
             "-loop", "0", OUT_PATH],
            check=True)
    print(f"Wrote {OUT_PATH} ({len(frames)} frames, {os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
