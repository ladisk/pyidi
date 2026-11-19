"""Drive SelectionGUI headless on the music-box frame and grab the window at every step.

Writes ``gui_frames/NNNN.png`` (window grabs) and ``gui_frames/timeline.json``: one record
per grab, with where a pointer would be (window pixels), whether that step is a click,
and the number of selected points. The headless setup follows
``docs/source/quick_start/make_feature_selection_animation.py``.

Run with:

    QT_QPA_PLATFORM=offscreen python gui_capture.py
"""
import json
import os
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

import pyidi  # noqa: E402
from pyidi.GUIs.feature_selection import STEP_FIND, STEP_MASK, SelectionGUI  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gui_frames")

#: the same (row, col) region as the animated scene (sel_prep.py)
POLYGON = [(134, 128), (93, 552), (427, 552), (455, 240), (300, 168)]
#: separation values stepped through on the Evaluate + select tab, 1 px at a time
SWEEP = [11, 12, 13, 14, 16, 18, 20, 22, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12]
WIN_W, WIN_H = int(os.environ.get("WIN_W", 1150)), int(os.environ.get("WIN_H", 720))


def grab(window):
    """Process pending Qt events and grab the window as an (H, W, 4) uint8 array."""
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    ptr = image.bits()
    ptr.setsize(image.height() * image.width() * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((image.height(), image.width(), 4)).copy()


def image_to_window(window, row, col):
    """Window pixel at which an image (row, col) is drawn."""
    scene = window.view.mapViewToScene(QtCore.QPointF(col, row))
    local = window.pg_widget.mapFromScene(scene)
    point = window.pg_widget.mapTo(window, local)
    return [point.x(), point.y()]


def widget_centre(window, widget):
    point = widget.mapTo(window, widget.rect().center())
    return [point.x(), point.y()]


def main():
    sys.ps1 = ">>> "                                     # take the interactive branch of __init__
    QtWidgets.QApplication.exec = lambda self=None: 0    # and do not block in it

    video = pyidi.datasets.load_music_box()
    window = SelectionGUI(video, subset_size=21)
    window.setFixedSize(WIN_W, WIN_H)
    # at small separations the subset rectangles merge into one block; the points read better alone
    window.show_subsets.setChecked(False)
    QtWidgets.QApplication.processEvents()

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    records = []

    def shot(cursor, click=False, label=""):
        frame = grab(window)
        name = f"{len(records):04d}.png"
        iio.imwrite(os.path.join(OUT, name), frame[..., :3])
        records.append(dict(file=name, cursor=cursor, click=click, label=label,
                            points=int(len(window.get_points())), shape=list(frame.shape[:2])))
        print(f"{name}: {label:<28} {records[-1]['points']:5d} points, cursor {np.round(cursor).tolist()}")

    # 1. as opened: the whole frame is already scored and selected
    window.select_step(STEP_FIND)
    window.refresh()
    shot(image_to_window(window, 280, 380), label="open")

    # 2. the Mask tab
    target = widget_centre(window, window.step_buttons[STEP_MASK])
    # no row selected yet, as after opening: otherwise the Mask tab rings every point of "Whole image"
    window.active_index = None
    window.select_step(STEP_MASK)
    window.select_tool("polygon")
    window.refresh()
    shot(target, click=True, label="mask tab")

    # 3. the polygon, corner by corner, exactly as on_mouse_click does it
    for vertex in POLYGON:
        window.add_vertex(vertex)
        window._retire_whole_image()
        window.refresh()
        shot(image_to_window(window, *vertex), click=True, label="polygon vertex")

    # 4. back to Evaluate + select
    target = widget_centre(window, window.step_buttons[STEP_FIND])
    window.select_step(STEP_FIND)
    window.refresh()
    shot(target, click=True, label="evaluate tab")

    # 5. the separation, stepped as a spin box is
    spin = widget_centre(window, window.separation_spin)
    corner = window.separation_spin.mapTo(window, QtCore.QPoint(0, 0))
    spin_rect = [corner.x(), corner.y(), window.separation_spin.width(), window.separation_spin.height()]
    for separation in SWEEP[1:]:
        window.separation_spin.setValue(separation)
        window.refresh()
        shot(spin, label=f"separation {separation}")

    with open(os.path.join(OUT, "timeline.json"), "w") as f:
        json.dump(dict(window=[WIN_W, WIN_H], spin_rect=spin_rect, records=records), f, indent=1)
    print("saved", len(records), "grabs")


if __name__ == "__main__":
    main()
