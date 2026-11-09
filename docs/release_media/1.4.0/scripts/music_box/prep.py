"""Download the music-box window, track a grid along every tooth, save for rendering."""
import os
import time

import numpy as np
import imageio.v3 as iio

import pyidi

OUT = os.path.dirname(os.path.abspath(__file__))

video = pyidi.datasets.load_music_box()   # 600 frames from frame 400
fps = video.fps
print(f"{video.N} frames {video.image_width}x{video.image_height} at {fps:.0f} fps")

reference = np.asarray(video.get_frame(0)).astype(np.float32)
frames = np.asarray(video.get_frames((0, 200))).astype(np.float32)
motion = frames.std(axis=0)
del frames

tooth_rows = np.arange(109, 420, 22)       # 15 teeth, as in the showcase notebook
columns = np.arange(200, 571, 10)
points = np.array([[r, c] for r in tooth_rows for c in columns])

lk = pyidi.LucasKanade(video)
lk.set_points(points)
lk.configure(roi_size=(21, 21), pad=3, int_order=3, reference_image=0, show_pbar=False)
t0 = time.perf_counter()
d = lk.get_displacements(autosave=False)
print(f"LK: {len(points)} points x {video.N} frames in {time.perf_counter() - t0:.2f} s")

# a handful of raw frames for the background (stack of all 600 is ~400 MB as float; keep uint16)
raw = np.asarray(video.get_frames((0, video.N)))
np.savez_compressed(os.path.join(OUT, "musicbox.npz"), reference=reference, motion=motion,
                    tooth_rows=tooth_rows, columns=columns, points=points,
                    displacements=d.reshape(len(tooth_rows), len(columns), video.N, 2),
                    fps=fps)
np.save(os.path.join(OUT, "raw_frames.npy"), raw)


def to8(a, lo=1, hi=99.5):
    a0, a1 = np.percentile(a, [lo, hi])
    return (np.clip((a - a0) / (a1 - a0), 0, 1) * 255).astype(np.uint8)


iio.imwrite(os.path.join(OUT, "reference.png"), to8(reference))
iio.imwrite(os.path.join(OUT, "motion.png"), to8(motion))
print("saved")
