"""Run the selection pipeline on the music-box reference frame and save what the animation needs."""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyidi.selection import Entry, select_points  # noqa: E402
from pyidi.selection.evaluate import evaluate  # noqa: E402
from pyidi.selection.masks import combined_mask  # noqa: E402
from pyidi.selection.select import select_peaks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ref = np.load(os.path.join(HERE, "..", "music_box", "teeth.npz"))["reference"].astype(np.float64)
SUBSET = 21

# (row, col) polygon around the teeth: along the top edge of the comb, down the free ends,
# back along the bottom edge and up the staircase of tooth roots
POLYGON = [(134, 128), (93, 552), (427, 552), (455, 240), (300, 168)]

evaluate(ref, "shi_tomasi", subset_size=SUBSET)          # warm-up
t0 = time.perf_counter()
res = evaluate(ref, "shi_tomasi", subset_size=SUBSET)
t_eval = (time.perf_counter() - t0) * 1000
score = res[0] if isinstance(res, tuple) else res
print(f"score: {score.shape}, {t_eval:.1f} ms for {ref.size / 1e6:.2f} MP, NaN border {np.isnan(score).sum()} px")

mask = combined_mask([Entry("polygon", POLYGON)], ref.shape)
print("mask pixels:", int(mask.sum()))

SEPARATIONS = [30, 22, 16, 12, 9, 7, 6, 8, 11, 14]
selections = {}
for s in sorted(set(SEPARATIONS)):
    t0 = time.perf_counter()
    p = np.array(select_peaks(score, mask=mask, separation=s))
    dt = (time.perf_counter() - t0) * 1000
    selections[s] = p
    print(f"separation {s:2d} px -> {len(p):5d} points in {dt:5.1f} ms")

whole = np.array(select_peaks(score, separation=11))
print("whole image at separation 11:", len(whole))

check = select_points(ref, [Entry("polygon", POLYGON)], subset_size=SUBSET, separation=12)
print("select_points at separation 12:", len(check), "| matches select_peaks:",
      len(check) == len(selections[12]))

np.savez_compressed(os.path.join(HERE, "selection.npz"), reference=ref, score=score, mask=mask,
                    polygon=np.array(POLYGON), whole=whole, t_eval_ms=t_eval, subset=SUBSET,
                    separations=np.array(SEPARATIONS),
                    **{f"sel_{s}": p for s, p in selections.items()})

fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0b0f14")
lo, hi = np.percentile(ref, [1, 99.7])
for a in ax:
    a.imshow(np.clip((ref - lo) / (hi - lo), 0, 1), cmap="gray")
    a.axis("off")
sc = np.log10(np.nan_to_num(score, nan=np.nanmin(score)) + 1e-9)
ax[0].imshow(sc, cmap="magma", alpha=0.75, vmin=np.percentile(sc, 50), vmax=np.percentile(sc, 99.9))
ax[0].set_title("shi_tomasi score (log)", color="w")
poly = np.array(POLYGON + [POLYGON[0]])
ax[1].plot(poly[:, 1], poly[:, 0], color="#4cc9f0", lw=1.5)
p = selections[12]
ax[1].plot(p[:, 1], p[:, 0], "o", color="#f72585", ms=2.5)
ax[1].set_title(f"separation 12 px: {len(p)} points", color="w")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "selection_preview.png"), dpi=80, facecolor="#0b0f14")
print("saved")
