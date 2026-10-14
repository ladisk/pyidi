# Changelog

This changelog starts at version 1.4.0. For earlier versions see the
[commit history](https://github.com/ladisk/pyidi/commits/master).

## Unreleased

### New method: SimplifiedOpticalFlowPyramid

`SimplifiedOpticalFlowPyramid` runs the simplified optical flow on a
coarse-to-fine image pyramid. Each point is handled on its own: the method
averages the reference intensity and both image gradients over a subset around
the point, reads the displacement along the gradient direction from the
intensity difference, and projects it onto the image axes through the gradient
components.

Plain simplified optical flow only resolves displacements up to about half a
pixel. On the pyramid, the estimate from a coarser level is scaled up and
applied as a shift of the subset on the next level, rounded to whole pixels
by default, so only the
sub-pixel residual is measured at full resolution. On a synthetic stripe
pattern translated by 6.7 px, a single level is off by more than 28 px, while
four levels stay within 0.2 px.

`configure(interpolation='linear')` and `configure(interpolation='cubic')`
keep the shift from a coarser level fractional and read the subset with a
spline of order 1 or 3, instead of rounding the shift to whole pixels. Both
remove the step the rounding leaves in the result. The linear spline smooths
the image enough to bias the result below about one pixel of displacement,
where rounding shifts by nothing at all; the cubic spline does not. Measured
on a speckle pattern in circular motion, as the error along the gradient in
px:

| largest displacement | `'nearest'` | `'linear'` | `'cubic'` |
| --- | --- | --- | --- |
| 0.2 px | 0.005 | 0.029 | 0.002 |
| 0.5 px | 0.036 | 0.055 | 0.005 |
| 1 px | 0.048 | 0.060 | 0.017 |
| 2 px | 0.095 | 0.086 | 0.077 |

The subset can be rectangular: `configure(subset_size=(size_y, size_x))` takes
any pair of odd sizes, and a single int still means a square subset. Choose the
levels either with `n_levels`, which uses levels 0 to `n_levels - 1`, or with
`levels`, which also lets you skip levels, as in `levels=[2, 0]`.

The subset means are read from an integral image (summed-area table), so each
point costs four reads whatever its subset size. A subset that reaches over the
image edge is averaged over the pixels inside the image only.

On a coarse level a point keeps its fractional position, `point / scale**level`,
and its subset is read with interpolation instead of being moved to the nearest
pixel of that level. Rounding put the coarse subset up to half a pixel of the
level off the point, which on a line edge moved by 4 px left an error of
0.52 px; without rounding the error is 0.03 px.

## 1.4.0

### Lucas-Kanade performance

The inner optimization loop is now compiled with `numba` and parallelized over
points. Measured against 1.3.3 on the same machine, with identical results:

| case | 1.3.3 | 1.4.0 | speedup |
| --- | --- | --- | --- |
| `data_synthetic.cih`, 200 points, 101 frames | 7.94 s | 0.10 s | 77x |
| synthetic 512x512, 400 points, 150 frames | 21.04 s | 0.24 s | 89x |
| `data_synthetic.mp4`, 60 points, 10 frames | 2.16 s | 0.06 s | 36x |

The compiled path is on by default. `configure(use_compiled_kernel=False)`
selects the previous NumPy implementation, which is itself faster than 1.3.3
(about 1.6x, and more on video and image files) because the frame is no longer
re-read for every point. Results from the two paths agree to floating-point
round-off. If numba cannot be imported, pyidi falls back to the NumPy
implementation automatically and warns, rather than failing to import.

The compiled kernel supports cubic interpolation only; `int_order` other than `3`
falls back to the NumPy implementation and warns once.

### Directional Lucas-Kanade performance

`DirectionalLucasKanade` now uses the same compiled kernel machinery. The spline
evaluation is shared with Lucas-Kanade; only the least-squares solve differs,
because the motion is constrained to one prescribed direction and so has a
single unknown instead of two. Measured against 1.3.3 on the same machine, with
identical results:

| case | 1.3.3 | 1.4.0 | speedup |
| --- | --- | --- | --- |
| `data_synthetic.cih`, 120 points, 101 frames | 1.46 s | 0.04 s | 38x |
| synthetic 512x512, 400 points, 150 frames | 7.46 s | 0.11 s | 65x |
| `data_synthetic.mp4`, 60 points, 10 frames | 1.25 s | 0.03 s | 41x |

As for Lucas-Kanade, the frame is now read once per time step instead of once
per point. That alone makes the NumPy path about 18x faster on video and image
files; on memory-mapped formats it makes no measurable difference.

### `SelectionGUI` is now the automatic feature selection interface

There is one point-selection window again. `SelectionGUI` names the interface
built on `pyidi.selection`, described in the next section; the window it named
in 1.3 is `SelectionGUIOld`, deprecated and removed in 1.5.

Two windows offering the same five tools was never the plan — the new one was
built alongside the old one so the old one would keep working while it was
written. It turned out to be a superset rather than a companion: every
selection method has a counterpart, and both filters are evaluators.

| `SelectionGUIOld` | `SelectionGUI` |
| --- | --- |
| Grid | Polygon with role `points`, or the `lattice` selector |
| Manual | Points tool |
| Along the line | Line tool |
| Brush | Brush tool |
| Remove point | Remove point tool |
| Shi-Tomasi filter | `shi_tomasi` evaluator |
| Gradient in direction | `gradient_direction` evaluator |

**Most scripts need no edit.** The constructor takes the same arguments and
`get_points()` returns the same `(n_points, 2)` array in `(row, col)` order, so
`SelectionGUI(video, subset_size=21)` followed by `set_points(gui)` keeps
working and opens the better window.

What does not carry over, for anything that reached deeper:

- **`get_filtered_points()` and `get_selected_points()` are gone.** There is one
  `get_points()`. Filtering is no longer a second pass over a selection that
  already exists — it is the selection.
- **The internal attributes have no counterpart** — `selections`,
  `subset_size_spinbox`, `candidate_points` and the rest.
- **`Grid` is not a mode.** Draw a polygon and set its row to the `points` role,
  or keep it a mask and choose the `lattice` selector.
- **Scores near the image border differ slightly**, because gradients are taken
  over real neighbours instead of ones reflected at the subset edge. The new
  value is the correct one.

`SelectionGUIOld` prints a `DeprecationWarning` on construction and is otherwise
unchanged. `FeatureSelectionGUI`, the working name used while this was being
built, never appeared in a release; the name raises a `RuntimeError` saying
where it went.

### Automatic feature selection

The `pyidi.selection` package, and the interface over it, add automatic feature
selection: score the whole image, then pick the best-separated features inside
the region you drew, rather than placing subsets on a grid and discarding the
poor ones. On a random speckle pattern or an intricate structure this is the
difference between sampling where the features are and sampling where the grid
happens to fall. Implements the workflow discussed in
[issue #51](https://github.com/ladisk/pyidi/issues/51), using the
mask/evaluate/select vocabulary agreed there.

The blocker was never the idea, it was the cost. `SelectionGUIOld` scores one
subset at a time — a Sobel and a 2x2 eigendecomposition per point — which is
fine for a few hundred grid points and takes minutes at one megapixel, so the
spacing control was really a compute budget. Shi-Tomasi is now computed as a
whole-image Sobel plus three box sums and a closed-form minimum eigenvalue:
the same quantity, a few separable passes, tens of milliseconds per megapixel.
Dense candidates therefore cost nothing and a region goes back to meaning
"where I want points".

The three steps are separated, and only the middle one is expensive:

- **mask** — regions define an *area*. Each row in the selections list carries
  a role: `mask` rows contribute their area, `points` rows contribute
  coordinates that bypass scoring entirely, and the role is switchable per row.
  Hand-picked points always survive whatever their score, and no automatic
  point is placed within the separation of one.
- **evaluate** — Shi-Tomasi and gradient-in-direction, computed over the whole
  image, with `NaN` marking the border where the subset window would leave it.
  Scores are named and cached on (evaluator, parameters, subset size), so
  several coexist and switching between them is free the second time.
  Evaluators are a registry: a new one is a function plus parameter
  descriptors, and it appears in the menu with no GUI change.
- **select** — a threshold plus a **separation**: the distance no two points
  may come closer than, which is the one control for how many you get. A bare
  threshold on a dense score image returns a solid blob of adjacent pixels on
  every corner, so the separation is what makes the result a set of features.
  A `lattice` selector reproduces regular grid sampling inside the same
  pipeline, for full-field work where even coverage matters more than feature
  strength.

The threshold is a **quality**: a fraction of the best feature in the region,
so 0.01 means "at least a hundredth as good as the best one here", on a
logarithmic slider because the useful settings span three decades. A percentile
ranks *pixels*, and on a dense score image the pixels are overwhelmingly
background — on a typical frame its 90th percentile is under a five-hundredth
of the best feature, so nine tenths of a percentile slider's travel sits inside
the featureless area and lowering it floods the frame with background rather
than admitting weaker features. The reference is the 99.9th percentile of the
scores rather than their maximum, so one specular highlight cannot drag every
useful setting into the floor of the slider. `percentile of scores` remains
available, and is the right rule for the `lattice` selector, whose candidates
are already spaced out. A third rule, a fraction of the literal maximum, was
offered and then dropped: it is quality with a reference one bright pixel can
move, so on any usable frame it is indistinguishable and on a bad one worse.

**Decimation** thins the points that were already selected — every n-th,
survivors left exactly where they are — for when the selection is right and
only the count is too high for the computation about to run. Widening the
separation re-selects and moves every point, which is a different thing, so the
two are separate controls. Hand-placed points are never decimated, and what a
region selected is recorded as occupied before it is thinned, so decimating one
region leaves gaps rather than inviting another to fill them.

Decimation is deliberately not the density control, because thinning the pixels
above the threshold that way does not work. On a 1024×1024 frame with 357 000
of them, thinned to twenty thousand points, keeping every n-th in score order
leaves 78 % of the subsets within three pixels of another one and keeping every
n-th in scan order 92 %; the separation leaves none. Score order fails because
consecutive ranks are neighbours on the same feature, scan order because the
stride aliases against the row length.

The **point cap** now says when it is what stopped the selection. It had no
other symptom — it simply stopped adding points, and since it keeps the
highest-scoring ones, a selection that hit it looked like a tight cluster on
the strongest features and read as though the threshold or the spacing had
caused it.

Selection is fast enough to drive from a slider. The exact greedy walk is
linear in the candidates, and a loose threshold leaves hundreds of thousands
of them — 40 ms to 300 ms depending on the separation, which nothing can drag.
So the candidates are reduced first, to the best pixel in each cell of a grid
half the separation across. That costs yield and not the guarantee: at a
separation of 11 it finds 1708 points where the exact walk finds 2193, in 9 ms
instead of 39, and the walk still runs so the separation still holds exactly.

The rest of the redraw was Python loops over the points rather than the
selection itself. Vectorised — deduplication through one `unique` over folded
coordinates, per-entry crediting through one mask lookup each, occupancy
through one indexed assignment — and with each entry's rasterisation cached
against a fingerprint of its geometry, the pipeline half of a redraw at 20 000
points on a megapixel frame went from 46 ms to 21 ms. Earlier in the same work
a redraw stopped running the selection three times over.

What is left is drawing, so redraws are **coalesced**: while one costs less
than a frame it still happens immediately, and above that the requests collapse
into a single deferred redraw carrying the latest values. A fast drag therefore
repaints as often as it can rather than queueing every position on the way, and
lands on the value the control stopped at.

Three things then made every redraw more expensive than it had to be. A masked
selection ran over the whole frame, though nothing outside the mask was ever
eligible: it now runs inside the mask's bounding box, snapped back to a whole
reduction cell so the block grid falls where it would have and the answer is
identical. Every selected point was stamped into a full-frame occupancy array
so that a later group could not fill its gaps -- a Python loop over all of them,
84 ms at seventeen thousand points, and there is usually no later group, so it
is now skipped unless one is coming. And the seeded whole-image row was
re-rasterising each region from scratch to decide whether to stand down,
bypassing the cache that already had the answer.

What is left of a redraw is drawing, and the two big point layers -- the
selected points and the dim blue candidates -- are now one stroked path each
rather than a `ScatterPlotItem`. The item keeps a record per spot and rebuilds
a symbol atlas, 17 ms for seventeen thousand points on every redraw; a path of
very short segments stroked with a round-cap pen draws the same dots from one
vectorised call, in 2 ms. Layers with a per-point colour, a symbol or a hover
behaviour still use the scatter item.

Together, on a 2560x1600 frame at separation 6: placing a polygon corner over a
drawn region went from 115 ms to 30 ms, and a redraw with the whole frame
selected from 342 ms to 158 ms.

A brush stroke updates nothing but itself. It used to cost a mouse move what a
full redraw costs -- a whole-frame RGBA overlay rebuilt and re-uploaded per dab,
and the entire point cloud handed back to the scatter item to take the covered
points out of it -- which is 24 ms a move at 17 000 points on a four-megapixel
frame, paid while the mouse is moving. The stroke is now a path of overlapping
discs rather than a raster, and the crossing-out is drawn *over* the red points
instead of replacing them, so a move costs the points it reached and nothing
else: 0.1 ms. The selection itself is re-run once, when the stroke lands.

Because a mask or threshold edit only re-derives from a cached score, the
interface updates while a slider is still moving; only a subset-size or
evaluator change recomputes.

The window presents this as **two** tabs, not three, and deliberately does not
number them. Evaluation does not depend on the mask, so mask and evaluate are
siblings feeding select rather than a sequence. The tabs are named for the
steps they hold: `Evaluate + select` holds the evaluator and the selection
controls together, since changing one changes what the other means; `Mask` is
where the candidates get trimmed. The selections list is on the `Mask` tab and
only there — every row in it, and every button under it, acts on something
drawn there — while the subset size stays on both, because both steps read it. The selections
list starts with a `Whole image` row, so points are on screen the moment the
window opens and masking is editing rather than a precondition. That row is
ordinary — uncheck it, paint it away, or delete it, and deleting it selects
nothing rather than reverting to the whole frame. Since mask rows combine as a
union, drawing your own region unchecks it, so the region restricts the
selection instead of being absorbed into a union that changes nothing.
`Clear all` starts over rather than clearing to nothing: it seeds that row
again, so the whole frame is selected, as when the window opened.

While masking, the points are shown in three tiers, because "no point here"
otherwise means two different things: red for a point being taken, dim blue for
a feature the mask is leaving out, and a magenta ring for the points the
selected row accounts for. The dim tier is the whole-frame selection, so it
does not move while a mask is edited — painting a region turns points from blue
to red where it lands rather than re-selecting underneath you — and it is
cached across mask edits, which is what keeps it affordable: about a fifth
added to a redraw on that tab.

The subset size takes odd values only, by stepping and by typing: a subset is
centred on its point, so an even extent has no centre to be, and the pipeline
reads one as the odd size above it in any case — a subset size of 10 scores
through an 11-pixel window and draws an 11-pixel rectangle — so the even values
were a second spelling of the odd ones. It sits below the tabs rather than on
one of them: the scoring window follows it, which is what makes it one of the few settings that stales
the cached score, and it is also the size of the rectangle drawn round each
point while masking. `Show score overlay` is on both tabs and the two controls
stay in step. Control groups are flat rather than nested, settings the current
selector does not read are hidden rather than greyed out, and the descriptive
paragraphs are tooltips -- between them they were most of a panel that was
narrow enough to clip its own values. A direction
can be dragged out on the image, as in `SelectionGUIOld`, as well as typed or
taken from the `X`/`Y` presets, and a line shows the direction in force.
Points under a deselect stroke are crossed out while the stroke is being
painted, rather than only disappearing once the mouse comes up.

The pipeline imports without Qt and is usable from a script through
`pyidi.selection.select_points` (one call) or `SelectionPipeline` (keeps the
score cache alive across parameter changes).

`Remove point` takes the point's whole reserved disc — its separation — out of
the mask, not the single pixel under the click. A selected point is re-derived
from the score on every run, so erasing one pixel just promotes its neighbour
and the point reappears a couple of pixels along. Removing one is undoable, as
every other edit is. Hand-placed points are still deleted outright, so clicking
the same pixel again puts one back.

A click outside the image adds nothing. The view is always larger than the
frame — the aspect is locked, so one axis has a margin, and zooming out adds
more — and a subset centred off the frame cannot be tracked; any coordinate that
reaches the pipeline from elsewhere is dropped there too.

Score images are kept in a bounded cache, eight deep. Each one is a full-frame
`float32`, 16 MB at 2560x1600, and every distinct set of evaluator parameters is
a different array, so a direction spin box dragged through sixty values would
otherwise hold sixty of them. The score overlay is redrawn as part of the
refresh, so it follows a change of subset size or of evaluator instead of going
stale.

A deselect stroke touches only the regions it actually covers, rather than
giving every region on the list a frame-sized erasure array, and an undo
snapshot holds those arrays by reference rather than copying them into each of
the fifty slots.

Ctrl is read off the mouse event rather than tracked by a key filter on the
window, which a panel widget with focus can swallow; letting go of Ctrl
part-way through a stroke finishes the stroke rather than abandoning it.

Changing an evaluator parameter goes through the same coalescing as every other
control. It is the one control that can make a redraw expensive, since a
parameter not scored before is a whole-frame evaluation.

**Scores differ slightly from the old per-subset filter near strong edges.**
`SelectionGUIOld` ran Sobel on the isolated subset, so gradients at the subset
border used values reflected from inside it; the whole-image version sees the
real neighbours. The new value is the correct one. This affects the new module
only — `SelectionGUIOld` is untouched and behaves exactly as before.

### The active tab and tool are visible

Both groups of buttons -- the two tabs across the top, and the region tools --
were left to the platform theme to mark, and a default theme separates a
checked `QPushButton` from an unchecked one by a shade or two. That is not a
difference you can find across a panel, and this interface asks the question
twice: which tab am I on, and which tool is active. A checked button is now
filled and bold, in the blue the selections list already highlights with.

Only the checked state is styled. Everything else stays whatever the platform
theme makes it, so the window does not have to carry a theme of its own to have
a legible one -- which is what the older window did, and why it looks dated on
a modern desktop.

### Erasing is a tool, not a mode the brush is in

`Remove w/ brush` joins the mask tools, next to `Remove point`. It replaces the
`Deselect painted area` toggle that used to sit inside the brush controls,
which made painting a mode within a mode: the same brush added or subtracted
depending on a checkable button several rows below it, and which one it was
about to do was not visible where the work happened.

The two tools that take away are now side by side, at the two scales they
work at -- one point, or everything a stroke covers. `Remove w/ brush` is the
brush in reverse and shares its radius, so a stroke erases exactly as wide as
it paints. What it does is unchanged: it subtracts only the part actually
painted over, a region keeps whatever the stroke missed, and it disappears
only once nothing of it is left.

Two controls on that tab also stop being shown when nothing can act on them.
`Brush radius` appears for the tools that paint, and `Point spacing` for a row
that lays points out along or inside its shape -- a polygon, line or brush
stroke in the `points` role. Neither is read otherwise: a `mask` row has its
points chosen by the selection, so their distance is the separation on the
other tab, and a `points`-tool row is the coordinates that were clicked. This
follows what the selector's own settings already did, appearing and
disappearing with the selector rather than greying out.

### Point selection consolidated on one window

> The window these entries describe is the one now called `SelectionGUIOld`.
> See "`SelectionGUI` is now the automatic feature selection interface" above.

pyidi had accumulated five separate point-selection implementations. Three were
dead code, one was documented but no longer developed, and the one under active
development was not reachable from the documented workflow. There is now one.

- **`SubsetSelection` has been removed.** The tkinter widget in
  `pyidi/GUIs/selection.py` is gone. The name is still importable, but
  instantiating it raises a `RuntimeError` naming the replacement, so existing
  scripts fail with an actionable message rather than an `ImportError`. Replace
  `SubsetSelection(video, roi_size=(21, 21), noverlap=0)` with
  `SelectionGUI(video, subset_size=21, subset_overlap=0)` — the current
  `SelectionGUI`, which takes the same arguments as the window described here.
- **It took over as the documented interface, and is itself superseded by the
  new `SelectionGUI`.** Five selection methods (grid in a polygon, manual
  points, along a polyline, brush, and remove-point) plus automatic filtering by
  Shi-Tomasi corner strength or gradient direction — the set the table above
  maps onto the new window. It requires the Qt extras:
  `pip install pyidi[qt]`.
- **Note for `LucasKanade` users:** it can select anisotropic
  subsets again. `subset_size` accepts a scalar or a `(height, width)` pair,
  in the same `(vertical, horizontal)` convention as
  `LucasKanade.configure(roi_size=...)`. In the UI, a `Square subsets`
  checkbox (checked by default) keeps the previous square-only behaviour;
  unchecking it frees the height and width spinboxes/sliders to be set
  independently.
- Removed the dead selection code: `tools.ManualROI`, `tools.GridOfROI` (both
  read a `video.reader.mraw` attribute that no longer exists), the unreachable
  `PickPoints` class in `_simplified_optical_flow.py`, and the stray
  `load_analysis copy.py`.

### `SelectionGUIOld` editing

- **The four separate selection stores are now one ordered list.** Grids,
  lines, brush strokes, and manually-clicked points all live in a single
  always-visible `selections` list in the right-hand panel, replacing the
  two mode-specific lists (Grid, Along-the-line) and their two delete
  buttons. Every grid, every drawn line, and every brush stroke gets its own
  row (`Grid 1`, `Line 1`, `Brush 1`, …); every manually-clicked point
  is collected into one shared `Manual` row. Each row shows a live point
  count, e.g. `Grid 1 — 142 pts`. The list is visible in every selection
  mode, so everything built so far stays in view regardless of which tool is
  active.
- **Clicking a row** makes it the active selection, switches the tool to that
  row's type so its vertices are immediately draggable, and highlights its
  points in the image with a magenta ring. The highlight is a Select-mode cue
  and is hidden in Filter mode, where the Select-mode points are not drawn.
  **Each row has a checkbox** that excludes its points from the result without
  deleting the row — useful for trying a region in and out.
- **Any selection can now be deleted, including a brush stroke or the
  `Manual` row.** One `Delete selected` button replaces the previous
  `Delete selected grid` / `Delete selected polygon` pair, and works for
  every kind. Deleting the last remaining selection now simply empties the
  list, rather than re-seeding an empty placeholder entry as it briefly did.
- **Row labels are no longer reused.** Deleting `Grid 2` and then creating
  another grid now gives `Grid 4`, not a second `Grid 3`.
- **Polygon and grid vertices can be dragged.** A left-drag starting within ~10
  screen pixels of an existing vertex moves it; a drag anywhere else still pans,
  and the grab radius is constant in screen pixels at any zoom. Clicking exactly
  on an existing vertex is now a no-op rather than adding a duplicate on top of
  it. The derived subset points are recomputed once when the drag finishes, not
  on every mouse-move.
- **Undo (Ctrl+Z)** now reverses deleting ANY selection — a grid, a line, a
  brush stroke, or the `Manual` row — in addition to adding and moving a
  vertex. It previously covered only grid/polyline deletion; brush strokes
  and the manual row could not be undone at all. A restored selection comes
  back at its original row with its original label. Filter results are still
  not undoable.
- The "Start new line" button now reads "Start new grid" in Grid mode. The
  status-bar hint said "Click 'Start new line' to begin a new grid" and now
  matches the button.
- **Filter candidates now follow the selection.** The Filter-mode filters score
  the subsets placed in Select mode, but their result was never revisited when
  those subsets changed. Deselecting an area with the brush (or removing a
  point, or deleting/unchecking a row) left its candidates on screen and — since
  `get_points()` returns the candidates once a filter has been run — in the
  returned points. Candidates outside the current selection are now dropped, and
  the threshold sliders can no longer bring them back. The per-subset scores are
  kept rather than discarded, so this is reversible: re-checking a row, or
  undoing its deletion, restores its candidates without re-running the filter.
- **Subset rectangles now have hairline borders.** They used to be painted
  entirely into one RGBA image the size of the frame, so a border could not be
  thinner than one *image* pixel — which grows into a thick band as soon as you
  zoom in, and on an 11 px subset already ate a fifth of its width. The
  translucent interior is still drawn that way (one upload however many subsets
  there are), but the borders are now a single vector path stroked with a
  cosmetic pen, whose width is in *screen* pixels, so they stay one pixel thin
  at any zoom. Building the interior no longer loops over the points in Python
  either, which makes the redraw after a subset-size or spacing change several
  times faster on large selections.
- **The order of `gui.points`/`gui.get_points()` is now creation order**
  across all selection types combined, rather than grouped by type (all
  manual points, then all line points, then all grid points, then all brush
  points). No supported use depends on point order.

### Rigid body motion in `DirectionalLucasKanade`

`DirectionalLucasKanade` gained `set_rigid_body_motion(rbm_ij)`: a per-frame
`(n_time_points, 2)` array giving a known, prescribed rigid-body translation.
The tracking window for every point now follows this prescribed motion, and
the motion is subtracted back out of the result, so `self.displacements`
reports the local motion relative to the rigid body motion rather than each
point's absolute pixel position. Only the component of the rigid body motion
aligned with each point's tracking direction (`dij`) is currently supported.
If `set_rigid_body_motion` is never called, it defaults to zero and existing
analyses are unaffected.

The same change also fixes the NumPy-path convergence check: `compute_delta`
(aliased as `compute_delta_numba`) returned a signed error, but the
optimizer's stopping test (`error < tol`) assumes a non-negative error, so
iterations could stop early on a spuriously negative error or fail to
converge. The error is now returned squared, so it is never negative.

### Eulerian video magnification

New `EulerianMagnifier` class in `pyidi.postprocessing` (also available as a
functional `eulerian_magnification()` wrapper) adds linear Eulerian Video
Magnification (Wu et al., SIGGRAPH 2012) as a pre-test visualization tool: a
Laplacian pyramid decomposition, a temporal band-pass filter applied per
pyramid level, and linear amplification of the band-passed signal added back
onto the original. It reveals subtle, often sub-pixel motion directly in the
raw recording, before any displacement identification is run, so it is useful
for checking whether (and where) a structure is moving and for picking
regions of interest or seed points ahead of a full analysis. **This is
qualitative visualization only, not a measurement** - the amplification
distorts motion amplitudes non-linearly and must not be read as displacement.

Configure with `freq_band=(low, high)` in Hz to isolate a suspected mode,
`amplification` for the gain, and `levels` for the pyramid depth. The
temporal filter is `filter_type="ideal"` (FFT brick-wall, default) or
`"butter"` (Butterworth). An optional 2D `mask` restricts amplification to a
region of interest, leaving the rest of the frame as recorded. `save()`
writes the result to mp4/avi/mov/gif, mapping the intensity range to 8-bit
for playback.

The optional `lambda_c` spatial-wavelength attenuation, meant to damp
amplification of fine, noisy detail while leaving broad structural motion at
full gain, initially ramped in the wrong direction: the finest pyramid level
got the strongest amplification and the coarsest the weakest, the reverse of
what Wu et al. specify. This is now fixed to ramp from the coarsest band down
to the finest. A warning is also now raised if `lambda_c` ends up attenuating
every level to zero (the output would then equal the input unchanged), and
`save()` raises rather than silently defaulting to 30 fps when no frame rate
is available. The test suite was substantially hardened alongside these
fixes, including mutation-verified tests that would have caught a disabled
band-pass, a sign-inverted amplification, or a dropped pyramid band.

### Example datasets

`pyidi.datasets` downloads example recordings from Zenodo on first use and
caches them in `~/.pyidi/datasets` (or in `PYIDI_DATA_DIR`):

```python
video = pyidi.datasets.load_music_box()
```

The first dataset is a high-speed video of a vibrating music-box comb
([10.5281/zenodo.22105821](https://doi.org/10.5281/zenodo.22105821), CC BY 4.0).
Only the requested frames are downloaded, using HTTP range requests, so the
default window costs 404 MiB instead of the 2 GiB of the published excerpt (or
the 36 GiB of the full recording). An interrupted download is resumed. The new
`examples/Showcase_music_box.ipynb` walks from the raw video to the notes of
the comb and to the operating deflection shape of a single tooth.

A dataset is a dictionary of metadata in the `pyidi.datasets.DATASETS` registry,
so the next one needs no new code: `list_datasets()` says what is available,
`load_dataset(name)` loads any of it, and `register_dataset()` accepts a
recording that is not part of pyidi. The named shortcuts, `load_music_box()` and
`fetch_music_box()`, remain. What the registry assumes is what every dataset
published this way has in common — a Zenodo record holding a Photron `cihx`
header next to an uncompressed `mraw` file of fixed-size frames, so that a
window can be addressed by byte offset.

### Point validation

`set_points()` now validates its input instead of accepting almost anything.

- Empty input, non-2-D input, a wrong column count, and coordinates outside the
  image now raise `ValueError` with a message that says what was wrong. Empty and
  1-D input previously raised `IndexError: tuple index out of range`; out-of-range
  and negative coordinates were previously accepted silently, and now that an
  untrackable point returns `NaN` rather than raising, they would otherwise
  surface only as a `NaN` result much later.
- **Sub-pixel points are now rounded to the nearest pixel, with a warning.**
  Previously the same float input crashed in `SimplifiedOpticalFlow` (used
  directly as an array index) but was silently truncated *toward zero* in
  `LucasKanade`, `DirectionalLucasKanade`, and `DIC`. All four now agree, and
  round rather than truncate.
- `set_points()` accepts any object exposing a `.points` attribute, so a
  selection GUI instance can be passed directly. Previously only `SubsetSelection`
  was recognised, and passing the Qt GUI failed with an opaque error.
- The napari `GUI` now routes its selections through `set_points()` as well, so
  points picked in the UI get the same checks as programmatic ones.

### Documentation overhaul

The documentation was restructured around the work done since 1.3.3. New
pages: Eulerian video magnification, reading a video (all supported formats,
including `.cine`, and the frame-rate caveats), results and reproducibility
(where analyses are saved, `load_analysis`, resuming, and what a `NaN` in the
result means), and an upgrading guide covering `SubsetSelection` ->
`SelectionGUI`, `use_numba` -> `use_compiled_kernel`, the stricter
`set_points()` contract, and the pre-1.0 `pyIDI` class.

The API reference gained the pages it was missing: the graphical interfaces
(`SelectionGUI`, `SelectionGUIOld`, the napari `GUI`, `ResultViewer` and
`Viewer`), documented from their docstrings with the Qt and napari imports
mocked so the build needs neither, and `pyidi.tools`, which the contributing
guide had been using as its worked example of a documented module.

The methods page gained a "choosing a method" comparison, a parameter table
per method, and a section on prescribed rigid-body motion in
`DirectionalLucasKanade`. The mode-shape magnification and fiducial-marker
pages, previously stubs reading "more documentation is coming soon", now
document the actual API. `CHANGELOG.md` is rendered into the documentation.

Sphinx gained `sphinx-design` (landing-page cards), `myst-parser` (Markdown),
`napoleon` (the Google-style docstrings in `fiducial.py` render correctly now)
and `intersphinx` (links into the numpy, scipy and Python documentation). The
build is warning-free.

Two source-level fixes fell out of writing this: `ResultViewer` documented its
`displacements` argument as `(n_frames, n_points, 2)` when it indexes it as
`(n_points, n_frames, 2)` — the shape `get_displacements()` actually returns —
and `VideoReader.get_frame`'s docstring had a mis-indented field that broke
its rendering. The old documentation also claimed `load_analysis()` returns
two values; it returns three (`video, idi, settings`).

### Behaviour changes

- **A point that cannot be tracked no longer aborts the analysis.** Previously a
  degenerate region of interest raised `ValueError` and no results were returned
  at all. Such a point is now `NaN` from the frame at which it was lost, every
  other point is computed normally, and a warning is issued. Which points failed
  is recorded in the new `failed_points` attribute. This applies to
  `use_compiled_kernel=False` as well, so code that relied on the exception to
  halt should check for `NaN` instead. It applies to both `LucasKanade` and
  `DirectionalLucasKanade`.
- The NumPy path now also detects a runaway optimization. Previously such a point
  returned displacements of tens of pixels, silently.
- `DirectionalLucasKanade.configure()` no longer accepts `use_numba`. The
  argument was accepted but documented as not implemented and did nothing; it is
  replaced by `use_compiled_kernel`, which defaults to `True` and does what the
  old name suggested. Calls that passed `use_numba` need updating.
- `DirectionalLucasKanade` previously warned and continued with a zero update
  when the image had no gradient along the search direction, which returned a
  point that silently never moved. Such a point is now reported through
  `failed_points` and set to `NaN`.
- `DirectionalLucasKanade.configure()` accepts a scalar `pad` in addition to the
  `(pad_y, pad_x)` pair it required before.

Detection of untrackable points is best effort: a point can return implausible
values without being flagged, so a result without `NaN` is not proof that every
point tracked correctly.

### Fixed

- **`import pyidi` failed outright when PyQt6 was installed without napari.**
  `pyidi.GUIs` gated every class on PyQt6 alone and then imported the napari
  `GUI` unconditionally, so `pip install pyqt6 pyqtgraph` without the `[qt]`
  extra produced a `ModuleNotFoundError` from inside a submodule and took the
  whole package down with it. Each class now checks its own dependencies:
  `SelectionGUI`, `SelectionGUIOld`, `ResultViewer` and `Viewer` need PyQt6 and
  pyqtgraph, the napari `GUI` needs napari and magicgui, and whichever are
  unavailable become stubs that import cleanly and raise `RuntimeError` on
  construction naming the missing packages. `Viewer` had no such stub at all
  and raised `NameError`.
- **Asymmetric `pad` in `DirectionalLucasKanade` crashed every point.**
  `_interpolate_reference` and `_warm_up_kernels` paired the `(pad_y, pad_x)`
  axes in the opposite order to `_padded_slice`, so a non-square `pad` (e.g.
  `configure(pad=(2, 5))`) built the reference spline over a grid of the
  wrong shape. All three now use the same axis pairing.
- **A point already lost before a checkpoint could come back with garbage
  displacements after resuming.** `failed_points` is rebuilt from scratch on
  resume and is not itself checkpointed, so a resumed analysis had no record
  that a point was already `NaN`. `np.round(NaN).astype(int)` is undefined
  (e.g. `INT64_MIN` on x86, `0` on arm64) rather than raising, so such a point
  could silently restart tracking from a finite but meaningless position.
  `LucasKanade` and `DirectionalLucasKanade` now check the previous
  displacement for NaN/inf before rounding it and keep the point marked
  failed if so, matching what the compiled kernel already did.
- **A single untracked point (`NaN`) could break the displacement-vector
  display.** The napari `GUI` and the Qt `result_viewer` scaled vectors by
  `np.max`/`np.max(np.abs(...))`, both of which propagate to `NaN` if any
  point in the result failed to track. They now use `np.nanmax`, with a
  fallback when every point failed.
- With `processes` greater than one, warnings about failed points raised
  inside a worker used worker-local point indices and, under the
  `forkserver`/`spawn` start methods, might not reach the console at all. The
  parent process now re-summarises failed points with global indices once
  the worker results are merged, for both `LucasKanade` and
  `DirectionalLucasKanade`.
- **Removed points in `SelectionGUIOld` no longer reappear after a recompute.**
  The `Remove point` tool used to delete from a selection's *derived* points,
  which were regenerated from the source geometry whenever the subset size or
  spacing changed, so a removed point could silently come back. Removals are
  now recorded per selection and re-applied after every recompute.
- **`SelectionGUIOld`'s brush had its row/column spacing swapped for anisotropic
  subsets.** For a non-square `subset_size=(height, width)`, the brush laid
  its grid out with the axes transposed — columns stepped by the height and
  rows stepped by the width. Square subsets were unaffected. Now fixed.
- **`Deselect painted area` no longer throws away a whole brush stroke.**
  Deselecting over any part of a painted region discarded the entire stroke,
  so nibbling a corner off a large brush selection wiped all of it. The
  deselect stroke is now subtracted from the painted mask, so only the
  overlapping area is lost and the rest of the stroke stays; the selection is
  removed only once nothing is left painted. Because the mask itself is
  edited rather than its derived points, the deselection also survives a
  subset-size or spacing change.
- **`configure(show_pbar=False)` was ignored by `LucasKanade` and `DIC`** when
  running in a single process; the progress bar was always shown.
  `DirectionalLucasKanade` already honoured the setting.
- **`Fiducial.detect_markers()` always failed on its own default.** The
  constructor stores the recording as an array, but detection required a
  `list` and raised `ValueError` for anything else, so the only way in was the
  list returned by `pre_process()` — which is documented as optional. Both an
  array and a list are now accepted. A recording deeper than 8 bits, which the
  OpenCV detectors cannot read, now says so instead of failing on an assertion
  inside OpenCV.
- **`Fiducial.revert_frames()` returned a frame it could not revert as black
  rather than as `NaN`.** The result was allocated with the dtype of the video,
  so the `NaN` marking a frame with no transformation — and the `NaN` border
  requested for the homography path — was cast to an ordinary pixel value on any
  integer recording. The result is now float, and a skipped frame is `NaN`
  throughout.
- **Mouse drags were offset from the cursor by 9 pixels.** The drag handlers read
  `ev.pos()`/`ev.buttonDownPos()`, which are local to the ViewBox, and passed them
  to `mapSceneToView()` and `sceneBoundingRect().contains()`, which expect scene
  coordinates. The click handlers already used `scenePos()` and were correct, so
  clicking and dragging disagreed. Most visibly this meant the **brush painted
  about 9 px away from the cursor**, and its bounds check was wrong by the same
  amount. All drag paths now use `scenePos()`/`buttonDownScenePos()`.

### Other

- `numba` now requires at least 0.59.
- Windows and macOS added to the CI test matrix, and Python 3.10, which
  `requires-python` has claimed all along without testing it.
- `setup.py` is removed. The build has used hatchling through `pyproject.toml`
  for several releases, so the file was never executed; it also read a
  `requirements.txt` that no longer exists and listed only two of the four
  subpackages, so anything that did execute it would have built a broken
  package.
- The PyPI classifiers list every supported Python, rather than 3.10 alone.
- When `processes` is greater than one, each worker is limited to a single numba
  thread, so process-level and thread-level parallelism cannot oversubscribe the
  CPU. This applies to both Lucas-Kanade methods.
- pyidi now asks numba for a fork-safe threading layer as the first thing it
  does, before importing anything that imports numba. This matters because
  pyMRAW starts numba's thread pool at import time, and the layer cannot be
  changed once the pool is up. Left alone it resolves to whatever is installed:
  TBB on machines that have it, but OpenMP on machines that do not, and GNU
  OpenMP kills any child forked from a process that has used it. Set
  `NUMBA_THREADING_LAYER` to override.
- On Linux, the worker pool no longer forks when a GNU OpenMP runtime is loaded
  into the process, since libgomp terminates any child forked from a process
  that has used it. numba's own threading layer is fork-safe, but OpenCV and
  some BLAS and SciPy builds pull in libgomp too, and that was enough to kill
  every worker. Such sessions now start their workers through `forkserver`.
  Where forking is safe it is still used, because it shares the video with the
  workers rather than pickling a copy to each of them.
- The kernels are compiled once in the parent process before the worker pool is
  created, instead of once per worker.
- `LucasKanade`'s `compute_inverse_numba` and `compute_delta_numba` are renamed
  to `compute_inverse` and `compute_delta` by the numba rewrite. Both old names
  stay importable as aliases, so code written against 1.3.3 keeps working.
- `DirectionalLucasKanade`'s `compute_delta_numba` is now actually compiled; its
  `@numba.njit` decorator had been commented out. The name stays importable as an
  alias of the new `compute_delta`.
- New `pyidi/selection_geometry.py` holds the ROI-grid geometry as pure numpy,
  with no GUI-toolkit dependency, shared by the napari `GUI` and the selection
  windows.
  Its functions do not share one coordinate convention - each docstring states
  which one it uses, and the tests pin the difference deliberately.
- `SelectionGUIOld` accepts a numpy array as documented. A 2-D or 3-D array
  previously raised `AttributeError` because the frame was only set for a
  `VideoReader`; anything unusable now raises `TypeError`.
- First tests for the GUI package: `tests/test_selection_geometry.py` and
  `tests/test_set_points_validation.py` (41 tests).
- Fixed `README.md`, which told users to call `video.set_points(...)`.
  `VideoReader` has no such method - points are set on the method object.
