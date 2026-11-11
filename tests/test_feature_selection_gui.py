"""Tests for ``SelectionGUI``, the interface over the selection pipeline.

Constructed headlessly, by the same recipe as
``tests/test_selection_gui_anisotropic.py``:

* ``QT_QPA_PLATFORM=offscreen`` before Qt is imported, so Qt renders to its
  software framebuffer instead of opening a display;
* ``sys.ps1`` set, so the constructor takes its "interactive" branch rather
  than ``sys.exit(...)``;
* ``QApplication.exec`` neutralised, since that branch still calls it and would
  otherwise block in the event loop.

What is worth testing here is not the widgets but the *contract between the
interface and the pipeline*: that a threshold or mask edit re-derives points
from a cached score while a subset-size change pays for a recomputation, that
roles and visibility do what the rows say they do, and that undo puts things
back. Those are the properties the whole design rests on.

Note that a window opens with one seeded ``Whole image`` mask row, so the
helpers below clear it when a test wants to reason about a region of its own.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

sys.ps1 = getattr(sys, "ps1", ">>> ")
QtWidgets.QApplication.exec = lambda self=None: 0

from pyidi.GUIs.feature_selection import (  # noqa: E402
    REDRAW_BUDGET_MS, STEP_FIND, STEP_HINTS, STEP_MASK, SelectionGUI)


def make_image():
    """A speckled frame, so every subset has something to score."""
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, size=(160, 240), dtype=np.uint8)


def make_gui(**kwargs):
    """A headless window on a fresh synthetic frame, as the user gets it."""
    return SelectionGUI(make_image(), **kwargs)


def empty_gui(**kwargs):
    """A window with the seeded whole-image row removed, so nothing is masked."""
    gui = make_gui(**kwargs)
    gui.pipeline.entries = []
    gui.active_index = None
    gui.undo_stack = []
    gui.refresh()
    return gui


def rect(r0, c0, r1, c1):
    """A rectangular polygon as ``(row, col)`` vertices."""
    return [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]


def gui_with_region(**kwargs):
    """A window whose only mask is one polygon covering most of the frame."""
    gui = empty_gui(**kwargs)
    gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
    gui.active_index = 0
    gui.refresh()
    return gui


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor_accepts_a_2d_image():
    gui = make_gui()
    try:
        assert gui.frame.shape == (160, 240)
        assert gui.pipeline.subset_size == (11, 11)
    finally:
        gui.close()


def test_constructor_accepts_a_frame_stack():
    stack = np.stack([make_image(), make_image()])
    gui = SelectionGUI(stack)
    try:
        np.testing.assert_array_equal(gui.frame, stack[0])
    finally:
        gui.close()


def test_constructor_rejects_an_unusable_input():
    with pytest.raises(TypeError, match='VideoReader'):
        SelectionGUI('not a video')


def test_constructor_normalises_an_anisotropic_subset_size():
    gui = make_gui(subset_size=(21, 7))
    try:
        assert gui.pipeline.subset_size == (21, 7)
        assert not gui.square_check.isChecked()
        assert gui.width_spin.isEnabled()
    finally:
        gui.close()


def test_the_image_is_not_transposed():
    """The whole module works in (row, col); nothing should flip the frame."""
    gui = make_gui()
    try:
        np.testing.assert_array_equal(gui.image_item.image, gui.frame)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def test_both_tabs_are_separately_reachable():
    gui = make_gui()
    try:
        for name in (STEP_FIND, STEP_MASK):
            gui.select_step(name)
            assert gui.step == name
            assert gui.step_stack.currentWidget() is gui.step_pages[name]
    finally:
        gui.close()


def test_the_window_opens_on_evaluate_and_select():
    """Evaluate and select do not depend on the mask, so they come first."""
    gui = make_gui()
    try:
        assert gui.step == STEP_FIND
        assert list(gui.step_pages) == [STEP_FIND, STEP_MASK]
    finally:
        gui.close()


def test_the_tabs_are_not_numbered():
    """Numbering would imply an order the pipeline does not have."""
    gui = make_gui()
    try:
        for name, button in gui.step_buttons.items():
            assert button.text() == name
    finally:
        gui.close()


def test_exactly_one_tab_reads_as_active():
    gui = make_gui()
    try:
        for step in (STEP_FIND, STEP_MASK, STEP_FIND):
            gui.select_step(step)
            checked = [name for name, b in gui.step_buttons.items() if b.isChecked()]
            assert checked == [step]
    finally:
        gui.close()


def test_the_checked_tab_and_tool_are_styled_not_left_to_the_theme():
    """A default theme separates checked from unchecked by a shade or two.

    Which is not a difference you can find across a panel, and this interface
    asks the question twice -- which tab, and which tool.
    """
    gui = make_gui()
    try:
        toolbar = gui.step_buttons[STEP_FIND].parent()
        tools = gui.tool_buttons['polygon'].parent()
        for widget in (toolbar, tools):
            assert ':checked' in widget.styleSheet()
            # Only the checked state, so unchecked buttons stay native and the
            # window does not have to carry a theme of its own.
            assert 'QPushButton {' not in widget.styleSheet()
    finally:
        gui.close()


def test_evaluate_and_select_share_one_panel():
    gui = make_gui()
    try:
        page = gui.step_pages[STEP_FIND]
        titles = {box.title() for box in page.findChildren(QtWidgets.QGroupBox)}
        assert {'Evaluate', 'Select'} <= titles
    finally:
        gui.close()


def test_the_highlight_is_a_mask_tab_cue():
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        assert gui.highlight_scatter.isVisible()
        gui.select_step(STEP_FIND)
        assert not gui.highlight_scatter.isVisible()
    finally:
        gui.close()


def test_selecting_without_a_mask_says_so_rather_than_failing():
    gui = empty_gui()
    try:
        gui.select_step(STEP_FIND)
        assert 'mask' in gui.select_note.text().lower()
        assert gui.get_points().shape == (0, 2)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The seeded whole-image row
# ---------------------------------------------------------------------------

def test_the_window_opens_with_candidates_over_the_whole_frame():
    """There has to be something to trim before trimming is a workflow."""
    gui = make_gui()
    try:
        assert len(gui.pipeline.entries) == 1
        assert gui.pipeline.entries[0].label == 'Whole image'
        assert gui.pipeline.entries[0].role == 'mask'
        assert gui.pipeline.mask.all()
        assert len(gui.get_points()) > 0
    finally:
        gui.close()


def test_the_seeded_row_costs_one_evaluation_and_no_more():
    gui = make_gui()
    try:
        assert gui.pipeline.store.n_evaluations == 1
        gui.get_points()
        assert gui.pipeline.store.n_evaluations == 1
    finally:
        gui.close()


def test_the_seeded_row_is_an_ordinary_row():
    gui = make_gui()
    try:
        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert gui.get_points().shape == (0, 2)
        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert len(gui.get_points()) > 0

        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries == []
        assert gui.get_points().shape == (0, 2)
    finally:
        gui.close()


def test_the_seeded_row_can_be_trimmed_with_the_deselect_brush():
    gui = make_gui()
    try:
        before = gui.pipeline.mask.sum()
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.select_tool('erase')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        assert 0 < gui.pipeline.mask.sum() < before
    finally:
        gui.close()


def test_drawing_a_region_adds_to_the_seeded_one():
    """The combined mask is a union, so a new region does not replace it."""
    gui = make_gui()
    try:
        gui.pipeline.add_entry('polygon', rect(20, 20, 60, 60))
        gui.refresh()
        assert len(gui.pipeline.entries) == 2
        assert gui.pipeline.mask.all()
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The contract that makes the interface feel live
# ---------------------------------------------------------------------------

def test_a_threshold_change_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        assert evaluations >= 1
        before = len(gui.get_points())
        gui.threshold_slider.setValue(990)
        assert gui.pipeline.store.n_evaluations == evaluations
        assert len(gui.get_points()) < before
    finally:
        gui.close()


def test_a_separation_change_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        before = len(gui.get_points())
        gui.separation_spin.setValue(30)
        assert gui.pipeline.store.n_evaluations == evaluations
        assert len(gui.get_points()) < before
    finally:
        gui.close()


def test_a_mask_edit_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        gui.pipeline.entries[0].geometry = rect(30, 30, 90, 120)
        gui.refresh()
        assert gui.pipeline.store.n_evaluations == evaluations
    finally:
        gui.close()


def test_a_subset_size_change_does_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        gui.height_spin.setValue(21)
        gui.get_points()
        assert gui.pipeline.store.n_evaluations > evaluations
    finally:
        gui.close()


def test_changing_the_evaluator_re_evaluates_and_keeps_the_old_score_cached():
    gui = gui_with_region()
    try:
        gui.get_points()
        evaluations = gui.pipeline.store.n_evaluations
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
        gui.get_points()
        assert gui.pipeline.store.n_evaluations == evaluations + 1

        # Switching back must be free: the Shi-Tomasi array is still in the cache.
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('shi_tomasi'))
        gui.get_points()
        assert gui.pipeline.store.n_evaluations == evaluations + 1
    finally:
        gui.close()


def test_the_parameter_panel_follows_the_evaluator():
    gui = make_gui()
    try:
        assert gui.param_widgets == {}          # Shi-Tomasi takes no parameters
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
        assert set(gui.param_widgets) == {'direction'}
        assert gui.param_widgets['direction']() == (0.0, 1.0)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Rows: visibility, roles, deletion
# ---------------------------------------------------------------------------

def test_a_row_shows_its_label_role_and_count():
    gui = gui_with_region()
    try:
        text = gui.entry_list.item(0).text()
        assert 'Polygon 1' in text
        assert 'mask' in text
        assert 'pts' in text
    finally:
        gui.close()


def test_unchecking_a_row_drops_its_contribution_without_deleting_it():
    gui = gui_with_region()
    try:
        assert len(gui.get_points()) > 0
        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert gui.get_points().shape == (0, 2)
        assert len(gui.pipeline.entries) == 1

        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert len(gui.get_points()) > 0
    finally:
        gui.close()


def test_switching_a_role_changes_what_the_row_contributes():
    gui = gui_with_region()
    try:
        gui.active_index = 0
        automatic = gui.get_points()
        gui.toggle_role()
        assert gui.pipeline.entries[0].role == 'points'
        literal = gui.get_points()
        # As a points row the polygon lays out its own grid instead of being
        # filtered, so the two results are different sets of points.
        assert len(literal) > 0
        assert not np.array_equal(np.sort(literal, axis=0), np.sort(automatic, axis=0))
        assert not gui.pipeline.mask.any()
    finally:
        gui.close()


def test_deleting_a_row_removes_it():
    gui = gui_with_region()
    try:
        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries == []
        assert gui.entry_list.count() == 0
    finally:
        gui.close()


def test_clear_all_goes_back_to_the_whole_frame():
    """Starting over means the state the window opens in, not a blank frame."""
    gui = gui_with_region()
    try:
        gui.pipeline.add_entry('points', [(50, 50)])
        gui.refresh()
        gui.clear_all()

        assert [entry.label for entry in gui.pipeline.entries] == ['Whole image']
        assert gui.pipeline.mask.all()
        assert len(gui.get_points())
    finally:
        gui.close()


def test_clear_all_is_undoable():
    gui = gui_with_region()
    try:
        before = gui.get_points()
        gui.clear_all()
        gui.undo()
        assert [entry.label for entry in gui.pipeline.entries] == ['Polygon 1']
        np.testing.assert_array_equal(gui.get_points(), before)
    finally:
        gui.close()


def test_deleting_the_whole_image_row_still_selects_nothing():
    """A different act from clearing: "not this area", not "forget what I did"."""
    gui = make_gui()
    try:
        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries == []
        assert gui.get_points().shape == (0, 2)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def test_undo_restores_a_deleted_row_at_its_original_position():
    gui = gui_with_region()
    try:
        first = gui.pipeline.entries[0]
        gui.pipeline.add_entry('points', [(50, 50)])
        gui.refresh()
        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries[0].kind == 'points'

        gui.undo()
        assert gui.pipeline.entries[0] is first
        assert gui.pipeline.entries[0].label == 'Polygon 1'
    finally:
        gui.close()


def test_undo_reverses_a_vertex_add():
    gui = empty_gui()
    try:
        gui.select_tool('polygon')
        for position in [(20, 20), (20, 100), (100, 100)]:
            gui.add_vertex(position)
        entry = gui.pipeline.entries[0]
        assert len(entry.geometry) == 3
        gui.undo()
        assert len(entry.geometry) == 2
    finally:
        gui.close()


def test_undo_reverses_a_brush_stroke():
    gui = empty_gui()
    try:
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        assert len(gui.pipeline.entries) == 1
        gui.undo()
        assert gui.pipeline.entries == []
    finally:
        gui.close()


def test_undo_reverses_a_deselection():
    gui = empty_gui()
    try:
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        painted = gui.pipeline.mask.sum()
        assert painted > 0

        gui.select_tool('erase')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        assert gui.pipeline.mask.sum() < painted

        gui.undo()
        assert gui.pipeline.mask.sum() == painted
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_the_subset_border_pen_is_cosmetic():
    """Cosmetic means the width is in screen pixels, so it survives a zoom."""
    gui = gui_with_region()
    try:
        assert gui.roi_outline.pen().isCosmetic()
        assert not gui.roi_outline.path().isEmpty()
    finally:
        gui.close()


def test_the_subset_overlay_covers_the_expected_area():
    gui = empty_gui(subset_size=(21, 7))
    try:
        gui.pipeline.add_entry('points', [(80, 120)])
        gui.refresh()
        overlay = gui.roi_overlay.image
        covered = overlay[..., 3] != 0
        rows = np.flatnonzero(covered.any(axis=1))
        cols = np.flatnonzero(covered.any(axis=0))
        assert rows.max() - rows.min() + 1 == 21
        assert cols.max() - cols.min() + 1 == 7
    finally:
        gui.close()


def test_hiding_the_subsets_clears_the_overlay():
    gui = gui_with_region()
    try:
        assert gui.roi_overlay.image is not None
        gui.show_subsets.setChecked(False)
        assert gui.roi_overlay.image is None
        assert gui.roi_outline.path().isEmpty()
    finally:
        gui.close()


def test_the_score_overlay_is_transparent_on_the_invalid_border():
    gui = gui_with_region()
    try:
        gui.show_score.setChecked(True)
        assert gui.score_overlay.isVisible()
        rgba = gui.score_overlay.image
        assert (rgba[:5, :, 3] == 0).all()
        assert (rgba[80, 100:140, 3] > 0).all()
    finally:
        gui.close()


def test_the_score_overlay_sits_beside_the_selection_controls():
    """Seeing the score while thresholding is how you tell "too tight" from "nothing there"."""
    gui = gui_with_region()
    try:
        page = gui.step_pages[STEP_FIND]
        assert gui.show_score in page.findChildren(QtWidgets.QCheckBox)
        assert gui.threshold_slider in page.findChildren(QtWidgets.QSlider)
    finally:
        gui.close()


def test_toggling_the_overlay_leaves_the_selection_alone():
    gui = gui_with_region()
    try:
        before = gui.get_points()
        gui.show_score.setChecked(True)
        gui.show_score.setChecked(False)
        np.testing.assert_array_equal(gui.get_points(), before)
    finally:
        gui.close()


def test_a_large_selection_redraws_quickly():
    """Thousands of subsets must stay interactive, hence the raster-plus-path split."""
    import time

    gui = make_gui(subset_size=5)
    try:
        rng = np.random.default_rng(11)
        rows = rng.integers(10, 150, 5000)
        cols = rng.integers(10, 230, 5000)
        points = np.column_stack([rows, cols])
        start = time.perf_counter()
        gui.draw_subset_rectangles(points, 2, 2)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f'redraw of 5000 subsets took {elapsed:.3f} s'
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def test_points_are_row_col_integers_inside_the_frame():
    gui = gui_with_region()
    try:
        points = gui.points
        assert points.dtype.kind == 'i'
        assert points.shape[1] == 2
        assert (points[:, 0] < gui.frame.shape[0]).all()
        assert (points[:, 1] < gui.frame.shape[1]).all()
        np.testing.assert_array_equal(points, gui.get_points())
    finally:
        gui.close()


def test_points_are_accepted_by_a_method_class(tmp_path):
    import warnings

    from pyidi import SimplifiedOpticalFlow, VideoReader

    gui = gui_with_region()
    try:
        points = gui.points
        assert len(points)
        video = VideoReader(np.stack([gui.frame, gui.frame]), root=str(tmp_path))
        method = SimplifiedOpticalFlow(video)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            method.set_points(points)
        np.testing.assert_array_equal(method.points, points)
    finally:
        gui.close()


def test_hand_picked_points_survive_a_high_threshold():
    gui = gui_with_region()
    try:
        gui.pipeline.add_entry('points', [(80, 120)])
        gui.threshold_slider.setValue(999)
        points = gui.get_points()
        assert (points == np.array([80, 120])).all(axis=1).any()
    finally:
        gui.close()


def test_the_count_label_follows_the_points():
    gui = gui_with_region()
    try:
        assert gui.count_label.text() == f'{len(gui.get_points())} points'
        gui.threshold_slider.setValue(995)
        gui.flush_refresh()
        assert gui.count_label.text() == f'{len(gui.get_points())} points'
    finally:
        gui.close()


def test_the_lattice_selector_gives_regular_spacing():
    gui = gui_with_region()
    try:
        gui.selector_combo.setCurrentIndex(gui.selector_combo.findData('lattice'))
        gui.pitch_spin.setValue(20)
        gui.threshold_slider.setValue(0)
        points = gui.get_points()
        assert len(points) > 4
        assert set(np.diff(sorted(set(points[:, 0].tolist())))) <= {20}
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------

def test_selection_gui_names_this_interface():
    """``SelectionGUI`` is this class, not the 1.3 window it replaced."""
    import pyidi

    assert pyidi.SelectionGUI is SelectionGUI


def test_the_old_interface_is_still_reachable_under_its_own_name():
    import pyidi
    from pyidi.GUIs.subset_selection import SelectionGUIOld

    assert pyidi.SelectionGUIOld is SelectionGUIOld
    assert pyidi.SelectionGUIOld is not pyidi.SelectionGUI


def test_the_old_interface_warns_on_construction():
    from pyidi import SelectionGUIOld

    with pytest.deprecated_call():
        gui = SelectionGUIOld(make_image())
    gui.close()


def test_the_working_name_says_where_it_went():
    """``FeatureSelectionGUI`` never shipped, but it is in the design notes."""
    import pyidi

    with pytest.raises(RuntimeError, match='now called SelectionGUI'):
        pyidi.FeatureSelectionGUI()


# ---------------------------------------------------------------------------
# Where the controls live
# ---------------------------------------------------------------------------

def test_the_subset_size_shows_on_every_tab():
    """It drives the score and it measures the drawn rectangles, so it is neither tab's."""
    gui = make_gui()
    try:
        for name in (STEP_FIND, STEP_MASK):
            gui.select_step(name)
            assert gui.height_spin.isVisible()
            assert gui.width_spin.isVisible()
        # ... by living outside the pages rather than being duplicated on each.
        for page in gui.step_pages.values():
            assert gui.height_spin not in page.findChildren(QtWidgets.QSpinBox)
    finally:
        gui.close()


def test_the_point_spacing_stays_with_the_mask():
    """It only affects rows that lay points out, which is a masking concern."""
    gui = make_gui()
    try:
        assert gui.spacing_spin in gui.step_pages[STEP_MASK].findChildren(QtWidgets.QSpinBox)
    finally:
        gui.close()


def test_both_tabs_offer_the_score_overlay():
    gui = make_gui()
    try:
        assert gui.show_score in gui.step_pages[STEP_FIND].findChildren(QtWidgets.QCheckBox)
        assert gui.show_score_mask in gui.step_pages[STEP_MASK].findChildren(QtWidgets.QCheckBox)
    finally:
        gui.close()


def test_the_score_toggles_are_ganged():
    """One overlay, two checkboxes: they must not disagree about its state."""
    gui = gui_with_region()
    try:
        gui.show_score_mask.setChecked(True)
        assert gui.show_score.isChecked()
        assert gui.score_overlay.isVisible()

        gui.show_score.setChecked(False)
        assert not gui.show_score_mask.isChecked()
        assert not gui.score_overlay.isVisible()
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Gradient direction
# ---------------------------------------------------------------------------

def use_gradient(gui):
    """Switch the evaluator to the direction-taking one."""
    gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
    return gui


def test_the_direction_can_be_dragged_out_on_the_image():
    gui = use_gradient(make_gui())
    try:
        gui.direction_button.setChecked(True)
        assert gui.drawing_direction

        gui.set_direction_from_drag((10.0, 10.0), (10.0, 40.0))    # straight along +col

        row, col = (spin.value() for spin in gui.direction_spins)
        assert row == pytest.approx(0.0, abs=1e-6)
        assert col == pytest.approx(1.0, abs=1e-6)
        assert not gui.drawing_direction        # one drag sets it once
    finally:
        gui.close()


def test_the_dragged_direction_reaches_the_evaluator():
    gui = use_gradient(make_gui())
    try:
        gui.set_direction_from_drag((0.0, 0.0), (30.0, 0.0))       # straight along +row
        spec = gui.pipeline.store.spec(gui.pipeline.default_score)
        assert spec.evaluator == 'gradient_direction'
        assert dict(spec.parameters)['direction'] == pytest.approx((1.0, 0.0))
    finally:
        gui.close()


def test_the_direction_line_shows_where_it_was_dragged():
    gui = use_gradient(make_gui())
    try:
        gui.set_direction_from_drag((20.0, 30.0), (60.0, 90.0))
        xs, ys = gui.direction_line.getData()
        np.testing.assert_allclose(xs, [30.0, 90.0])               # x is the column
        np.testing.assert_allclose(ys, [20.0, 60.0])
    finally:
        gui.close()


def test_a_preset_costs_one_evaluation_not_two():
    """Writing two components must not evaluate once per component."""
    gui = use_gradient(make_gui())
    try:
        before = gui.pipeline.store.n_evaluations
        gui.set_direction(1.0, 1.0)
        gui.flush_refresh()
        assert gui.pipeline.store.n_evaluations == before + 1
        row, col = (spin.value() for spin in gui.direction_spins)
        assert (row, col) == pytest.approx((2 ** -0.5, 2 ** -0.5), abs=1e-3)
    finally:
        gui.close()


def test_a_zero_direction_is_ignored():
    gui = use_gradient(make_gui())
    try:
        before = [spin.value() for spin in gui.direction_spins]
        gui.set_direction(0.0, 0.0)
        assert [spin.value() for spin in gui.direction_spins] == before
    finally:
        gui.close()


def test_leaving_the_gradient_evaluator_drops_its_direction_line():
    gui = use_gradient(make_gui())
    try:
        gui.set_direction_from_drag((0.0, 0.0), (0.0, 30.0))
        assert gui.direction_line.getData()[0] is not None

        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('shi_tomasi'))
        assert gui.direction_line.getData()[0] is None
        assert gui.direction_spins == []
        assert not gui.drawing_direction
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Masking is visible in the points
# ---------------------------------------------------------------------------

def test_drawing_a_region_stands_the_whole_image_row_down():
    """Masks are a union, so a drawn region would otherwise change nothing."""
    gui = make_gui()
    try:
        seeded = gui.pipeline.entries[0]
        assert seeded.label == 'Whole image' and seeded.visible

        gui.pipeline.add_entry('polygon', rect(20, 20, 60, 60))
        gui._retire_whole_image()
        gui.refresh()

        assert not seeded.visible
        assert gui.pipeline.mask[40, 40]
        assert not gui.pipeline.mask[120, 200]      # outside the drawn region
    finally:
        gui.close()


def test_standing_the_whole_image_row_down_is_undoable():
    gui = make_gui()
    try:
        seeded = gui.pipeline.entries[0]
        gui.pipeline.add_entry('polygon', rect(20, 20, 60, 60))
        gui._retire_whole_image()
        gui.undo()
        assert seeded.visible
    finally:
        gui.close()


def test_an_empty_polygon_leaves_the_whole_image_row_alone():
    """Two vertices enclose nothing, so there is no region to make way for yet."""
    gui = make_gui()
    try:
        gui.pipeline.add_entry('polygon', [(20.0, 20.0), (20.0, 60.0)])
        gui._retire_whole_image()
        assert gui.pipeline.entries[0].visible
    finally:
        gui.close()


def test_a_points_row_leaves_the_whole_image_row_alone():
    """A literal point adds to the selection rather than restricting it."""
    gui = make_gui()
    try:
        gui.pipeline.add_entry('points', [(80, 120)])
        gui._retire_whole_image()
        assert gui.pipeline.entries[0].visible
    finally:
        gui.close()


def crossed_out(gui):
    """How many points the deselect brush is currently showing as doomed."""
    xs = gui.doomed_scatter.getData()[0]
    return 0 if xs is None else len(xs)


def test_a_deselect_stroke_crosses_out_the_points_it_covers():
    """Feedback while the stroke is being painted, not only when the mouse comes up."""
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.select_tool('erase')

        gui.brush_start()
        gui.brush_radius.setValue(40)
        gui.brush_move((80.0, 120.0))

        doomed = crossed_out(gui)
        assert doomed > 0
        # The crosses go over the red points rather than replacing them: a stroke
        # then costs only the points it has reached, not the whole cloud.
        assert len(gui.point_scatter.getData()[0]) == len(gui.get_points())
    finally:
        gui.close()


def test_painting_a_stroke_leaves_the_point_cloud_alone():
    """The cost of a mouse move is the points it reached, not every point drawn.

    Handing tens of thousands of positions back to the scatter item on every
    move is what made a long stroke lag behind the cursor.
    """
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.select_tool('erase')
        before = gui.point_scatter.getData()[0].copy()

        gui.brush_start()
        gui.brush_radius.setValue(40)
        gui.brush_move((80.0, 120.0))

        assert crossed_out(gui) > 0
        assert np.array_equal(gui.point_scatter.getData()[0], before)
    finally:
        gui.close()


def test_the_stroke_is_drawn_without_rebuilding_a_full_frame_image():
    """A raster overlay costs the whole frame per move, however small the dab."""
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_move((80.0, 120.0))
        assert not gui.brush_overlay.path().isEmpty()
        gui.brush_end()
        assert gui.brush_overlay.path().isEmpty()
    finally:
        gui.close()


def test_the_crossed_out_points_are_gone_once_the_stroke_lands():
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.select_tool('erase')
        before = len(gui.get_points())

        gui.brush_start()
        gui.brush_radius.setValue(40)
        gui.brush_move((80.0, 120.0))
        stroke = gui._paint.copy()
        gui.brush_end()

        after = gui.get_points()
        assert len(after) < before
        # The count is not simply `before - doomed`: freeing the area lets the
        # minimum-distance rule admit points it had previously suppressed.
        assert not stroke[after[:, 0], after[:, 1]].any()
        assert crossed_out(gui) == 0
    finally:
        gui.close()


def test_a_selecting_stroke_crosses_nothing_out():
    """Painting a new region takes nothing away, so nothing should look doomed."""
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_radius.setValue(40)
        gui.brush_move((80.0, 120.0))
        assert crossed_out(gui) == 0
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------

def test_no_group_box_contains_another():
    """Two nested frames cost two sets of margins out of an already narrow panel."""
    gui = make_gui()
    try:
        for page in gui.step_pages.values():
            for box in page.findChildren(QtWidgets.QGroupBox):
                assert not box.findChildren(QtWidgets.QGroupBox), box.title()
    finally:
        gui.close()


def test_only_the_current_selectors_settings_are_shown():
    """A greyed-out row is a line of panel spent saying the line does not apply."""
    gui = make_gui()
    try:
        assert gui.separation_spin.isVisible()
        assert not gui.pitch_spin.isVisible()

        gui.selector_combo.setCurrentIndex(gui.selector_combo.findData('lattice'))
        assert gui.pitch_spin.isVisible()
        assert not gui.separation_spin.isVisible()
    finally:
        gui.close()


def test_the_evaluator_describes_itself_in_a_tooltip():
    """It used to be a paragraph on the panel, which you read once."""
    gui = make_gui()
    try:
        assert 'corner' in gui.evaluator_combo.toolTip().lower()
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
        assert 'direction' in gui.evaluator_combo.toolTip().lower()
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Quality threshold and decimation
# ---------------------------------------------------------------------------

def test_the_threshold_defaults_to_quality():
    """Percentile spends most of its travel inside the featureless background."""
    gui = make_gui()
    try:
        assert gui.threshold_mode.currentData() == 'quality'
        assert gui.pipeline.selector_params['threshold_mode'] == 'quality'
        assert gui.pipeline.selector_params['threshold'] == pytest.approx(0.01, rel=0.02)
    finally:
        gui.close()


def test_the_quality_slider_is_logarithmic():
    """The useful settings span three decades, so a linear slider wastes most of itself."""
    gui = make_gui()
    try:
        seen = []
        for position in (0, 250, 500, 750, 1000):
            gui.threshold_slider.setValue(position)
            seen.append(gui.pipeline.selector_params['threshold'])
        assert seen[0] == pytest.approx(0.001)
        assert seen[-1] == pytest.approx(1.0)
        ratios = [b / a for a, b in zip(seen, seen[1:])]
        assert all(r == pytest.approx(ratios[0], rel=1e-6) for r in ratios)
    finally:
        gui.close()


def test_switching_the_rule_moves_the_slider_to_that_rules_default():
    """The same position means 90 under one rule and 0.5 under another."""
    gui = make_gui()
    try:
        gui.threshold_mode.setCurrentIndex(gui.threshold_mode.findData('percentile'))
        assert gui.pipeline.selector_params['threshold'] == pytest.approx(90.0)

        gui.threshold_mode.setCurrentIndex(gui.threshold_mode.findData('quality'))
        assert gui.pipeline.selector_params['threshold'] == pytest.approx(0.01, rel=0.02)
    finally:
        gui.close()


def test_quality_keeps_the_points_off_the_blank_background():
    """The reason the default changed, end to end."""
    frame = np.full((200, 300), 240, dtype=np.uint8)
    corners = np.zeros(frame.shape, dtype=bool)
    for row in range(40, 180, 60):
        for col in range(40, 280, 60):
            frame[row:row + 20, col:col + 20] = 20
            corners[row - 9:row + 29, col - 9:col + 29] = True
    frame = np.clip(frame.astype(int) + np.random.default_rng(3).integers(-5, 6, frame.shape),
                    0, 255).astype(np.uint8)

    gui = SelectionGUI(frame, subset_size=11)
    try:
        for position in (1000, 750, 500, 333):
            gui.threshold_slider.setValue(position)
            points = gui.get_points()
            assert len(points)
            assert corners[points[:, 0], points[:, 1]].all(), gui.threshold_label.text()
    finally:
        gui.close()


def test_decimation_thins_without_moving_the_survivors():
    gui = gui_with_region()
    try:
        gui.separation_spin.setValue(6)
        before = {tuple(point) for point in gui.get_points().tolist()}
        assert len(before) > 20

        gui.decimation_spin.setValue(3)
        after = [tuple(point) for point in gui.get_points().tolist()]
        assert set(after) <= before
        assert len(after) == pytest.approx(len(before) / 3, rel=0.2)
    finally:
        gui.close()


def test_decimation_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        before = gui.pipeline.store.n_evaluations
        gui.decimation_spin.setValue(4)
        assert gui.pipeline.store.n_evaluations == before
    finally:
        gui.close()


def test_a_redraw_runs_the_pipeline_once():
    """It used to run three times: the total, the row counts and the highlight."""
    gui = gui_with_region()
    try:
        calls = []
        original = gui.pipeline.picked_points
        gui.pipeline.picked_points = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        gui.select_step(STEP_MASK)         # the highlight is drawn on this tab
        calls.clear()
        gui.refresh()
        assert len(calls) == 1
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Coalescing the redraws a dragged control produces
# ---------------------------------------------------------------------------

def test_a_cheap_redraw_happens_immediately():
    """A live slider is the whole point; deferring a redraw we can afford buys nothing."""
    gui = gui_with_region()
    try:
        gui._last_refresh_ms = 0.0
        gui.threshold_slider.setValue(700)
        assert not gui._refresh_timer.isActive()
        assert len(gui._points) == len(gui.get_points())
    finally:
        gui.close()


def test_an_expensive_redraw_is_deferred_and_coalesced():
    """Twenty positions on the way past cost one redraw, not twenty."""
    gui = gui_with_region()
    try:
        gui._last_refresh_ms = REDRAW_BUDGET_MS + 1.0
        stale = gui._points
        for position in range(700, 720):
            gui.threshold_slider.setValue(position)
        assert gui._refresh_timer.isActive()
        np.testing.assert_array_equal(gui._points, stale)     # nothing redrawn yet

        gui.flush_refresh()
        assert not gui._refresh_timer.isActive()
        np.testing.assert_array_equal(gui._points, gui.get_points())
    finally:
        gui.close()


def test_the_deferred_redraw_lands_on_the_value_the_control_stopped_at():
    gui = gui_with_region()
    try:
        gui._last_refresh_ms = REDRAW_BUDGET_MS + 1.0
        gui.threshold_slider.setValue(400)
        gui.threshold_slider.setValue(900)
        gui.flush_refresh()
        assert len(gui._points) == len(gui.get_points())
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# One density control, and it is not a stride
# ---------------------------------------------------------------------------

def test_the_separation_is_the_density_control():
    gui = gui_with_region()
    try:
        gui.separation_spin.setValue(6)
        loose = gui.get_points()
        gui.separation_spin.setValue(18)
        tight = gui.get_points()
        assert len(tight) < len(loose)
        for points, separation in ((loose, 6), (tight, 18)):
            gaps = np.hypot(points[:, None, 0] - points[None, :, 0],
                            points[:, None, 1] - points[None, :, 1])
            np.fill_diagonal(gaps, np.inf)
            assert gaps.min() >= separation
    finally:
        gui.close()


def test_the_separation_cannot_be_turned_off():
    """Zero was the setting that made the point cap look like a tight cluster."""
    gui = make_gui()
    try:
        assert gui.separation_spin.minimum() == 1
    finally:
        gui.close()


def test_the_threshold_menu_offers_quality_and_percentile_only():
    gui = make_gui()
    try:
        rules = [gui.threshold_mode.itemData(i) for i in range(gui.threshold_mode.count())]
        assert rules == ['quality', 'percentile']
    finally:
        gui.close()


def test_the_cap_says_so_rather_than_looking_like_a_threshold():
    gui = gui_with_region()
    try:
        gui.max_points_spin.setValue(20)
        gui.separation_spin.setValue(2)
        gui.flush_refresh()
        assert '20' in gui.select_note.text()
        assert 'cap' in gui.select_note.text().lower()

        gui.separation_spin.setValue(40)
        gui.flush_refresh()
        assert gui.select_note.text() == ''
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# What the mask is leaving out
# ---------------------------------------------------------------------------

def rendered(gui):
    """The canvas as three ``(h, w)`` integer channels: red, green, blue.

    Copied out of the QImage, which owns the buffer and frees it on the way out,
    and signed, so that one channel can be subtracted from another.
    """
    image = gui.pg_widget.grab().toImage().convertToFormat(
        QtGui.QImage.Format.Format_RGB32)
    raw = np.frombuffer(image.constBits().asarray(image.sizeInBytes()), dtype=np.uint8)
    pixels = raw.reshape(image.height(), image.width(), 4).astype(np.int16)
    return pixels[..., 2], pixels[..., 1], pixels[..., 0]


def test_the_two_point_layers_actually_paint():
    """Rendered, not just handed the right coordinates.

    The dots are one stroked path rather than a scatter item, and Qt draws
    nothing at all for a zero-length subpath -- a failure no assertion about what
    the item was *given* can see.
    """
    gui = empty_gui()
    try:
        gui.pipeline.add_entry('polygon', rect(20, 20, 60, 60))
        gui.active_index = 0
        gui.select_step(STEP_MASK)
        gui.show_subsets.setChecked(False)
        gui.refresh()

        red, green, blue = rendered(gui)
        # The red points, told apart from the magenta rings by their blue.
        assert ((red > 150) & (red - blue > 60) & (red - green > 40)).sum() > 20
        # The dim blue candidates outside the polygon.
        assert ((blue > 120) & (blue - red > 60)).sum() > 20
    finally:
        gui.close()


def greyed(gui):
    """The dimmed candidate positions currently drawn, as ``(row, col)``."""
    x, y = gui.candidate_scatter.getData()
    if x is None or not len(x):
        return np.zeros((0, 2), dtype=int)
    return np.column_stack([y - 0.5, x - 0.5]).astype(int)


def test_the_mask_tab_shows_the_features_the_mask_leaves_out():
    """"No point here" is otherwise two different things wearing one face."""
    gui = empty_gui()
    try:
        gui.pipeline.add_entry('polygon', rect(20, 20, 60, 60))
        gui.active_index = 0
        gui.select_step(STEP_MASK)
        outside = greyed(gui)
        assert len(outside)
        mask = gui.pipeline.mask
        assert not mask[outside[:, 0], outside[:, 1]].any()
    finally:
        gui.close()


def test_masking_an_area_takes_its_points_out_of_the_dimmed_set():
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        before = len(greyed(gui))
        gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
        gui.refresh()
        assert len(greyed(gui)) < before
    finally:
        gui.close()


def test_the_dimmed_candidates_are_a_mask_tab_cue():
    """On the other tab everything drawn is selected, so a second tier would only confuse."""
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        assert len(greyed(gui))
        gui.select_step(STEP_FIND)
        assert not len(greyed(gui))
    finally:
        gui.close()


def test_editing_a_mask_does_not_move_the_candidates():
    """They are the whole-frame selection, so painting turns points red where it lands."""
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        before = gui.pipeline.candidate_points()
        gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
        gui.refresh()
        assert gui.pipeline.candidate_points() is before
    finally:
        gui.close()


def test_a_new_threshold_does_move_the_candidates():
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        before = len(gui.pipeline.candidate_points())
        gui.threshold_slider.setValue(gui.threshold_slider.maximum())
        gui.flush_refresh()
        assert len(gui.pipeline.candidate_points()) < before
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Tab names and where the panel's boxes live
# ---------------------------------------------------------------------------

def test_the_tabs_are_named_for_the_pipeline_steps():
    """Not "find points": the steps have names, and the tabs hold exactly those."""
    assert STEP_MASK.lower() == 'mask'
    assert 'evaluate' in STEP_FIND.lower() and 'select' in STEP_FIND.lower()
    # An '&' would be read as a mnemonic and swallowed out of the button label.
    assert '&' not in STEP_FIND


def test_the_selections_list_is_a_mask_tab_control():
    """Every row in it, and every button under it, acts on something drawn there."""
    gui = make_gui()
    try:
        gui.select_step(STEP_MASK)
        assert gui.selection_box.isVisible()
        gui.select_step(STEP_FIND)
        assert not gui.selection_box.isVisible()
    finally:
        gui.close()


def test_the_subset_group_stays_on_both_tabs():
    """The counterpart to the list moving: this one really is read by both steps."""
    gui = make_gui()
    try:
        for name in (STEP_FIND, STEP_MASK):
            gui.select_step(name)
            assert gui.height_spin.isVisible()
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------

def test_every_setting_explains_itself():
    gui = make_gui()
    try:
        for widget in (gui.selector_combo, gui.threshold_mode, gui.threshold_slider,
                       gui.separation_spin, gui.pitch_spin, gui.max_points_spin,
                       gui.decimation_spin, gui.spacing_spin, gui.height_spin,
                       gui.width_spin, gui.show_subsets, gui.show_score,
                       gui.role_button, gui.brush_radius):
            assert widget.toolTip(), widget
    finally:
        gui.close()


def test_no_tooltip_or_hint_uses_a_name_the_interface_dropped():
    """Stale help is worse than none: it names a control that is not there."""
    gui = make_gui()
    try:
        texts = [w.toolTip() for w in gui.findChildren(QtWidgets.QWidget)]
        texts += list(STEP_HINTS.values())
        for text in texts:
            lowered = text.lower()
            assert 'minimum distance' not in lowered
            assert 'fraction of the maximum' not in lowered
            assert 'find points' not in lowered
            # 'Deselect painted area' was the checkable button that 'Remove
            # area' replaced; nothing should still send anyone looking for it.
            assert 'deselect painted area' not in lowered
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Odd subset sizes
# ---------------------------------------------------------------------------

def test_the_subset_size_steps_in_odds():
    """A subset is centred on its point, so an even extent has no centre to be."""
    gui = make_gui(subset_size=11)
    try:
        gui.height_spin.stepUp()
        assert gui.height_spin.value() == 13
        gui.height_spin.stepDown()
        gui.height_spin.stepDown()
        assert gui.height_spin.value() == 9
    finally:
        gui.close()


def test_an_even_subset_size_cannot_be_typed_in():
    gui = make_gui()
    try:
        gui.height_spin.lineEdit().setText('12')
        gui.height_spin.interpretText()
        assert gui.height_spin.value() == 13
    finally:
        gui.close()


def test_an_even_subset_size_is_rounded_up_on_the_way_in():
    """The pipeline and the control that shows it must not disagree."""
    gui = make_gui(subset_size=(10, 20))
    try:
        assert gui.pipeline.subset_size == (11, 21)
        assert (gui.height_spin.value(), gui.width_spin.value()) == (11, 21)
    finally:
        gui.close()


def test_the_square_toggle_keeps_both_odd():
    gui = make_gui(subset_size=(11, 21))
    try:
        gui.square_check.setChecked(True)
        assert gui.width_spin.value() == gui.height_spin.value()
        assert gui.pipeline.subset_size == (11, 11)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Removing a point by clicking it
# ---------------------------------------------------------------------------

def test_the_remove_tool_takes_a_point_away_every_time():
    """The regression: only the first click did anything.

    ``erased`` was grown with an in-place write, and the pipeline identifies
    that array by object -- so once it existed, no later click changed anything
    the rasterisation cache could see.
    """
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('remove')
        for _ in range(4):
            target = tuple(int(v) for v in gui._points[0])
            gui.remove_nearest_point(target)
            gui.refresh()
            assert not any(tuple(p) == target for p in gui._points)
    finally:
        gui.close()


def test_removing_a_point_is_undoable():
    gui = gui_with_region()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('remove')
        before = len(gui._points)
        target = tuple(int(v) for v in gui._points[0])

        gui.remove_nearest_point(target)
        gui.refresh()
        assert not any(tuple(p) == target for p in gui._points)

        gui.undo()
        assert len(gui._points) == before
        assert any(tuple(p) == target for p in gui._points)
    finally:
        gui.close()


def test_a_click_far_from_every_point_removes_nothing():
    gui = gui_with_region()
    try:
        before = gui.pipeline.entries[0].erased
        gui.remove_nearest_point((0, 0))
        assert gui.pipeline.entries[0].erased is before
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Clicks that land off the image
#
# The aspect is locked, so one axis always has a margin, and zooming out adds
# more. A subset centred off the frame is not something that can be tracked.
# ---------------------------------------------------------------------------

def test_a_click_outside_the_image_adds_no_point():
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('points')
        rows, cols = gui.pipeline.shape
        gui.add_point((rows + 40.0, 20.0))
        gui.add_point((20.0, -12.0))
        assert gui.pipeline.entries == []
    finally:
        gui.close()


def test_a_click_on_the_image_still_adds_one():
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('points')
        gui.add_point((80.0, 120.0))
        assert gui.pipeline.entries[0].geometry == [(80, 120)]
    finally:
        gui.close()


def test_an_off_frame_point_cannot_reach_the_result():
    """Whatever route it arrived by -- it is an index error in every array."""
    gui = empty_gui()
    try:
        gui.pipeline.add_entry('points', [(80, 120), (999, 120)])
        gui.refresh()
        assert [tuple(p) for p in gui.get_points()] == [(80, 120)]

        gui.select_tool('erase')
        gui.brush_start()
        gui.brush_move((80.0, 120.0))       # would index row 999 of a 160-row frame
        gui.brush_end()
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# What an evaluator change costs
#
# This is the one control that can make a redraw expensive: a parameter the
# store has not scored before is a whole-frame evaluation.
# ---------------------------------------------------------------------------

def select_evaluator(gui, name):
    """Put the evaluator combo on a registry name."""
    for index in range(gui.evaluator_combo.count()):
        if gui.evaluator_combo.itemData(index) == name:
            gui.evaluator_combo.setCurrentIndex(index)
            return
    raise AssertionError(f'no {name!r} in the evaluator menu')


def test_a_parameter_change_is_coalesced_like_every_other_control():
    gui = gui_with_region()
    try:
        select_evaluator(gui, 'gradient_direction')
        gui.refresh()
        gui._last_refresh_ms = REDRAW_BUDGET_MS + 1
        gui.direction_spins[0].setValue(0.37)
        assert gui._refresh_timer.isActive()
    finally:
        gui.close()


def test_dragging_a_parameter_does_not_hoard_a_score_per_value():
    """Each one is a full-frame float32; sixty of them is a gigabyte at 4 MP."""
    gui = gui_with_region()
    try:
        select_evaluator(gui, 'gradient_direction')
        for step in range(40):
            gui.direction_spins[0].setValue(0.02 * step)
            gui.flush_refresh()
        assert len(gui.pipeline.store._cache) <= gui.pipeline.store.max_cached
    finally:
        gui.close()


def test_the_score_overlay_follows_a_change_of_subset_size():
    """It is drawn from the score, so it has to be part of the redraw."""
    gui = gui_with_region()
    try:
        gui.show_score.setChecked(True)
        before = gui.score_overlay.image.copy()
        gui.height_spin.setValue(31)
        gui.flush_refresh()
        assert not np.array_equal(gui.score_overlay.image, before)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Ctrl, and what a stroke costs to remember
# ---------------------------------------------------------------------------

class FakeDrag:
    """The parts of a pyqtgraph drag event the canvas actually reads."""

    def __init__(self, ctrl=False, start=False, finish=False):
        self._ctrl = ctrl
        self._start = start
        self._finish = finish
        self.accepted = False

    def modifiers(self):
        return (QtCore.Qt.KeyboardModifier.ControlModifier if self._ctrl
                else QtCore.Qt.KeyboardModifier.NoModifier)

    def isStart(self):
        return self._start

    def isFinish(self):
        return self._finish

    def accept(self):
        self.accepted = True

    def scenePos(self):
        return QtCore.QPointF(0.0, 0.0)

    def buttonDownScenePos(self):
        return QtCore.QPointF(0.0, 0.0)


def test_ctrl_is_read_off_the_event_not_tracked():
    """A panel widget with focus can swallow the key; a tracked flag then lies."""
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        assert not gui.view._handle_brush_drag(FakeDrag(ctrl=False, start=True))
        assert gui.view._handle_brush_drag(FakeDrag(ctrl=True, start=True))
        assert gui.painting
    finally:
        gui.close()


def test_letting_go_of_ctrl_mid_stroke_still_finishes_the_stroke():
    """Otherwise the drag stops being handled and the stroke is silently lost."""
    gui = empty_gui()
    try:
        gui.select_step(STEP_MASK)
        gui.select_tool('brush')
        gui.view._handle_brush_drag(FakeDrag(ctrl=True, start=True))
        gui.brush_move((80.0, 120.0))

        assert gui.view._handle_brush_drag(FakeDrag(ctrl=False))
        assert gui.view._handle_brush_drag(FakeDrag(ctrl=False, finish=True))
        assert not gui.painting
        assert gui.pipeline.mask.sum() > 0
    finally:
        gui.close()


def test_an_undo_snapshot_does_not_copy_the_erased_arrays():
    """One frame of booleans per region, in each of fifty undo slots.

    Safe to hold by reference because ``erased`` is always replaced, never
    written into -- the contract on :class:`~pyidi.selection.masks.Entry`.
    """
    gui = gui_with_region()
    try:
        gui.select_tool('brush')
        gui.select_tool('erase')
        gui.brush_start()
        gui.brush_move((80.0, 120.0))
        gui.brush_end()

        entry = gui.pipeline.entries[0]
        assert entry.erased is not None
        held = {id(state[1]) for state in gui._snapshot()['state']}
        assert id(entry.erased) in held
    finally:
        gui.close()


def test_the_seeded_row_is_recognised_by_identity_not_by_its_label():
    gui = make_gui()
    try:
        seeded = gui._whole_image
        seeded.label = 'Renamed by hand'
        gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
        gui._retire_whole_image()
        assert not seeded.visible
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The mask tab shows only what the current tool and rows can act on
# ---------------------------------------------------------------------------

def test_the_brush_radius_follows_the_tools_that_paint():
    """Painting is gated on the tool, so with any other one the radius does nothing."""
    gui = make_gui()
    try:
        gui.select_step(STEP_MASK)

        gui.select_tool('polygon')
        assert not gui.brush_radius.isVisible()

        gui.select_tool('brush')
        assert gui.brush_radius.isVisible()

        # Both brush tools share it: a stroke erases as wide as it paints.
        gui.select_tool('erase')
        assert gui.brush_radius.isVisible()

        gui.select_tool('points')
        assert not gui.brush_radius.isVisible()
    finally:
        gui.close()


def test_erasing_is_a_tool_rather_than_a_mode_the_brush_is_in():
    """The two things that take away are tools; neither hides inside the other."""
    gui = make_gui()
    try:
        gui.select_step(STEP_MASK)
        assert 'erase' in gui.tool_buttons
        assert 'remove' in gui.tool_buttons

        gui.select_tool('brush')
        assert not gui.deselect_mode

        gui.select_tool('erase')
        assert gui.deselect_mode

        # Leaving the tool leaves the mode, because they are the same thing.
        gui.select_tool('brush')
        assert not gui.deselect_mode
    finally:
        gui.close()


def test_point_spacing_appears_only_for_a_row_that_lays_points_out():
    """It reaches ``literal_points`` and nowhere else."""
    gui = make_gui()
    try:
        gui.select_step(STEP_MASK)
        # Only the seeded whole-image mask so far: nothing spacing can move.
        assert not gui.spacing_spin.isVisible()

        entry = gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
        gui.refresh()
        assert not gui.spacing_spin.isVisible()

        entry.role = 'points'
        gui.refresh()
        assert gui.spacing_spin.isVisible()

        entry.role = 'mask'
        gui.refresh()
        assert not gui.spacing_spin.isVisible()
    finally:
        gui.close()


def test_point_spacing_stays_hidden_for_hand_clicked_points():
    """A ``points``-tool row is the coordinates you clicked; spacing has no say."""
    gui = make_gui()
    try:
        gui.select_step(STEP_MASK)
        gui.pipeline.add_entry('points', [(40, 40), (60, 60)])
        gui.refresh()
        assert not gui.spacing_spin.isVisible()

        gui.pipeline.add_entry('polyline', [(30, 30), (30, 200)])
        gui.refresh()
        assert gui.spacing_spin.isVisible()
    finally:
        gui.close()
