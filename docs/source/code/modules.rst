API reference
=============

Everything below is generated from the docstrings in the source. For a
task-oriented introduction, start from the :doc:`tutorial
<../quick_start/basic_usage>` instead.

Top-level namespace
-------------------

These names are importable directly from ``pyidi``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Purpose
   * - :class:`~pyidi.video_reader.VideoReader`
     - Read a recording of any supported format.
   * - ``SimplifiedOpticalFlow``, ``LucasKanade``, ``DirectionalLucasKanade``,
       ``DIC``
     - The displacement identification methods.
   * - :class:`~pyidi.GUIs.feature_selection.SelectionGUI`
     - Interactive point selection (requires the ``[qt]`` extra).
   * - :class:`~pyidi.GUIs.subset_selection.SelectionGUIOld`
     - The 1.3 point-selection window, deprecated and removed in 1.5.
   * - :class:`~pyidi.GUIs.gui.GUI`,
       :class:`~pyidi.GUIs.result_viewer.ResultViewer`,
       :class:`~pyidi.GUIs.result_viewer.Viewer`
     - napari and Qt viewers (require the ``[qt]`` extra).
   * - ``load_analysis``
     - Reload a saved analysis from disk.
   * - :class:`~pyidi.fiducial.Fiducial`
     - Fiducial-marker tracking and rigid-body compensation.
   * - ``postprocessing``
     - Eulerian video magnification and mode-shape magnification.
   * - ``pyIDI``
     - The legacy pre-1.0 class, kept for compatibility only.

Video reader
------------

.. automodule:: pyidi.video_reader
    :members:

Example datasets
----------------

.. automodule:: pyidi.datasets
    :members:

Identification methods
----------------------

IDIMethod base class
^^^^^^^^^^^^^^^^^^^^

Every method inherits from ``IDIMethod``, which provides the shared
configuration handling, multiprocessing, checkpointing and result
persistence.

.. automodule:: pyidi.methods.idi_method
    :members:

Simplified optical flow
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._simplified_optical_flow
    :members:

Simplified optical flow on an image pyramid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._simplified_optical_flow_pyramid
    :members:

Lucas-Kanade
^^^^^^^^^^^^

.. automodule:: pyidi.methods._lucas_kanade
    :members:

Directional Lucas-Kanade
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._directional_lucas_kanade
    :members:

Digital Image Correlation (DIC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._dic
    :members:

Post-processing
---------------

.. _api-eulerian:

Eulerian video magnification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.postprocessing._eulerian_magnification
    :members:

Mode-shape magnification
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.postprocessing._motion_magnification
    :members:

Fiducial markers
----------------

.. automodule:: pyidi.fiducial
    :members:

Point selection geometry
------------------------

Pure-numpy ROI-grid geometry, shared by the napari ``GUI``, ``SelectionGUI``
and ``SelectionGUIOld``. No GUI toolkit is needed to import or use it.

.. warning::

    These functions do not all share one coordinate convention — each
    docstring states which one it uses.

.. automodule:: pyidi.selection_geometry
    :members:

Feature selection pipeline
--------------------------

The mask -> evaluate -> select pipeline behind
:doc:`../quick_start/feature_selection`. Importable without Qt, so the whole
selection can be scripted.

.. automodule:: pyidi.selection.masks
    :members:

.. automodule:: pyidi.selection.evaluate
    :members:

.. automodule:: pyidi.selection.scores
    :members:

.. automodule:: pyidi.selection.select
    :members:

.. automodule:: pyidi.selection.pipeline
    :members:

Graphical interfaces
--------------------

Each class needs the ``[qt]`` extra, but not the same part of it: the selection
windows and the viewers are PyQt6 and pyqtgraph, while the napari ``GUI`` needs
napari and magicgui. A class whose dependencies are missing is replaced by a
stub that imports cleanly and raises ``RuntimeError`` when constructed.

Only the scripting surface is listed here — the windows are driven by hand, and
their event handlers are not part of the public API.

.. autoclass:: pyidi.GUIs.feature_selection.SelectionGUI
    :no-members:
    :members: get_points, points

.. autoclass:: pyidi.GUIs.subset_selection.SelectionGUIOld
    :no-members:
    :members: get_points, points

.. autoclass:: pyidi.GUIs.gui.GUI
    :no-members:

.. autoclass:: pyidi.GUIs.result_viewer.Viewer
    :no-members:

.. autoclass:: pyidi.GUIs.result_viewer.ResultViewer
    :no-members:

Utilities
---------

.. automodule:: pyidi.tools
    :members:

Saved analyses
--------------

.. automodule:: pyidi.load_analysis
    :members:

Legacy pyIDI class
------------------

.. autoclass:: pyidi.pyidi_legacy.pyIDI
    :members:
