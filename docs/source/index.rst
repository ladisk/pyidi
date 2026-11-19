:og:description: Image-based Displacement Identification from high-speed video in Python.

pyIDI
=====

**Image-based Displacement Identification (IDI)** from high-speed video, in Python.

pyIDI reads a recording, tracks the points you select, and returns their
sub-pixel displacement history — an array of shape
``(n_points, n_frames, 2)`` you can feed straight into a modal analysis.

.. image:: quick_start/music_box.gif
    :alt: The teeth of a music-box comb ringing, drawn from the displacements pyIDI
        identified in a 7500 fps video, next to the tip displacement and note of each
        tooth.
    :width: 800

.. code:: python

    from pyidi import VideoReader, LucasKanade

    video = VideoReader('measurement.cih')

    lk = LucasKanade(video)
    lk.set_points(points=[[150, 200], [150, 260], [150, 320]])
    lk.configure(roi_size=(21, 21))

    displacements = lk.get_displacements()   # (n_points, n_frames, 2), in pixels

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`rocket` Getting started
        :link: quick_start/basic_usage
        :link-type: doc

        Install pyIDI, load a video, select points and run your first
        identification.

    .. grid-item-card:: :octicon:`graph` Displacement methods
        :link: quick_start/disp_id_methods
        :link-type: doc

        Simplified Optical Flow, Lucas-Kanade, Directional DIC and full-field
        DIC — what each one is for and how to configure it.

    .. grid-item-card:: :octicon:`eye` Selecting points
        :link: quick_start/feature_selection
        :link-type: doc

        The ``SelectionGUI``: draw a region, score the whole image, and let
        the selection find the best-separated features inside it.

    .. grid-item-card:: :octicon:`beaker` Post-processing
        :link: postprocessing/eulerian_magnification
        :link-type: doc

        Eulerian video magnification for pre-test motion visualization, and
        mode-shape magnification of identified displacements.

    .. grid-item-card:: :octicon:`code` API reference
        :link: code/modules
        :link-type: doc

        Every public class and function, generated from the source.

    .. grid-item-card:: :octicon:`arrow-right` Upgrading
        :link: migration
        :link-type: doc

        Coming from 1.3.3 or from the pre-1.0 ``pyIDI`` class? Start here.

What pyIDI does
---------------

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: Reads what your camera wrote

        Photron ``.cih``/``.cihx``, Phantom ``.cine``, Pharsighted ``.SLOW``,
        image sequences, ordinary video files, and plain
        :class:`numpy.ndarray` stacks — behind one
        :class:`~pyidi.video_reader.VideoReader` interface.

    .. grid-item-card:: Tracks to sub-pixel accuracy

        Four identification methods, from a fast whole-field gradient
        estimate to an iterative full-field DIC solve, sharing one
        configuration, checkpointing and result-saving framework.

    .. grid-item-card:: Scales to long recordings

        The Lucas-Kanade inner loop is compiled with ``numba`` and runs
        across threads or processes, with crash-resistant checkpointing for
        long analyses.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   installation
   quick_start/basic_usage
   quick_start/video_reader
   datasets
   quick_start/feature_selection
   quick_start/napari
   quick_start/points_selection

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Identification

   quick_start/disp_id_methods
   quick_start/results

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Post-processing

   postprocessing/eulerian_magnification
   mode_shape_magnification
   fiducial_marker

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   code/modules
   migration
   changelog
   contributing/documenting

Citing pyIDI
------------

If you use pyIDI in your research, please cite the article behind the method
you used:

    Masmeijer, T., Habtour, E., Zaletelj, K., & Slavič, J. (2024).
    **Directional DIC method with automatic feature selection**.
    *Mechanical Systems and Signal Processing*, 224.
    https://doi.org/10.1016/j.ymssp.2024.112080

    Čufar, K., Slavič, J., & Boltežar, M. (2024).
    **Mode-shape magnification in high-speed camera measurements**.
    *Mechanical Systems and Signal Processing*, 213, 111336.
    https://doi.org/10.1016/J.YMSSP.2024.111336

    Zaletelj, K., Gorjup, D., Slavič, J., & Boltežar, M. (2023).
    **Multi-level curvature-based parametrization and model updating using a
    3D full-field response**. *Mechanical Systems and Signal Processing*,
    187, 109927. https://doi.org/10.1016/j.ymssp.2022.109927

    Zaletelj, K., Slavič, J., & Boltežar, M. (2022).
    **Full-field DIC-based model updating for localized parameter
    identification**. *Mechanical Systems and Signal Processing*, 164.
    https://doi.org/10.1016/j.ymssp.2021.108287

    Gorjup, D., Slavič, J., & Boltežar, M. (2019).
    **Frequency domain triangulation for full-field 3D operating-deflection-shape
    identification**. *Mechanical Systems and Signal Processing*, 133.
    https://doi.org/10.1016/j.ymssp.2019.106287

The package itself is archived on Zenodo:
https://doi.org/10.5281/zenodo.4017153

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
