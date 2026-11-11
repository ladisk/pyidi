.. _basic_usage-label:

Tutorial
========

pyIDI identifies displacements from a raw video. Every analysis has the same
four steps:

.. code:: python

    from pyidi import VideoReader, LucasKanade

    video = VideoReader('measurement.cih')      # 1. read the recording
    lk = LucasKanade(video)                     # 2. pick a method
    lk.set_points(points)                       # 3. say where to look
    lk.configure(roi_size=(21, 21))             #    and how
    displacements = lk.get_displacements()      # 4. run it

The result is a :class:`numpy.ndarray` of shape ``(n_points, n_frames, 2)``,
in **pixels**, relative to the reference frame. The last axis is
``(row, column)`` — see :ref:`point-conventions`.

The rest of this page walks through each step.

1. Loading the video
--------------------

.. code:: python

    from pyidi import VideoReader

    video = VideoReader('filename.cih')

``VideoReader`` accepts Photron ``.cih``/``.cihx``, Phantom ``.cine``,
Pharsighted ``.SLOW``, image sequences (PNG, TIFF, BMP, JPEG, GIF), ordinary
video files (MP4, AVI, MKV, MOV and others), and :class:`numpy.ndarray`
objects of shape ``(n_time_points, image_height, image_width)``.

Check that the frame rate is what you think it is — a wrong ``fps`` silently
puts every identified frequency in the wrong place:

.. code:: python

    print(video.N, video.image_height, video.image_width, video.fps)

    video.configure(fps=10000)    # if the file does not carry it, or carries it wrong

See :doc:`video_reader` for the details of each format.

2. Choosing a method
--------------------

The video object is passed to one of the
:class:`IDIMethod <pyidi.methods.idi_method.IDIMethod>` subclasses:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Use it when
   * - :class:`SimplifiedOpticalFlow <pyidi.methods._simplified_optical_flow.SimplifiedOpticalFlow>`
     - You want a fast first look at a whole field of points, with small
       displacements (well below a pixel).
   * - :class:`SimplifiedOpticalFlowPyramid <pyidi.methods._simplified_optical_flow_pyramid.SimplifiedOpticalFlowPyramid>`
     - You want the speed of simplified optical flow, but the displacements
       are larger than half a pixel: a coarse-to-fine image pyramid extends
       the range.
   * - :class:`LucasKanade <pyidi.methods._lucas_kanade.LucasKanade>`
     - This is the default choice: iterative sub-pixel translation of a
       subset, accurate over larger displacements.
   * - :class:`DirectionalLucasKanade <pyidi.methods._directional_lucas_kanade.DirectionalLucasKanade>`
     - The motion is (or is assumed to be) along one known direction per
       point — this buys accuracy on features that are only well defined
       across that direction, such as a single edge.
   * - :class:`DIC <pyidi.methods._dic.DIC>`
     - You need more than translation: strain and in-plane rotation of each
       subset.

.. code:: python

    from pyidi import SimplifiedOpticalFlow

    sof = SimplifiedOpticalFlow(video)

:doc:`disp_id_methods` covers what each method does and how to configure it.

3. Setting the points
---------------------

.. _point-conventions:

Point conventions
^^^^^^^^^^^^^^^^^

Displacements are computed at points, each of which stands for the subset
(region of interest) drawn around it. Points are an array of shape
``(n_points, 2)`` in **image coordinates: row first, column second**.

.. code:: python

    points = [[1, 2],       # row 1,  column 2
              [1, 5],       # row 1,  column 5
              [2, 10]]      # row 2,  column 10

The first column is the **row** (``y``, axis 0) and the second is the
**column** (``x``, axis 1). The same convention applies to the result array
and to ``roi_size=(vertical, horizontal)``.

.. code:: python

    sof.set_points(points=points)

``set_points()`` validates its input rather than accepting anything:

* empty input, input that is not 2-D, or a wrong number of columns raises
  ``ValueError``;
* coordinates outside the image raise ``ValueError``, listing the offending
  points;
* sub-pixel (float) coordinates are rounded to the nearest pixel, with a
  warning saying how many were changed.

Selecting points interactively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not already know where the points go, pick them in the
:ref:`SelectionGUI <point-selection>` or the :ref:`napari viewer <napari>`.
The GUI instance can be passed straight to ``set_points()``:

.. code:: python

    from pyidi import SelectionGUI

    gui = SelectionGUI(video, subset_size=21)
    sof.set_points(gui)              # or: sof.set_points(gui.points)

4. Configuring
--------------

Every method exposes its parameters through ``configure()``:

.. code:: python

    lk.configure(roi_size=(21, 21), max_nfev=20, tol=1e-8, processes=1)

Each argument is stored as an attribute of the same name, which is what makes
an analysis reproducible: the settings are written to ``settings.json`` next
to the results, and can be reloaded later.

.. note::

    Most methods can spread the work over several processes. With
    ``processes`` greater than one, the points are split into groups and each
    group is handled by a separate process.

    In a script (as opposed to a Jupyter notebook), multiprocessing code must
    be guarded:

    .. code:: python

        if __name__ == '__main__':
            displacements = lk.get_displacements()

5. Getting the displacements
----------------------------

.. code:: python

    displacements = sof.get_displacements()

``get_displacements()`` also accepts configuration keyword arguments, so the
configure step can be folded in:

.. code:: python

    displacements = lk.get_displacements(roi_size=(21, 21), processes=4)

The result has shape ``(n_points, n_frames, 2)``. To get the displacement
history of point ``i`` in the row direction:

.. code:: python

    u_row = displacements[i, :, 0]
    u_col = displacements[i, :, 1]

.. warning::

    Results may contain ``NaN``. A point that cannot be tracked no longer
    aborts the analysis — it goes to ``NaN`` from the frame at which it was
    lost, and the rest of the points are computed normally. Use ``np.nanmax``
    and friends, and check ``lk.failed_points``. See :ref:`failed-points`.

A complete example
------------------

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pyidi import VideoReader, LucasKanade

    video = VideoReader('data/data_synthetic.cih')

    lk = LucasKanade(video)
    lk.set_points(points=np.array([[31, 35], [31, 215]]))
    lk.configure(roi_size=(11, 11), int_order=3)

    displacements = lk.get_displacements()

    t = np.arange(video.N) / video.fps
    plt.plot(t, displacements[0, :, 0], label='point 0, row')
    plt.plot(t, displacements[1, :, 0], label='point 1, row')
    plt.xlabel('time [s]')
    plt.ylabel('displacement [pixel]')
    plt.legend()
    plt.show()

Next
----

* :doc:`results` — where the results are saved, how to reload them, and how to
  view them.
* :doc:`disp_id_methods` — the methods in detail.
* :doc:`../postprocessing/eulerian_magnification` — see the motion before you
  measure it.
