.. _point-selection:

Point selection UI
==================

A convinient UI is available to make the point selection easier.

To use the UI, a ``VideoReader`` object must first be created:

.. code:: python

    from pyidi import VideoReader, SubsetSelection

    video = VideoReader(input_file)

where ``input_file`` can be a Photron ``.cih``/``.cihx`` path, an image, a
video file, a numpy array, or a ``.SLOW`` file.

A ``SubsetSelection`` object can then be created:

.. code:: python

    Points = SubsetSelection(video, roi_size=(21, 21), noverlap=0)

where ``roi_size`` is the size of a single Region-Of-Interest/subset in ``y`` and
``x`` direction respsectivly. The ``noverlap`` argument prescribes the overlap of the
neighbouring ROIs. The density of the grid can be adjusted using ``noverlap``.

The UI enables multiple modes of point selection. Currently, the following are
supported:

- ``ROI grid``: A regular grid of ROIs is created based on the selected polygon.
- ``Deselect ROI polygon``: After defining the polygon and getting the points, this method can
  be used to define a polygon within which the points are not selected.
- ``Only polygon``: Same as ROI grid but the points are not computed. Only polygon points
  are available.
- ``Manual ROI select``: Manually select the ROIs at desired locations.

Once the selection in the UI is complete, the points can be retrieved:

.. code:: python

  points = Points.points


.. image:: selection.gif
