.. _basic_usage-label:

pyIDI
=====

``pyidi`` is a python package for displacement identification from raw video.

The basic usage of the package is presented.

Loading the video
-----------------
First create the ``pyIDI`` object:

.. code:: python

    video = pyidi.pyIDI('filename.cih')

Setting the points
------------------
Displacements are computed for certain points or certain regions of interest that are represented by a point.

Points must be of shape ``n_ponits x 2``:

.. code:: python

    points = [[1, 2],
              [1, 5],
              [2, 10]]

where the first column indicates indices along **axis 0**, and the second column indices along **axis 1**.

The points must be passed to ``pyIDI`` object:

.. code:: python

    video.set_points(points=points)

If the points are not known, a :ref:`point-selection` can be used to select the points.

Setting the method
------------------
The method for displacement identification must be selected:

.. code:: python

    video.set_method(method='sof') # Simplified optical flow method

After the method is selected, the arguments can be configured. Note that the docstring is now
showing the required arguments for the selected method.

.. code:: python

    video.method.configure(*args, **kwargs)

Get displacement
----------------
Finally, displacements can be identified:

.. code:: python

    displacements = video.get_displacements()
