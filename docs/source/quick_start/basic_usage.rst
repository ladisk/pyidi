.. _basic_usage-label:

Basic usage
===========

Loading the video
-----------------
First create the ``pyIDI`` object:
::
    video = pyidi.pyIDI('filename.cih')

Setting the points
------------------
Displacements are computed for certain points or certain regions of interest that are represented by a point.

Points must be of shape ``n_ponits x 2``:
::
    points = [[1, 2],
              [1, 5],
              [2, 10]]

where the first column indicates indices along **axis 0**, and the second column indices along **axis 1**.

The points must be passed to ``pyIDI`` object:
::
    video.set_points(points=points)

If points are not known, a helper tool can be used to select them. This can be done by first setting a method for displacement identification:
::
    video.set_method(method='lk') # The Lucas-Kanade algorithm for translations

Now the ``set_points()`` can be called and a selection tool is triggered.

Another option is using ``tools`` module in ``pyIDI`` module:
::
    points_obj = pyidi.tools.ManualROI(video, roi_size=(11, 11))

or another tool:
::
    points_obj = pyidi.tools.RegularROIGrid(video, roi_size=(11, 11), noverlap=0)

Points can then be accessed:
::
    points = points_obj.points

Setting the method
------------------
If the method was not yet set during points selection process, the method must be selected:
::
    video.set_method(method='sof') # Simplified optical flow method

The method arguments can be configured:
::
    video.method.configure(*args, **kwargs)

Get displacement
----------------
Finally, displacements can be identified:
::
    displacements = video.get_displacements()


Multiprocessing
---------------
In the case of ``lk`` method (Lucas-Kanade translation), the parallel computation of displacements is faster. To access the multiprocessing option, simply input
the number of processes you wish to run. The points will be automatically equally split:
::
    displacements = video.get_displacements(processes=4)

Note that the ``video`` object must already have set method and attributes.