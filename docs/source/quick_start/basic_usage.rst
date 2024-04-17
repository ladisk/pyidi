.. _basic_usage-label:

pyIDI
=====

``pyidi`` is a python package for displacement identification from raw video.

Currently the pyIDI method works with Photron ``.cih`` and ``.cihx`` files, however, ``numpy.ndarray`` can
also be passed as ``cih_file`` argument. If an array is passed, it must have a shape of: ``(n time points, image height, image width)``.

.. note::

    In version 0.30.0, the argument ``cih_file`` was renamed to ``input_file``. This was done
    because we introduced the :py:class:`VideoReader <pyidi.video_reader.VideoReader>` class, 
    which can read multiple file formats such as ``.cih``, ``.cihx``, ``.avi``, ``.mp4``, ``.png``, ``.jpg``, ``numpy.ndarray`` etc.

The basic usage of the package is presented.

Loading the video
-----------------
First create the :py:class:`pyIDI <pyidi.pyidi.pyIDI>` object:

.. code:: python

    video = pyidi.pyIDI('filename.cih')

Setting the points
------------------
Displacements are computed for certain points or certain regions of interest that are represented by a point.

Points must be of shape ``n_points x 2``:

.. code:: python

    points = [[1, 2],
              [1, 5],
              [2, 10]]

where the first column indicates indices along **axis 0**, and the second column indices along **axis 1**.

The points must be passed to ``pyIDI`` object:

.. code:: python

    video.set_points(points=points)

If the points are not known, a :ref:`point-selection` or newer :ref:`napari` can be used to select the points.

Setting the method
------------------
The method for displacement identification must be selected:

.. code:: python

    video.set_method(method='sof') # Simplified optical flow method

After the method is selected, the arguments can be configured. Note that the docstring is now
showing the required arguments for the selected method.

.. code:: python

    video.method.configure(*args, **kwargs)

For more details on the available methods, see the currently implemented :ref:`implemented_disp_id_methods`.

Get displacement
----------------
Finally, displacements can be identified:

.. code:: python

    displacements = video.get_displacements()

Saved analysis
--------------

The settings of the analysis and the identified displacements are saved in a directory next
to the loaded ``cih_file``.

Directory content before the analysis:

- video_to_analyze.cih

Directory content after the analysis:

* video_to_analyze.cih
* video_to_analyze_pyidi_analysis

    * analysis_001
    
        * points.pkl
        * results.pkl
        * settings.txt

Loading saved analysis
----------------------

The saved analysis can be loaded using the ``load_analysis`` function:

.. code:: python

    analysis_path = 'video_to_analyze_pyidi_analysis/analysis_001'

    video_loaded, info_dict = pyidi.load_analysis(analysis_path)

Now we can access the ``video_loaded`` attributes, e.g.:

.. code:: python

    video_loaded.displacements
