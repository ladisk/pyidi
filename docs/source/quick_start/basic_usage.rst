.. _basic_usage-label:

Tutorial
========

``pyidi`` is a python package for displacement identification from raw video.

Currently the pyIDI method works with Photron ``.cih`` and ``.cihx`` files, however, ``numpy.ndarray`` can
also be passed as ``cih_file`` argument. If an array is passed, it must have a shape of: ``(n time points, image height, image width)``.

Loading the video
-----------------
First create the :py:class:`VideoReader <pyidi.video_reader.VideoReader>` object:

.. code:: python

    from pyidi import VideoReader

    video = VideoReader('filename.cih')

Setting the method
------------------
The video object must be passed to the :py:class:`IDIMethod <pyidi.methods.idi_method.IDIMethod>` class.
Available methods are: 

* :py:class:`SimplifiedOpticalFlow <pyidi.methods._simplified_optical_flow.SimplifiedOpticalFlow>`

* :py:class:`LucasKanade <pyidi.methods._lucas_kanade.LucasKanade>`

* :py:class:`DirectionalLucasKanade <pyidi.methods._directional_lucas_kanade.DirectionalLucasKanade>`

To use the Simplified Optical Flow method, the object must be instantiated:

.. code:: python

    from pyidi import SimplifiedOpticalFlow
    
    sof = SimplifiedOpticalFlow(video)

After the method object is instantiated, the points can be set and the arguments can be configured.

For more details on the available methods, see the currently implemented :ref:`implemented_disp_id_methods`.

Setting the points
------------------
Displacements are computed for certain points or certain regions of interest that are represented by a point.

Points must be of shape ``n_points x 2``:

.. code:: python

    points = [[1, 2],
              [1, 5],
              [2, 10]]

where the first column indicates indices along **axis 0**, and the second column indices along **axis 1**.

The points must be passed to ``method`` object:

.. code:: python

    sof.set_points(points=points)

If the points are not known, a :ref:`point-selection` or newer :ref:`napari` can be used to select the points.

Configuring the method
----------------------
The method can be configured using:

.. code:: python
    
    sof.configure(...)


Get displacement
----------------
Finally, displacements can be identified:

.. code:: python

    displacements = sof.get_displacements()

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
