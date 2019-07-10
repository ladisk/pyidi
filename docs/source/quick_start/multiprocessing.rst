.. _multiprocessing-label:

Multiprocessing
===============
In the case of ``lk`` method (Lucas-Kanade translation), the parallel computation of displacements is faster. A ``multi()`` function from ``tools`` module can be used
to apply multiprocessing:
::
    displacements = pyidi.tools.multi(video, points, processes=2)

Note that the ``video`` object must already have set method and attributes.