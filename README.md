# pyidi
Image-based Displacement Identification (IDI) implementation in python.

See the [documentation](https://pyidi.readthedocs.io/en/latest/index.html) for `pyIDI`.

# BASIC USAGE:
Create an instance:
```
v = pyidi.pyIDI(cih_file='video.cih')
```
Currently the pyIDI method works with Photron ``.cih`` and ``.cihx`` files, however, ``numpy.ndarray`` can
also be passed as ``cih_file`` argument. If an array is passed, it must have a shape of: ``(n time points, image height, image width)``.

Set the points where displacements will be determined:
```
p = np.array([[0, 1], [1, 1], [2, 1]]) # example of points
v.set_points(points=p)
```
Or use point selection UI to set individual points or grid inside selected area. for more information about UI see [documentation](https://pyidi.readthedocs.io/en/quick_start/napari.html) Launch viewer with:

```
v()
```

The method of identification has to be specified:
```
v.set_method(method='sof', **method_kwargs)
```
After points are set, displacements can be calculated (using method, set in `set_method`):
```
displacements = v.get_displacements()
```
Multiprocessing can also be used by passing the `processes` argument:
```
displacements = v.get_displacements(processes=4)
```

# DEVELOPER GUIDELINES:
* Add _name_of_method.py with class that inherits after `IDIMethods`
* This class must have methods:
	* `calculate_displacements` with attribute `displacements`
	* `get_points` (static method - sets attribute video.points)
* In `pyIDI` add a new method of identification in `avaliable_methods` dictionary.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4017153.svg)](https://doi.org/10.5281/zenodo.4017153)
[![Build Status](https://travis-ci.com/ladisk/pyidi.svg?branch=master)](https://travis-ci.com/ladisk/pyidi)

