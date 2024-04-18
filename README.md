[![Documentation Status](https://readthedocs.org/projects/pyidi/badge/?version=latest)](https://pyidi.readthedocs.io/en/latest/?badge=latest)
![example workflow](https://github.com/ladisk/pyidi/actions/workflows/python_package_testing.yaml/badge.svg)

# pyidi
Image-based Displacement Identification (IDI) implementation in python.

See the [documentation](https://pyidi.readthedocs.io/en/latest/index.html) for `pyIDI`.

### Use Napari UI for quick displacement identification:
<img src="docs/source/quick_start/gifs/napari_full_sof.gif" width="800" />


# BASIC USAGE:
Create an instance:
```
video = pyidi.pyIDI(input_file='video.cih')
```

The `pyIDI` method works with various formats: `.cih`, `.cihx`, `.png`, `.avi` etc. Additionally, it can also work with `numpy.ndarray` as input.
If an array is passed, it must have a shape of: ``(n time points, image height, image width)``.

Set the points where displacements will be determined:
```
p = np.array([[0, 1], [1, 1], [2, 1]]) # example of points
video.set_points(points=p)
```
Or use point selection UI to set individual points or grid inside selected area. For more information about UI see [documentation](https://pyidi.readthedocs.io/en/quick_start/napari.html). Launch viewer with:

```
video.gui()
```

The method of identification has to be specified:
```
video.set_method(method='sof', **method_kwargs)
```
After points are set, displacements can be calculated (using method, set in `set_method`):
```
displacements = video.get_displacements()
```
Multiprocessing can also be used by passing the `processes` argument:
```
displacements = video.get_displacements(processes=4)
```

# DEVELOPER GUIDELINES:
* Add _name_of_method.py with class that inherits after `IDIMethods`
* This class must have methods:
	* `calculate_displacements` with attribute `displacements`
	* `get_points` (static method - sets attribute video.points)
* In `pyIDI` add a new method of identification in `avaliable_methods` dictionary.

# Citing
If you are using the `pyIDI` package for your research, consider citing our articles:
- Čufar, K., Slavič, J., & Boltežar, M. (2024). **Mode-shape magnification in high-speed camera measurements**. Mechanical Systems and Signal Processing, 213, 111336. https://doi.org/10.1016/J.YMSSP.2024.111336
- Zaletelj, K., Gorjup, D., Slavič, J., & Boltežar, M. (2023). **Multi-level curvature-based parametrization and model updating using a 3D full-field response**. Mechanical Systems and Signal Processing, 187, 109927. https://doi.org/10.1016/j.ymssp.2022.109927
- Zaletelj, K., Slavič, J., & Boltežar, M. (2022). **Full-field DIC-based model updating for localized parameter identification**. Mechanical Systems and Signal Processing, 164. https://doi.org/10.1016/j.ymssp.2021.108287
- Gorjup, D., Slavič, J., & Boltežar, M. (2019). **Frequency domain triangulation for full-field 3D operating-deflection-shape identification**. Mechanical Systems and Signal Processing, 133. https://doi.org/10.1016/j.ymssp.2019.106287

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4017153.svg)](https://doi.org/10.5281/zenodo.4017153)
[![Build Status](https://travis-ci.com/ladisk/pyidi.svg?branch=master)](https://travis-ci.com/ladisk/pyidi)

