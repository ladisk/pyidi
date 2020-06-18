# pyidi
Image-based Displacement Identification (IDI) implementation in python.

The documentation for this repository is accessible [here](https://pyidi.readthedocs.io/en/latest/index.html).

# BASIC USAGE:
![Showcase GIF](usage_gif.gif)

Create an instance:
```
v = pyidi.pyIDI('video.cih')
```
Set the points where displacements will be determined. In this step the method of identification is specified.

If `points` is given, these are the ones used:
```
p = np.array([[0, 1], [1, 1], [2, 1]]) # example of points
v.set_points(points=p)
```
The method of identification has to be specified:
```
v.set_method(method='simplified_optical_flow', **method_kwargs)
```
If the `points` argument is not given and the `method` is supplied to `set_points`, the `get_points` method is triggered:
```
v.set_points(method='simplified_optical_flow', **set_points_kwargs)
v.set_method(method='simplified_optical_flow', **method_kwargs)
```
After points are set, displacements can be calculated (using method, set in `set_method`):
```
displacements = v.get_displacements()
```

# DEVELOPER GUIDELINES:
* Add _name_of_method.py with class that inherits after `IDIMethods`
* This class must have methods:
	* `calculate_displacements` with attribute `displacements`
	* `get_points` (static method - sets attribute video.points)
* In `pyIDI` add a new method of identification in `avaliable_methods` dictionary.

[![Build Status](https://travis-ci.com/ladisk/pyidi.svg?branch=master)](https://travis-ci.com/ladisk/pyidi)