# pyidi
Image-based Displacement Identification (IDI) implementation in python.

# BASIC USAGE:
Create an instance:
```
v = pydic.pyIDI('video.cih')
```
Set the points where displacements will be determined. In this step the method of identification is specified.

If `points` is given, these are the ones used:
```
p = np.array([[0, 1], [1, 1], [2, 1]]) # example of points
v.set_points(points=p, method='simplified_optical_flow')
```
If `points` is **not** given, the `get_points` method is triggered:
```
v.set_points(method='simplified_optical_flow')
```
After points are set, displacements can be calculated (using method, set in `set_points`):
```
displacements = v.get_displacements()
```

# DEVELOPER GUIDELINES:
* Add _name_of_method.py with class that inherits after `IDIMethods`
* This class must have methods:
	* `calculate_displacements` with attribute `displacements`
	* `get_points` (static method - sets attribute video.points)
* In `pyIDI` add a new method of identification in `avaliable_methods` dictionary.
