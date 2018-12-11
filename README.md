# pyidc

Image-based Displacement Calculation (IDC) implementation in python.


# STRUCTURE OF THE CLASSES:
- pyIDC
    - load_video
	- _images
	- _info
	- get_displacements(points, method)
	    - Call appropriate method (DigitalImageCorrelation)

- IDCMethods
	- Common functions

- DigitalImageCorrelation(IDCMethods)
	- Calculation with DIC
	- __init__ returns displacements

- SimplifiedOpticalFlow(IDCMethods)
	- Calculation with SOF
	- __init__ returns displacements

# FILE STRUCTURE:
- pyidc
	- pyidc
	    - __init__.py
	    - pyidc.py
	    - idc_tools.py
	    - idc_methods.py
	    - _digital_image_correlation.py
	    - _simplified_optical_flow.py
	    - ⋮
	- setup.py
	- README.rst
	- LICENSE.txt
	- ⋮

# BASIC USAGE:
```
v = pydic.pydic(‘video.cih’)
displacements = v.get_displacements(points, method=’DIC’)
```
