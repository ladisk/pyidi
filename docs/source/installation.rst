.. _installation-label:

Installation
============

Requirements
------------

pyIDI requires **Python >= 3.10**.

Basic install
-------------

.. code:: bash

    pip install pyidi

Optional extras
---------------

``[qt]``
^^^^^^^^

Installs napari, PyQt6, pyqtgraph, and magicgui — required for any GUI usage
(``SelectionGUI``, ``ResultViewer``, ``GUI``, ``SubsetSelection``).
Without this extra, importing those classes raises a ``RuntimeError``.

.. code:: bash

    pip install pyidi[qt]

``[dev]``
^^^^^^^^^

Installs development and testing dependencies: sphinx, pytest, ipykernel,
ipywidgets, and related packages.

.. code:: bash

    pip install pyidi[dev]

Combining extras
^^^^^^^^^^^^^^^^

.. code:: bash

    pip install pyidi[qt,dev]

ArUco / fiducial detection
--------------------------

pyIDI relies on ``opencv-contrib-python`` (not the base ``opencv-python``
package) for ArUco marker detection. This dependency was updated in a recent
release. If you have ``opencv-python`` already installed, uninstall it first
to avoid conflicts:

.. code:: bash

    pip uninstall opencv-python
    pip install pyidi

Editable / development install
-------------------------------

To install in editable mode with all optional dependencies:

.. code:: bash

    pip install -e ".[dev,qt]"
