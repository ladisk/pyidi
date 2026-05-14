import typing

try:
    import PyQt6

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6 or typing.TYPE_CHECKING:
    from .subset_selection import SelectionGUI
    from .result_viewer import ResultViewer
    from .result_viewer import Viewer
    from .gui import GUI
else:
    class SelectionGUI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SelectionGUI requires the qt extras: pip install pyidi[qt]")

    class ResultViewer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ResultViewer requires the qt extras: pip install pyidi[qt]")

    class GUI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GUI requires the qt extras: pip install pyidi[qt]")

from .selection import SubsetSelection
