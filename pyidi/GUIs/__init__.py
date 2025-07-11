try:
    import PyQt6

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .subset_selection import SelectionGUI
    from .result_viewer import ResultViewer
else:
    class SelectionGUI:
        def __init__(self, video):
            pass
            
        def show_displacement(self, data):
            raise RuntimeError("SelectionGUI requires PyQt6: pip install pyidi[qt]")
    
    class ResultViewer:
        def __init__(self):
            pass
            
        def show_displacement(self, data):
            raise RuntimeError("ResultViewer requires PyQt6: pip install pyidi[qt]")
        
from .selection import SubsetSelection
from .gui import GUI