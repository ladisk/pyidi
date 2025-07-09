try:
    import pyqt6

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .main_window import SelectionGUI
else:
    class DisplacementGUI:
        def __init__(self):
            pass
            
        def show_displacement(self, data):
            raise RuntimeError("GUI requires PyQt6: pip install pyidi[gui]")