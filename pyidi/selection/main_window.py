import numpy as np
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem
import pyqtgraph as pg
from matplotlib.path import Path
# import pyidi  # Assuming pyidi is a custom module for video handling

class BrushViewBox(pg.ViewBox):
    def __init__(self, parent_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(self.PanMode)
        self.parent_gui = parent_gui

    def mouseClickEvent(self, ev):
        if self.parent_gui.mode == "manual" and self.parent_gui.method_buttons["Brush"].isChecked():
            if self.parent_gui.ctrl_held:
                ev.accept()
                self.parent_gui.handle_brush_start(ev)
            else:
                ev.ignore()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.parent_gui.mode == "manual" and self.parent_gui.method_buttons["Brush"].isChecked():
            if self.parent_gui.ctrl_held:
                ev.accept()
                if ev.isStart():
                    self.parent_gui._painting = True
                    self.parent_gui._brush_path = []
                    self.parent_gui.handle_brush_start(ev)
                elif ev.isFinish():
                    self.parent_gui._painting = False
                    self.parent_gui.handle_brush_end(ev)
                else:
                    self.parent_gui.handle_brush_move(ev)
                return
        # fallback: pan
        super().mouseDragEvent(ev, axis)

class SelectionGUI(QtWidgets.QMainWindow):
    def __init__(self, video):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        super().__init__()

        self.setWindowTitle("ROI Selection Tool")
        self.resize(1200, 800)

        self._paint_mask = None  # Same shape as the image
        self._paint_radius = 10  # pixels
        self.ctrl_held = False
        self.installEventFilter(self)

        self.selected_points = []
        self.manual_points = []
        self.candidate_points = []
        self.drawing_polygons = [{'points': [], 'roi_points': []}]
        self.active_polygon_index = 0
        self.grid_polygons = [{'points': [], 'roi_points': []}]
        self.active_grid_index = 0

        # Central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        # Top-level layout for the central widget
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Toolbar (fixed height)
        self.mode_toolbar = QtWidgets.QWidget()
        self.mode_toolbar_layout = QtWidgets.QHBoxLayout(self.mode_toolbar)
        self.mode_toolbar_layout.setContentsMargins(5, 4, 5, 4)
        self.mode_toolbar_layout.setSpacing(6)

        self.manual_mode_button = QtWidgets.QPushButton("Manual")
        self.automatic_mode_button = QtWidgets.QPushButton("Automatic")
        for btn in [self.manual_mode_button, self.automatic_mode_button]:
            btn.setCheckable(True)
            btn.setMinimumWidth(100)
            self.mode_toolbar_layout.addWidget(btn)

        self.manual_mode_button.setChecked(True)
        self.manual_mode_button.clicked.connect(lambda: self.switch_mode("manual"))
        self.automatic_mode_button.clicked.connect(lambda: self.switch_mode("automatic"))

        self.mode_toolbar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.mode_toolbar.setMaximumHeight(self.manual_mode_button.sizeHint().height() + 12)

        self.main_layout.addWidget(self.mode_toolbar)

        # Add splitter directly and stretch it
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter, stretch=1)

        # Graphics layout for image and points display
        self.ui_graphics()
        
        self.ui_right_menu()

        # Style
        self.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QPushButton {
                background-color: #444;
                color: white;
                padding: 6px 12px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0078d7;
                border: 1px solid #005bb5;
            }
        """)

        # Connect selection change handler
        self.button_group.idClicked.connect(self.method_selected)

        # Connect mouse click
        self.pg_widget.scene().sigMouseClicked.connect(self.on_mouse_click)

        # Set the initial image
        self.image_item.setImage(video)

        # Ensure method-specific widgets are visible on startup
        self.method_selected(self.button_group.checkedId())
        self.auto_method_selected(0)

        # Set the initial mode
        self.switch_mode("manual")  # Default to manual mode

        # Start the GUI
        self.show()
        if app is not None:
            app.exec()

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Control:
                self.ctrl_held = True
        elif event.type() == QtCore.QEvent.Type.KeyRelease:
            if event.key() == QtCore.Qt.Key.Key_Control:
                self.ctrl_held = False
        return super().eventFilter(source, event)

    def create_help_button(self, tooltip_text: str) -> QtWidgets.QToolButton:
        """Create a small '?' help button with a tooltip."""
        button = QtWidgets.QToolButton()
        button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxQuestion))
        button.setToolTip(tooltip_text)
        button.setCursor(QtCore.Qt.CursorShape.WhatsThisCursor)
        button.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                padding: 0px;
            }
            QToolButton:hover {
                color: #0078d7;
            }
        """)
        button.setFixedSize(20, 20)
        return button

    def ui_graphics(self):
        # Image viewer
        self.pg_widget = GraphicsLayoutWidget()
        self.view = BrushViewBox(parent_gui=self, lockAspect=True)
        self.pg_widget.addItem(self.view)

        
        self.image_item = ImageItem()
        self.polygon_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.polygon_points_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=6)
        self.scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 200), size=8)
        self.roi_overlay = ImageItem()

        self.candidate_scatter = ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 255, 0, 200),
            size=6
        )

        self.view.addItem(self.image_item)
        self.view.addItem(self.polygon_line)
        self.view.addItem(self.polygon_points_scatter)
        self.view.addItem(self.roi_overlay)  # Add scatter for showing square points
        self.view.addItem(self.scatter)  # Add scatter for showing points
        self.view.addItem(self.candidate_scatter)

        self.splitter.addWidget(self.pg_widget)

    def ui_right_menu(self):
        # The right-side menu
        self.method_widget = QtWidgets.QWidget()
        self.stack = QtWidgets.QStackedLayout(self.method_widget)

        self.manual_widget = QtWidgets.QWidget()
        self.manual_layout = QtWidgets.QVBoxLayout(self.manual_widget)
        self.stack.addWidget(self.manual_widget)

        self.automatic_widget = QtWidgets.QWidget()
        self.automatic_layout = QtWidgets.QVBoxLayout(self.automatic_widget)
        self.stack.addWidget(self.automatic_widget)

        self.ui_manual_right_menu() # The manual right menu

        self.ui_auto_right_menu() # The automatic right menu

        # Set the layout and add to splitter
        self.splitter.addWidget(self.method_widget)
        self.splitter.setStretchFactor(0, 5)  # Image area grows more
        self.splitter.setStretchFactor(1, 0)  # Menu fixed by content

        # Set initial width for right panel
        self.method_widget.setMinimumWidth(150)
        self.method_widget.setMaximumWidth(600)
        self.splitter.setSizes([1000, 220])  # Initial left/right width

        self.automatic_layout.addStretch(1)

    def ui_manual_right_menu(self):
        # Number of selected subsets
        self.points_label = QtWidgets.QLabel("Selected subsets: 0")
        font = self.points_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.points_label.setFont(font)
        self.manual_layout.addWidget(self.points_label)

        # Method selection buttons
        self.button_group = QtWidgets.QButtonGroup(self.method_widget)
        self.button_group.setExclusive(True)

        self.method_buttons = {}
        method_names = [
            "Grid",
            "Manual",
            "Along the line",
            "Brush",
            "Remove point",
        ]
        for i, name in enumerate(method_names):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            if i == 0:
                button.setChecked(True)  # Default selection
            self.button_group.addButton(button, i)
            self.manual_layout.addWidget(button)
            self.method_buttons[name] = button

        # Subset size input
        self.manual_layout.addSpacing(20)
        self.manual_layout.addWidget(QtWidgets.QLabel("Subset size:"))

        self.subset_size_spinbox = QtWidgets.QSpinBox()
        self.subset_size_spinbox.setRange(1, 1000)
        self.subset_size_spinbox.setValue(11)
        self.subset_size_spinbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.subset_size_spinbox.setSingleStep(2)
        self.subset_size_spinbox.setMinimum(1)
        self.subset_size_spinbox.setMaximum(999)
        self.subset_size_spinbox.setWrapping(False)
        self.subset_size_spinbox.valueChanged.connect(self.update_selected_points)
        self.manual_layout.addWidget(self.subset_size_spinbox)

        # Show ROI rectangles
        self.show_roi_checkbox = QtWidgets.QCheckBox("Show subsets")
        self.show_roi_checkbox.setChecked(True)
        self.show_roi_checkbox.stateChanged.connect(self.update_selected_points)
        self.manual_layout.addWidget(self.show_roi_checkbox)

        # Clear button
        self.manual_layout.addSpacing(20)
        self.clear_button = QtWidgets.QPushButton("Clear selections")
        self.clear_button.clicked.connect(self.clear_selection)
        self.manual_layout.addWidget(self.clear_button)

        # Separator line
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.manual_layout.addWidget(separator)
        self.manual_layout.addSpacing(20)

        # Distance between subsets (only visible for Grid and Along the line)
        self.distance_label = QtWidgets.QLabel("Distance between subsets:")
        self.distance_label.setVisible(False)  # Hidden by default
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.distance_slider.setRange(-50, 50)
        self.distance_slider.setSingleStep(1)
        self.distance_slider.setValue(0)
        self.distance_slider.setVisible(False)
        self.manual_layout.addWidget(self.distance_slider)
        self.manual_layout.addWidget(self.distance_label)

        def update_label_and_recompute(val):
            self.distance_label.setText(f"Distance between subsets: {str(val)}")
            self.recompute_roi_points()
        self.distance_slider.valueChanged.connect(update_label_and_recompute)

        # self.manual_layout.addWidget(self.distance_label)
        self.manual_layout.addWidget(self.distance_slider)

        # Start new line (only visible in "Along the line" mode)
        self.start_new_line_button = QtWidgets.QPushButton("Start new line")
        self.start_new_line_button.clicked.connect(self.start_new_line)
        self.start_new_line_button.setVisible(False)  # Hidden by default
        self.manual_layout.addWidget(self.start_new_line_button)

        self.manual_layout.addStretch(1)

        # Polygon manager (visible only for "Along the line")
        self.polygon_list = QtWidgets.QListWidget()
        self.polygon_list.setVisible(False)
        self.polygon_list.currentRowChanged.connect(self.on_polygon_selected)
        self.manual_layout.addWidget(self.polygon_list)

        self.delete_polygon_button = QtWidgets.QPushButton("Delete selected polygon")
        self.delete_polygon_button.clicked.connect(self.delete_selected_polygon)
        self.delete_polygon_button.setVisible(False)
        self.manual_layout.addWidget(self.delete_polygon_button)

        # Grid polygon manager
        self.grid_list = QtWidgets.QListWidget()
        self.grid_list.setVisible(False)
        self.grid_list.currentRowChanged.connect(self.on_grid_selected)
        self.manual_layout.addWidget(self.grid_list)

        self.delete_grid_button = QtWidgets.QPushButton("Delete selected grid")
        self.delete_grid_button.clicked.connect(self.delete_selected_grid)
        self.delete_grid_button.setVisible(False)
        self.manual_layout.addWidget(self.delete_grid_button)

    def ui_auto_right_menu(self):
        # Title and method selector
        self.automatic_layout.addWidget(QtWidgets.QLabel("Automatic method:"))

        self.auto_method_group = QtWidgets.QButtonGroup(self.automatic_widget)
        self.auto_method_group.setExclusive(True)

        self.auto_method_buttons = {}
        method_names = [
            "Shi-Tomasi",
        ]
        for i, name in enumerate(method_names):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            if i == 0:
                button.setChecked(True)
            self.auto_method_group.addButton(button, i)
            self.automatic_layout.addWidget(button)
            self.auto_method_buttons[name] = button

        self.auto_method_group.idClicked.connect(self.auto_method_selected)

        self.automatic_layout.addSpacing(10)

        # Dynamic method-specific widgets (for now shared)
        self.candidate_count_label = QtWidgets.QLabel("N candidate points: 0")
        font = self.candidate_count_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.candidate_count_label.setFont(font)
        self.automatic_layout.addWidget(self.candidate_count_label)

        self.clear_candidates_button = QtWidgets.QPushButton("Clear candidates")
        self.clear_candidates_button.clicked.connect(self.clear_candidates)
        self.automatic_layout.addWidget(self.clear_candidates_button)

        # Checkbox to show/hide scatter and ROI overlay
        self.show_points_checkbox = QtWidgets.QCheckBox("Show points/ROIs")
        self.show_points_checkbox.setChecked(False)
        def toggle_points_and_roi(state):
            self.roi_overlay.setVisible(state)
            self.scatter.setVisible(state)
        self.show_points_checkbox.stateChanged.connect(toggle_points_and_roi)
        self.automatic_layout.addWidget(self.show_points_checkbox)

        # Horizontal line separator for visual clarity
        hline = QtWidgets.QFrame()
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.automatic_layout.addWidget(hline)

        self.automatic_layout.addSpacing(10)

        # Shi-Tomasi method settings
        self.threshold_label = QtWidgets.QLabel("Threshold:")
        self.threshold_label.setVisible(False)
        self.automatic_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setValue(10)
        self.threshold_slider.setVisible(False)
        self.automatic_layout.addWidget(self.threshold_slider)

        def update_label_and_recompute(val):
            self.threshold_label.setText(f"Threshold: {str(val)}")
            self.compute_candidate_points_shi_tomasi()  # Placeholder method
        self.threshold_slider.valueChanged.connect(update_label_and_recompute)

        self.automatic_layout.addStretch(1)

    def auto_method_selected(self, id: int):
        method_name = list(self.auto_method_buttons.keys())[id]
        print(f"Selected automatic method: {method_name}")
        # Here you can switch method behavior, show/hide widgets, etc.
        is_shi_tomasi = method_name == "Shi-Tomasi"
        self.threshold_label.setVisible(is_shi_tomasi)
        self.threshold_slider.setVisible(is_shi_tomasi)

        if is_shi_tomasi:
            self.compute_candidate_points_shi_tomasi()

    def method_selected(self, id: int):
        method_name = list(self.method_buttons.keys())[id]
        print(f"Selected method: {method_name}")
        is_along = method_name == "Along the line"
        is_grid = method_name == "Grid"
        is_brush = method_name == "Brush"

        # Always enable mouse; painting is now conditional on Ctrl
        self.view.setMouseEnabled(True, True)

        show_spacing = is_along or is_grid

        self.start_new_line_button.setVisible(is_along or is_grid)
        self.polygon_list.setVisible(is_along)
        self.delete_polygon_button.setVisible(is_along)
        self.grid_list.setVisible(is_grid)
        self.delete_grid_button.setVisible(is_grid)

        self.distance_label.setVisible(show_spacing)
        self.distance_slider.setVisible(show_spacing)

    def switch_mode(self, mode: str):
        self.mode = mode
        if mode == "manual":
            self.manual_mode_button.setChecked(True)
            self.automatic_mode_button.setChecked(False)
            self.stack.setCurrentWidget(self.manual_widget)

            self.roi_overlay.setVisible(True)
            self.scatter.setVisible(True)
            # self.candidate_scatter.setVisible(False)

        elif mode == "automatic":
            self.manual_mode_button.setChecked(False)
            self.automatic_mode_button.setChecked(True)
            self.stack.setCurrentWidget(self.automatic_widget)

            self.compute_candidate_points_shi_tomasi()
            self.show_points_checkbox.setChecked(False)
            self.roi_overlay.setVisible(False)
            self.scatter.setVisible(False)
            # self.candidate_scatter.setVisible(True)

    def on_mouse_click(self, event):
        if self.mode == "automatic":
            return
        
        if self.method_buttons["Manual"].isChecked():
            self.handle_manual_selection(event)
        elif self.method_buttons["Along the line"].isChecked():
            self.handle_polygon_drawing(event)
        elif self.method_buttons["Grid"].isChecked():
            self.handle_grid_drawing(event)
        elif self.method_buttons["Remove point"].isChecked():
            self.handle_remove_point(event)
        elif self.method_buttons["Brush"].isChecked():
            self.handle_brush_start(event)

    def update_selected_points(self):
        polygon_points = [pt for poly in self.drawing_polygons for pt in poly['roi_points']]
        grid_points = [pt for g in self.grid_polygons for pt in g['roi_points']]
        self.selected_points = self.manual_points + polygon_points + grid_points

        if not self.selected_points:
            self.scatter.clear()
            self.roi_overlay.clear()
            return

        subset_size = self.subset_size_spinbox.value()
        half = subset_size // 2

        # selected_points = np.round(np.array(self.selected_points) - 0.5)
        selected_points = np.array(self.selected_points)

        # --- Rectangles ---
        if self.show_roi_checkbox.isChecked():
            h, w = self.image_item.image.shape[:2]
            overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

            for y, x in selected_points:
                x0 = int(round(x - half))
                y0 = int(round(y - half))
                x1 = int(round(x + half+1))
                y1 = int(round(y + half+1))

                # Ensure bounds
                if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
                    continue

                # Fill interior (semi-transparent green)
                overlay[y0:y1, x0:x1, 1] = 180  # green
                overlay[y0:y1, x0:x1, 3] = 40   # alpha

                # Outline (more opaque green)
                overlay[y0, x0:x1, 1] = 255  # top
                overlay[y1 - 1, x0:x1, 1] = 255  # bottom
                overlay[y0:y1, x0, 1] = 255  # left
                overlay[y0:y1, x1 - 1, 1] = 255  # right

                overlay[y0, x0:x1, 3] = 150
                overlay[y1 - 1, x0:x1, 3] = 150
                overlay[y0:y1, x0, 3] = 150
                overlay[y0:y1, x1 - 1, 3] = 150

            self.roi_overlay.setImage(overlay, autoLevels=False)
            self.roi_overlay.setZValue(1)
        else:
            self.roi_overlay.clear()

        # --- Center Dots ---
        self.scatter.setData(
            pos=selected_points + 0.5,
            symbol='o',
            size=6,
            brush=pg.mkBrush(255, 100, 100, 200),
            pen=pg.mkPen(None)
        )
        self.points_label.setText(f"Selected subsets: {len(self.selected_points)}")

    def recompute_roi_points(self):
        subset_size = self.subset_size_spinbox.value()
        spacing = self.distance_slider.value()

        # Update all "along the line" polygons
        for poly in self.drawing_polygons:
            if len(poly['points']) >= 2:
                poly['roi_points'] = points_along_polygon(poly['points'], subset_size, spacing)

        # Update all "grid" polygons
        for grid in self.grid_polygons:
            if len(grid['points']) >= 3:
                grid['roi_points'] = rois_inside_polygon(grid['points'], subset_size, spacing)

        self.update_selected_points()

    def start_new_line(self):
        print("Starting a new line...")

        if self.method_buttons["Along the line"].isChecked():
            self.drawing_polygons.append({'points': [], 'roi_points': []})
            self.active_polygon_index = len(self.drawing_polygons) - 1
            self.polygon_list.addItem(f"Polygon {self.active_polygon_index + 1}")
            self.polygon_list.setCurrentRow(self.active_polygon_index)
            self.update_polygon_display()

        elif self.method_buttons["Grid"].isChecked():
            self.grid_polygons.append({'points': [], 'roi_points': []})
            self.active_grid_index = len(self.grid_polygons) - 1
            self.grid_list.addItem(f"Grid {self.active_grid_index + 1}")
            self.grid_list.setCurrentRow(self.active_grid_index)    
            self.update_grid_display()

        self.update_selected_points()

    def clear_selection(self):
        print("Clearing selections...")

        # Clear manual points
        self.manual_points = []

        # Clear line-based polygons
        self.drawing_polygons = [{'points': [], 'roi_points': []}]
        self.polygon_list.clear()
        self.polygon_list.addItem("Polygon 1")
        self.polygon_list.setCurrentRow(0)
        self.active_polygon_index = 0
        self.polygon_line.clear()
        self.polygon_points_scatter.clear()

        # Clear grid-based polygons
        self.grid_polygons = [{'points': [], 'roi_points': []}]
        self.grid_list.clear()
        self.grid_list.addItem("Grid 1")
        self.grid_list.setCurrentRow(0)
        self.active_grid_index = 0

        if hasattr(self, 'grid_line'):
            self.grid_line.clear()
        if hasattr(self, 'grid_points_scatter'):
            self.grid_points_scatter.clear()

        # Clear selected points and visual overlays
        self.selected_points = []

        if hasattr(self, 'scatter'):
            self.scatter.clear()
        if hasattr(self, 'roi_overlay'):
            self.roi_overlay.clear()
        
        # Clear candidate points from automatic filtering
        self.clear_candidates()

        self.points_label.setText("Selected subsets: 0")

        self.update_selected_points()  # Refresh display

    def set_image(self, img: np.ndarray):
        """Display image in the manual tab."""
        self.image_item.setImage(img)

    def get_points(self):
        """Get all selected points from manual and polygons."""
        return np.array(self.selected_points)
    
    def get_filtered_points(self):
        """Get candidate points from automatic filtering."""
        return self.candidate_points.copy() if hasattr(self, 'candidate_points') else []

    # Grid selection
    def handle_grid_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Add first grid polygon to the list if not yet shown
            if self.grid_list.count() == 0:
                self.grid_list.addItem("Grid 1")
                self.grid_list.setCurrentRow(0)

            grid = self.grid_polygons[self.active_grid_index]
            grid['points'].append((x, y))

            # Compute ROI points only if closed polygon
            if len(grid['points']) >= 3:
                subset_size = self.subset_size_spinbox.value()
                spacing = self.distance_slider.value()
                grid['roi_points'] = rois_inside_polygon(grid['points'], subset_size, spacing)

            self.update_grid_display()
            self.update_selected_points()

    def on_grid_selected(self, index):
        if 0 <= index < len(self.grid_polygons):
            self.active_grid_index = index

    def delete_selected_grid(self):
        row = self.grid_list.currentRow()
        if row >= 0 and len(self.grid_polygons) > 1:
            del self.grid_polygons[row]
            self.grid_list.takeItem(row)
            self.active_grid_index = max(0, row - 1)
            self.grid_list.setCurrentRow(self.active_grid_index)
            self.update_grid_display()
            self.update_selected_points()

    def update_grid_display(self):
        # Combine all points from all grid polygons for scatter
        all_pts = [pt for poly in self.grid_polygons for pt in poly['points']]
        
        # Create or update scatter plot for grid polygon vertices
        if not hasattr(self, 'grid_points_scatter'):
            self.grid_points_scatter = ScatterPlotItem(
                pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 200, 0, 200),
                size=6
            )
            self.view.addItem(self.grid_points_scatter)
        self.grid_points_scatter.setData(pos=all_pts)

        # Combine all polygon outlines with np.nan-separated segments
        xs, ys = [], []
        for poly in self.grid_polygons:
            path = poly['points']
            if len(path) >= 2:
                xs.extend([p[0] for p in path] + [path[0][0], np.nan])  # Close polygon
                ys.extend([p[1] for p in path] + [path[0][1], np.nan])
            elif len(path) == 1:
                xs.extend([path[0][0], path[0][0], np.nan])
                ys.extend([path[0][1], path[0][1], np.nan])

        # Create or update line plot for polygon outlines
        if not hasattr(self, 'grid_line'):
            self.grid_line = pg.PlotDataItem(
                pen=pg.mkPen('c', width=2)  # Cyan line
            )
            self.view.addItem(self.grid_line)
        self.grid_line.setData(xs, ys)

    # Manual selection
    def handle_manual_selection(self, event):
        """Handle manual selection of points."""
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x-0.5), round(y-0.5)
            self.manual_points.append((x_int, y_int))
            self.update_selected_points()

    # Along the line selection
    def handle_polygon_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Add first polygon to the list if not yet shown
            if self.polygon_list.count() == 0:
                self.polygon_list.addItem("Polygon 1")
                self.polygon_list.setCurrentRow(0)

            poly = self.drawing_polygons[self.active_polygon_index]
            poly['points'].append((x, y))

            # Update ROI points only for this polygon
            if len(poly['points']) >= 2:
                subset_size = self.subset_size_spinbox.value()
                spacing = self.distance_slider.value()
                poly['roi_points'] = points_along_polygon(poly['points'], subset_size, spacing)

            self.update_polygon_display()
            self.update_selected_points()

    def delete_selected_polygon(self):
        row = self.polygon_list.currentRow()
        if row >= 0 and len(self.drawing_polygons) > 1:
            del self.drawing_polygons[row]
            self.polygon_list.takeItem(row)
            self.active_polygon_index = max(0, row - 1)
            self.polygon_list.setCurrentRow(self.active_polygon_index)
            self.update_polygon_display()
            self.update_selected_points()

    def update_polygon_display(self):
        all_pts = [pt for poly in self.drawing_polygons for pt in poly['points']]
        self.polygon_points_scatter.setData(pos=all_pts)

        xs, ys = [], []
        for poly in self.drawing_polygons:
            path = poly['points']
            if len(path) >= 2:
                xs.extend([p[0] for p in path] + [np.nan])
                ys.extend([p[1] for p in path] + [np.nan])
            elif len(path) == 1:
                xs.extend([path[0][0], path[0][0], np.nan])
                ys.extend([path[0][1], path[0][1], np.nan])

        self.polygon_line.setData(xs, ys)

    def on_polygon_selected(self, index):
        if 0 <= index < len(self.drawing_polygons):
            self.active_polygon_index = index

    # Remove point selection
    def handle_remove_point(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Find nearest point
            if not self.selected_points:
                return

            pts = np.array(self.selected_points)
            distances = np.linalg.norm(pts - np.array([x, y]), axis=1)
            idx = np.argmin(distances)
            closest = tuple(pts[idx])

            # Remove from manual if present
            if closest in self.manual_points:
                self.manual_points.remove(closest)

            # Remove from polygons
            for poly in self.drawing_polygons:
                if closest in poly['roi_points']:
                    poly['roi_points'].remove(closest)

            # Remove from grid
            for grid in self.grid_polygons:
                if closest in grid['roi_points']:
                    grid['roi_points'].remove(closest)

            self.update_selected_points()
    
    # Automatic filtering
    def compute_candidate_points_shi_tomasi(self):
        """Compute good feature points using structure tensor analysis (Shi–Tomasi style)."""
        from scipy.ndimage import sobel

        subset_size = self.subset_size_spinbox.value()
        roi_size = subset_size // 2
        threshold_ratio = self.threshold_slider.value() / 1000.0

        img = self.image_item.image.astype(np.float32)
        candidates = []

        # All selected points (not just manual)
        for row, col in self.selected_points:
            y, x = int(round(row)), int(round(col))

            if (y - roi_size < 0 or y + roi_size + 1 > img.shape[0] or
                x - roi_size < 0 or x + roi_size + 1 > img.shape[1]):
                continue

            roi = img[y - roi_size: y + roi_size + 1,
                    x - roi_size: x + roi_size + 1]

            # Compute gradients
            gx = sobel(roi, axis=1)
            gy = sobel(roi, axis=0)

            Gx2 = np.sum(gx ** 2)
            Gy2 = np.sum(gy ** 2)
            GxGy = np.sum(gx * gy)

            matrix = np.array([[Gx2, GxGy],
                            [GxGy, Gy2]])

            eigvals = np.linalg.eigvalsh(matrix)  # sorted ascending
            min_eig = eigvals[0]

            candidates.append((x + 0.0, y + 0.0, min_eig))

        if not candidates:
            self.candidate_points = []
            self.update_candidate_display()
            return

        # Threshold by normalized eigenvalue
        eigvals = np.array([v[2] for v in candidates])
        max_eig = np.max(eigvals)
        eig_threshold = max_eig * threshold_ratio

        self.candidate_points = [(round(y)+0.5, round(x)+0.5) for (x, y, e) in candidates if e > eig_threshold]
        self.update_candidate_display()
        self.update_candidate_points_count()

    def update_candidate_points_count(self):
        """Update the displayed count of candidate points."""
        if self.candidate_points:
            count_text = f"N candidate points: {len(self.candidate_points)}"
        else:
            count_text = "N candidate points: 0"

        self.candidate_count_label.setText(count_text)

    def update_candidate_display(self):
        """Show candidate points as scatter dots on the image."""
        if not hasattr(self, 'candidate_scatter'):
            self.candidate_scatter = ScatterPlotItem(
                pen=pg.mkPen(None),
                brush=pg.mkBrush(0, 255, 0, 150),  # green with transparency
                size=6,
                symbol='o'
            )
            self.view.addItem(self.candidate_scatter)

        if self.candidate_points:
            self.candidate_scatter.setData(pos=self.candidate_points)
        else:
            self.candidate_scatter.clear()

    def clear_candidates(self):
        """Clear candidate points."""
        print("Clearing candidate points...")
        self.candidate_points = []
        self.update_candidate_points_count()
        if hasattr(self, 'candidate_scatter'):
            self.candidate_scatter.clear()

        self.update_selected_points()  # Update main display to remove candidates
    
    # Brush
    def handle_brush_start(self, ev):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)
        if self.image_item.image is None:
            return
        h, w = self.image_item.image.shape[:2]
        self._paint_mask = np.zeros((h, w), dtype=bool)
        self.handle_brush_move(ev)

    def handle_brush_move(self, ev):
        if self._paint_mask is None:
            return

        pos = ev.pos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            y, x = int(round(mouse_point.x())), int(round(mouse_point.y()))
            r = self._paint_radius

            h, w = self._paint_mask.shape
            yy, xx = np.ogrid[max(0, y - r):min(h, y + r + 1),
                            max(0, x - r):min(w, x + r + 1)]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
            self._paint_mask[max(0, y - r):min(h, y + r + 1),
                            max(0, x - r):min(w, x + r + 1)][mask] = True

            self.update_brush_overlay()

    def handle_brush_end(self, ev):
        QtWidgets.QApplication.restoreOverrideCursor()

        if self._paint_mask is None:
            return

        subset_size = self.subset_size_spinbox.value()
        spacing = self.distance_slider.value()
        brush_rois = rois_inside_mask(self._paint_mask, subset_size, spacing)
        self.manual_points.extend(brush_rois)

        self._paint_mask = None
        self.update_selected_points()
        self.update_brush_overlay()


    def update_brush_overlay(self):
        if not hasattr(self, 'brush_overlay'):
            self.brush_overlay = ImageItem()
            self.view.addItem(self.brush_overlay)

        if self._paint_mask is not None:
            rgba = np.zeros((*self._paint_mask.shape, 4), dtype=np.uint8)
            rgba[self._paint_mask] = [0, 200, 255, 80]  # Cyan with transparency
            self.brush_overlay.setImage(rgba, autoLevels=False)
            self.brush_overlay.setZValue(2)
        else:
            self.brush_overlay.clear()
    ################################################################################################
    # Automatic subset detection
    ################################################################################################


def points_along_polygon(polygon, subset_size, spacing=0):
    if len(polygon) < 2:
        return []

    step = subset_size + spacing
    if step <= 0:
        step = 1

    result_points = []

    for i in range(len(polygon) - 1):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[i + 1])
        segment = p2 - p1
        length = np.linalg.norm(segment)

        if length == 0:
            continue

        direction = segment / length
        n_points = int(length // step)

        for j in range(n_points + 1):
            pt = p1 + j * step * direction
            result_points.append((round(pt[0] - 0.5), round(pt[1] - 0.5)))

    return result_points

def rois_inside_polygon(polygon, subset_size, spacing):
    if len(polygon) < 3:
        return []

    polygon = np.array(polygon)
    min_x, max_x = int(np.floor(np.min(polygon[:, 0]))), int(np.ceil(np.max(polygon[:, 0])))
    min_y, max_y = int(np.floor(np.min(polygon[:, 1]))), int(np.ceil(np.max(polygon[:, 1])))

    step = subset_size + spacing
    if step <= 0:
        step = 1  # minimum step to avoid infinite loop
    xs = np.arange(min_x, max_x+1, step)
    ys = np.arange(min_y, max_y+1, step)

    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    mask = Path(polygon).contains_points(points)
    return [tuple(p) for p in points[mask]]

def rois_inside_mask(mask, subset_size, spacing):
    step = subset_size + spacing
    if step <= 0:
        step = 1

    h, w = mask.shape
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    grid_x, grid_y = np.meshgrid(xs, ys)

    candidate_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T  # (y, x)

    # Only keep points where the mask is True
    selected = [tuple(p) for p in candidate_points if mask[p[0], p[1]]]
    return selected

if __name__ == "__main__":
    # import pyidi
    # filename = "data/data_showcase.cih"
    # video = pyidi.VideoReader(filename)
    # example_image = (video.get_frame(0).T)[:, ::-1]


    import requests
    from PIL import Image
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    # Example black and white image (public domain)
    url = "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/camera.png"
    # Fetch the image
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("L")  # Convert to grayscale
    # Convert to numpy array
    example_image = (np.array(img).T)[:, ::-1]


    Points = SelectionGUI(example_image.astype(np.uint8))

    print(Points.get_points())  # Print selected points for testing
