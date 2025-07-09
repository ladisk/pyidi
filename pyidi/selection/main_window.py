from operator import index
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem
import pyqtgraph as pg
import numpy as np
import sys

from along_line import points_along_polygon

class SelectionGUI(QtWidgets.QMainWindow):
    def __init__(self, video):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        super().__init__()

        self.setWindowTitle("ROI Selection Tool")
        self.resize(1200, 800)

        self.selected_points = []
        self.manual_points = []
        self.drawing_polygons = [{'points': [], 'roi_points': []}]
        self.active_polygon_index = 0
        self.grid_polygons = [{'points': [], 'roi_points': []}]
        self.active_grid_index = 0

        # Central widget with tab layout
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Manual Tab ---
        self.manual_tab = QtWidgets.QWidget()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.manual_layout = QtWidgets.QHBoxLayout(self.manual_tab)
        self.manual_layout.addWidget(self.splitter)

        # Graphics layout for image and points display
        self.ui_graphics()
        
        # Right-side menu for methods
        self.ui_manual_right_menu()

        self.tabs.addTab(self.manual_tab, "Manual")

        # --- Automatic Tab ---
        self.automatic_tab = QtWidgets.QWidget()
        self.automatic_layout = QtWidgets.QVBoxLayout(self.automatic_tab)
        self.tabs.addTab(self.automatic_tab, "Automatic")

        # Style
        self.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background: #333;
                color: white;
                padding: 10px;
                border-radius: 4px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background: #0078d7;
            }
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

        # Start the GUI
        self.show()
        if app is not None:
            app.exec()

    def ui_graphics(self):
        # Image viewer
        self.pg_widget = GraphicsLayoutWidget()
        self.view = self.pg_widget.addViewBox(lockAspect=True)
        
        self.image_item = ImageItem()
        self.polygon_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.polygon_points_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=6)
        self.scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 200), size=8)
        self.roi_overlay = ImageItem()

        self.view.addItem(self.image_item)
        self.view.addItem(self.polygon_line)
        self.view.addItem(self.polygon_points_scatter)
        self.view.addItem(self.roi_overlay)  # Add scatter for showing square points
        self.view.addItem(self.scatter)  # Add scatter for showing points

        self.splitter.addWidget(self.pg_widget)

    def ui_manual_right_menu(self):
        # Method buttons on the right
        self.method_widget = QtWidgets.QWidget()
        self.method_layout = QtWidgets.QVBoxLayout(self.method_widget)

        self.button_group = QtWidgets.QButtonGroup(self.method_widget)
        self.button_group.setExclusive(True)

        self.method_buttons = {}
        method_names = [
            "Grid",
            "Manual",
            "Along the line"
        ]
        for i, name in enumerate(method_names):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            if i == 0:
                button.setChecked(True)  # Default selection
            self.button_group.addButton(button, i)
            self.method_layout.addWidget(button)
            self.method_buttons[name] = button

        # Subset size input
        self.method_layout.addSpacing(20)
        self.method_layout.addWidget(QtWidgets.QLabel("Subset size:"))

        self.subset_size_spinbox = QtWidgets.QSpinBox()
        self.subset_size_spinbox.setRange(1, 1000)
        self.subset_size_spinbox.setValue(11)
        self.subset_size_spinbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.subset_size_spinbox.setSingleStep(2)
        self.subset_size_spinbox.setMinimum(1)
        self.subset_size_spinbox.setMaximum(999)
        self.subset_size_spinbox.setWrapping(False)
        self.subset_size_spinbox.valueChanged.connect(self.update_selected_points)
        self.method_layout.addWidget(self.subset_size_spinbox)

        # Show ROI rectangles
        self.show_roi_checkbox = QtWidgets.QCheckBox("Show subsets")
        self.show_roi_checkbox.setChecked(True)
        self.show_roi_checkbox.stateChanged.connect(self.update_selected_points)
        self.method_layout.addWidget(self.show_roi_checkbox)

        # Clear button
        self.method_layout.addSpacing(20)
        self.clear_button = QtWidgets.QPushButton("Clear selections")
        self.clear_button.clicked.connect(self.clear_selection)
        self.method_layout.addWidget(self.clear_button)

        # Start new line (only visible in "Along the line" mode)
        self.start_new_line_button = QtWidgets.QPushButton("Start new line")
        self.start_new_line_button.clicked.connect(self.start_new_line)
        self.start_new_line_button.setVisible(False)  # Hidden by default
        self.method_layout.addWidget(self.start_new_line_button)

        self.method_layout.addStretch(1)

        # Polygon manager (visible only for "Along the line")
        self.polygon_list = QtWidgets.QListWidget()
        self.polygon_list.setVisible(False)
        self.polygon_list.currentRowChanged.connect(self.on_polygon_selected)
        self.method_layout.addWidget(self.polygon_list)

        self.delete_polygon_button = QtWidgets.QPushButton("Delete selected polygon")
        self.delete_polygon_button.clicked.connect(self.delete_selected_polygon)
        self.delete_polygon_button.setVisible(False)
        self.method_layout.addWidget(self.delete_polygon_button)

        # Grid polygon manager
        self.grid_list = QtWidgets.QListWidget()
        self.grid_list.setVisible(False)
        self.grid_list.currentRowChanged.connect(self.on_grid_selected)
        self.method_layout.addWidget(self.grid_list)

        self.delete_grid_button = QtWidgets.QPushButton("Delete selected grid")
        self.delete_grid_button.clicked.connect(self.delete_selected_grid)
        self.delete_grid_button.setVisible(False)
        self.method_layout.addWidget(self.delete_grid_button)

        # Set the layout and add to splitter
        self.splitter.addWidget(self.method_widget)
        self.splitter.setStretchFactor(0, 5)  # Image area grows more
        self.splitter.setStretchFactor(1, 0)  # Menu fixed by content

        # Set initial width for right panel
        self.method_widget.setMinimumWidth(150)
        self.method_widget.setMaximumWidth(600)
        self.splitter.setSizes([1000, 220])  # Initial left/right width

    def method_selected(self, id: int):
        method_name = list(self.method_buttons.keys())[id]
        print(f"Selected method: {method_name}")
        is_along = method_name == "Along the line"
        is_grid = method_name == "Grid"

        self.start_new_line_button.setVisible(is_along or is_grid)
        self.polygon_list.setVisible(is_along)
        self.delete_polygon_button.setVisible(is_along)

        self.grid_list.setVisible(is_grid)
        self.delete_grid_button.setVisible(is_grid)

    def on_mouse_click(self, event):
        if self.method_buttons["Manual"].isChecked():
            self.handle_manual_selection(event)
        elif self.method_buttons["Along the line"].isChecked():
            self.handle_polygon_drawing(event)
        elif self.method_buttons["Grid"].isChecked():
            self.handle_grid_drawing(event)
                
    def handle_manual_selection(self, event):
        """Handle manual selection of points."""
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x-0.5)+0.5, round(y-0.5)+0.5
            self.manual_points.append((x_int, y_int))
            self.update_selected_points()

    def handle_polygon_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x - 0.5) + 0.5, round(y - 0.5) + 0.5

            # Add first polygon to the list if not yet shown
            if self.polygon_list.count() == 0:
                self.polygon_list.addItem("Polygon 1")
                self.polygon_list.setCurrentRow(0)

            poly = self.drawing_polygons[self.active_polygon_index]
            poly['points'].append((x_int, y_int))

            # Update ROI points only for this polygon
            if len(poly['points']) >= 2:
                subset_size = self.subset_size_spinbox.value()
                poly['roi_points'] = points_along_polygon(poly['points'], subset_size)

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

    def points_from_all_polygons(self):
        subset_size = self.subset_size_spinbox.value()
        all_points = []
        for path in self.drawing_polygons:
            if len(path) >= 2:
                all_points.extend(points_along_polygon(path, subset_size))
        return all_points
    
    def delete_selected_polygon(self):
        row = self.polygon_list.currentRow()
        if row >= 0 and len(self.drawing_polygons) > 1:
            del self.drawing_polygons[row]
            self.polygon_list.takeItem(row)
            self.active_polygon_index = max(0, row - 1)
            self.polygon_list.setCurrentRow(self.active_polygon_index)
            self.update_polygon_display()
            self.update_selected_points()
    
    def on_polygon_selected(self, index):
        if 0 <= index < len(self.drawing_polygons):
            self.active_polygon_index = index

    def handle_grid_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x - 0.5) + 0.5, round(y - 0.5) + 0.5

            # Add first grid polygon to the list if not yet shown
            if self.grid_list.count() == 0:
                self.grid_list.addItem("Grid 1")
                self.grid_list.setCurrentRow(0)

            grid = self.grid_polygons[self.active_grid_index]
            grid['points'].append((x_int, y_int))

            # Compute ROI points only if closed polygon
            if len(grid['points']) >= 3:
                subset_size = self.subset_size_spinbox.value()
                grid['roi_points'] = self.rois_inside_polygon(grid['points'], subset_size)

            self.update_grid_display()
            self.update_selected_points()

    def rois_inside_polygon(self, polygon, subset_size):
        from matplotlib.path import Path

        if len(polygon) < 3:
            return []

        polygon = np.array(polygon)
        min_x, max_x = int(np.min(polygon[:, 0])), int(np.max(polygon[:, 0]))
        min_y, max_y = int(np.min(polygon[:, 1])), int(np.max(polygon[:, 1]))

        xs = np.arange(min_x, max_x + 1, subset_size)
        ys = np.arange(min_y, max_y + 1, subset_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        mask = Path(polygon).contains_points(points)
        return [tuple(p) for p in points[mask]]
    
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

        # --- Rectangles ---
        if self.show_roi_checkbox.isChecked():
            h, w = self.image_item.image.shape[:2]
            overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

            for y, x in self.selected_points:
                x0 = int(round(x - half))
                y0 = int(round(y - half))
                x1 = int(round(x + half))
                y1 = int(round(y + half))

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
            pos=self.selected_points,
            symbol='o',
            size=6,
            brush=pg.mkBrush(255, 100, 100, 200),
            pen=pg.mkPen(None)
        )


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


    def set_image(self, img: np.ndarray):
        """Display image in the manual tab."""
        self.image_item.setImage(img)

    def get_points(self):
        """Get all selected points from manual and polygons."""
        return self.selected_points

if __name__ == "__main__":
    example_image = np.random.rand(512, 512) * 255
    Points = SelectionGUI(example_image.astype(np.uint8))

    print(Points.get_points())  # Print selected points for testing
