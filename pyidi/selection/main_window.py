from operator import index
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem
import pyqtgraph as pg
import numpy as np
import sys

from along_line import points_along_polygon

class SelectionGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Selection Tool")
        self.resize(1200, 800)

        self.selected_points = []
        self.selection_rects = []
        self.drawing_polygons = [{'points': [], 'roi_points': []}]
        self.active_polygon_index = 0

        self.scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 200), size=8)

        # Central widget with tab layout
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Manual Tab ---
        self.manual_tab = QtWidgets.QWidget()
        self.manual_layout = QtWidgets.QHBoxLayout(self.manual_tab)

        # Image viewer
        self.pg_widget = GraphicsLayoutWidget()
        self.view = self.pg_widget.addViewBox(lockAspect=True)
        self.image_item = ImageItem()
        self.polygon_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.polygon_points_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=6)
        self.view.addItem(self.image_item)
        self.view.addItem(self.polygon_line)
        self.view.addItem(self.polygon_points_scatter)
        self.view.addItem(self.scatter)  # Add scatter for showing points
        self.manual_layout.addWidget(self.pg_widget, stretch=1)

        # Method buttons on the right
        self.method_widget = QtWidgets.QWidget()
        self.method_layout = QtWidgets.QVBoxLayout(self.method_widget)

        self.button_group = QtWidgets.QButtonGroup(self.method_widget)
        self.button_group.setExclusive(True)

        self.method_buttons = {}
        method_names = [
            "ROI grid",
            "Deselect polygon",
            "Only polygon",
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
        self.manual_layout.addWidget(self.method_widget)

        # Polygon manager (visible only for "Along the line")
        self.polygon_list = QtWidgets.QListWidget()
        self.polygon_list.setVisible(False)
        self.polygon_list.currentRowChanged.connect(self.on_polygon_selected)
        self.method_layout.addWidget(self.polygon_list)

        self.delete_polygon_button = QtWidgets.QPushButton("Delete selected polygon")
        self.delete_polygon_button.clicked.connect(self.delete_selected_polygon)
        self.delete_polygon_button.setVisible(False)
        self.method_layout.addWidget(self.delete_polygon_button)

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

    def method_selected(self, id: int):
        method_name = list(self.method_buttons.keys())[id]
        print(f"Selected method: {method_name}")
        is_along = method_name == "Along the line"
        self.start_new_line_button.setVisible(is_along)
        self.polygon_list.setVisible(is_along)
        self.delete_polygon_button.setVisible(is_along)

    def on_mouse_click(self, event):
        if self.method_buttons["Manual"].isChecked():
            self.handle_manual_selection(event)
        elif self.method_buttons["Along the line"].isChecked():
            self.handle_polygon_drawing(event)
            
    def handle_manual_selection(self, event):
        """Handle manual selection of points."""
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x-0.5)+0.5, round(y-0.5)+0.5
            self.selected_points.append((x_int, y_int))
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

    def update_selected_points(self):
        self.selected_points = [pt for poly in self.drawing_polygons for pt in poly['roi_points']]

        for rect in self.selection_rects:
            self.view.removeItem(rect)
        self.selection_rects.clear()

        if self.selected_points:
            spots = [{'pos': pt} for pt in self.selected_points]
            self.scatter.setData(spots)

            subset_size = self.subset_size_spinbox.value()
            half_size = subset_size / 2

            for x, y in self.selected_points:
                rect = pg.QtWidgets.QGraphicsRectItem(
                    x - half_size, y - half_size, subset_size, subset_size
                )
                rect.setPen(pg.mkPen((100, 255, 100, 200), width=2))
                self.view.addItem(rect)
                self.selection_rects.append(rect)
        else:
            self.scatter.clear()

    def start_new_line(self):
        print("Starting a new line...")
        self.drawing_polygons.append({'points': [], 'roi_points': []})
        self.active_polygon_index = len(self.drawing_polygons) - 1
        self.polygon_list.addItem(f"Polygon {self.active_polygon_index + 1}")
        self.polygon_list.setCurrentRow(self.active_polygon_index)
        self.update_polygon_display()
        self.update_selected_points()


    def clear_selection(self):
        print("Clearing selections...")
        self.drawing_polygons = [{'points': [], 'roi_points': []}]
        self.polygon_list.clear()
        self.polygon_list.addItem("Polygon 1")
        self.polygon_list.setCurrentRow(0)
        self.active_polygon_index = 0
        self.polygon_line.clear()
        self.polygon_points_scatter.clear()
        self.update_selected_points()

    def set_image(self, img: np.ndarray):
        """Display image in the manual tab."""
        self.image_item.setImage(img)

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = SelectionGUI()

    # Example grayscale image
    example_image = np.random.rand(512, 512) * 255
    gui.set_image(example_image.astype(np.uint8))

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
