from PyQt6 import QtWidgets, QtCore
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem
import pyqtgraph as pg
import numpy as np
import sys


class SelectionGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Selection Tool")
        self.resize(1200, 800)

        self.selected_points = []
        self.selection_rects = []
        self.drawing_polygons = [[]]  # list of paths, each path = list of (x, y)


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
        self.view.addItem(self.scatter)  # Add scatter for showing points
        self.view.addItem(self.polygon_line)
        self.view.addItem(self.polygon_points_scatter)
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
        self.start_new_line_button.setVisible(method_name == "Along the line")

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

            self.drawing_polygons[-1].append((x_int, y_int))

            self.update_polygon_display()

            self.selected_points = self.points_from_all_polygons()
            self.update_selected_points()

    def update_polygon_display(self):
        # Flatten all points for scatter
        all_pts = [pt for path in self.drawing_polygons for pt in path]
        self.polygon_points_scatter.setData(pos=all_pts)

        # Combine segments into one continuous line for display
        xs = []
        ys = []
        for path in self.drawing_polygons:
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
                all_points.extend(self.points_along_polygon(path, subset_size))
        return all_points

    def update_selected_points(self):
        # Clear previous rectangles
        for rect in self.selection_rects:
            self.view.removeItem(rect)
        self.selection_rects.clear()

        # Update scatter points
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
        self.drawing_polygons.append([])
        self.update_polygon_display()
        self.selected_points = self.points_from_all_polygons()
        self.update_selected_points()


    def clear_selection(self):
        print("Clearing selections...")
        self.selected_points = []
        self.drawing_polygons = [[]]
        self.polygon_line.clear()
        self.polygon_points_scatter.clear()
        self.update_selected_points()

    def set_image(self, img: np.ndarray):
        """Display image in the manual tab."""
        self.image_item.setImage(img)

    def points_along_polygon(self, polygon, subset_size):
        if len(polygon) < 2:
            return []

        # List of points along the path
        result_points = []

        for i in range(len(polygon) - 1):
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[i + 1])
            segment = p2 - p1
            length = np.linalg.norm(segment)

            if length == 0:
                continue

            direction = segment / length
            n_points = int(length // subset_size)

            for j in range(n_points + 1):
                pt = p1 + j * subset_size * direction
                result_points.append((round(pt[0] - 0.5) + 0.5, round(pt[1] - 0.5) + 0.5))

        return result_points



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
