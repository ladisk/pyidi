import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

class ResultViewer(QtWidgets.QMainWindow):
    def __init__(self, video, displacements, grid, fps=30, magnification=1, point_size=10, colormap="cool"):
        super().__init__()
        self.video = video
        self.displacements = displacements
        self.grid = grid
        self.fps = fps
        self.magnification = magnification
        self.points_size = point_size
        self.current_frame = 0

        self.disp_max = np.max(displacements)
        self.colormap = colormap

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.update_frame()

    def init_ui(self):
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
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #3a3a3a;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: 4px;
                padding: 2px 10px;
                color: #e0e0e0;
                background-color: #4a4a4a;
                border: 1px solid #666;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
        """)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Add splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter, stretch=1)

        # === Video Display ===
        self.view = pg.GraphicsLayoutWidget()
        self.img_item = pg.ImageItem()
        self.scatter = pg.ScatterPlotItem(size=self.points_size, brush='r', pxMode=True)
        self.viewbox = self.view.addViewBox()
        self.viewbox.addItem(self.img_item)
        self.viewbox.addItem(self.scatter)
        self.viewbox.setAspectLocked(True)
        self.viewbox.invertY(True)
        self.arrow_shafts = []
        self.splitter.addWidget(self.view)

        # === Right Control Panel ===
        self.control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        
        # Display controls group
        display_group = QtWidgets.QGroupBox("Display Controls")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        
        # Point size control
        display_layout.addWidget(QtWidgets.QLabel("Point size (px):"))
        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 100)
        self.point_size_spin.setValue(self.points_size)
        self.point_size_spin.valueChanged.connect(self.update_point_size)
        display_layout.addWidget(self.point_size_spin)

        # Magnification control
        display_layout.addWidget(QtWidgets.QLabel("Magnify:"))
        self.mag_spin = QtWidgets.QSpinBox()
        self.mag_spin.setRange(1, 10000)
        self.mag_spin.setValue(self.magnification)
        self.mag_spin.valueChanged.connect(self.update_frame)
        display_layout.addWidget(self.mag_spin)

        # Show arrows checkbox
        self.arrows_checkbox = QtWidgets.QCheckBox("Show arrows")
        self.arrows_checkbox.stateChanged.connect(self.update_frame)
        display_layout.addWidget(self.arrows_checkbox)
        
        self.control_layout.addWidget(display_group)

        # Playback controls group
        playback_group = QtWidgets.QGroupBox("Playback Controls")
        playback_layout = QtWidgets.QVBoxLayout(playback_group)

        # FPS control
        self.fps_label = QtWidgets.QLabel(f"FPS: {self.fps}")
        playback_layout.addWidget(self.fps_label)
        
        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 240)
        self.fps_slider.setValue(self.fps)
        self.fps_slider.valueChanged.connect(self.update_fps_from_slider)
        playback_layout.addWidget(self.fps_slider)
        
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.valueChanged.connect(self.update_fps_from_spinbox)
        playback_layout.addWidget(self.fps_spin)
        
        self.control_layout.addWidget(playback_group)

        self.control_layout.addStretch(1)
        
        self.splitter.addWidget(self.control_widget)
        
        # Set splitter proportions
        self.splitter.setStretchFactor(0, 5)  # Video area grows more
        self.splitter.setStretchFactor(1, 0)  # Controls panel fixed by content
        
        # Set initial width for right panel
        self.control_widget.setMinimumWidth(150)
        self.control_widget.setMaximumWidth(600)
        self.splitter.setSizes([800, 200])  # Initial left/right width

        # === Bottom Playback Controls ===
        playback_layout = QtWidgets.QHBoxLayout()
        playback_layout.setContentsMargins(5, 5, 5, 5)

        self.play_button = QtWidgets.QPushButton("▶️")
        self.play_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_button)
        

        playback_layout.addWidget(QtWidgets.QLabel("  Frame: "))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.video.shape[0] - 1)
        self.slider.valueChanged.connect(self.on_slider)
        playback_layout.addWidget(self.slider)

        main_layout.addLayout(playback_layout)

        # === Finalize ===
        self.setCentralWidget(central_widget)
        self.setWindowTitle("Displacement Viewer")


    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶️")
        else:
            self.set_timer_interval()
            self.timer.start()
            self.play_button.setText("⏹️")

    def set_timer_interval(self):
        self.fps = self.fps_spin.value()
        interval_ms = int(1000 / self.fps)
        self.timer.setInterval(interval_ms)

    def on_fps_change(self):
        if self.timer.isActive():
            self.set_timer_interval()

    def update_fps_from_slider(self, value):
        self.fps = value
        self.fps_label.setText(f"FPS: {value}")
        self.fps_spin.blockSignals(True)  # Prevent recursive updates
        self.fps_spin.setValue(value)
        self.fps_spin.blockSignals(False)
        self.on_fps_change()

    def update_fps_from_spinbox(self, value):
        self.fps = value
        self.fps_label.setText(f"FPS: {value}")
        self.fps_slider.blockSignals(True)  # Prevent recursive updates
        self.fps_slider.setValue(value)
        self.fps_slider.blockSignals(False)
        self.on_fps_change()

    def update_point_size(self):
        size = self.point_size_spin.value()
        self.scatter.setSize(size)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.video.shape[0]
        self.slider.setValue(self.current_frame)

    def on_slider(self, val):
        self.current_frame = val
        self.update_frame()

    def update_frame(self):
        frame = self.video[self.current_frame]
        self.img_item.setImage(frame.T)

        scale = self.mag_spin.value()
        displ = self.displacements[:, self.current_frame, :] * scale
        displaced_pts = self.grid + displ
        self.scatter.setData(pos=displaced_pts[:, [0, 1]])

        if self.arrows_checkbox.isChecked():
            self.scatter.setVisible(False)

            displ = displaced_pts - self.grid
            magnitudes = np.linalg.norm(displ, axis=1)

            norm = mcolors.Normalize(vmin=0, vmax=self.disp_max*scale)
            cmap = plt.colormaps[self.colormap]  # Use the specified colormap

            # Clear old shafts
            for shaft in self.arrow_shafts:
                self.viewbox.removeItem(shaft)
            self.arrow_shafts.clear()

            # Add colored shafts
            for pt0, pt1, mag in zip(self.grid, displaced_pts, magnitudes):
                color = cmap(norm(mag))
                color_rgb = tuple(int(255 * c) for c in color[:3])

                shaft = pg.PlotCurveItem(
                    x=[pt0[0], pt1[0]],
                    y=[pt0[1], pt1[1]],
                    pen=pg.mkPen(color_rgb, width=self.point_size_spin.value())
                )
                self.arrow_shafts.append(shaft)
                self.viewbox.addItem(shaft)

        else:
            self.scatter.setVisible(True)
            for shaft in self.arrow_shafts:
                self.viewbox.removeItem(shaft)
            self.arrow_shafts.clear() 
    
    ########################################################################################~
    # # Uncomment this method if you want to simulate sinusoidal motion (mode shape.)
    ########################################################################################~
    # def update_frame(self):
    #     # Always show the first frame
    #     frame = self.video[0]
    #     self.img_item.setImage(frame.T)

    #     # Get current time index
    #     t = self.current_frame
    #     scale = self.mag_spin.value() / 100

    #     # Simulate sinusoidal motion (like a mode shape oscillating over time)
    #     omega = 2 * np.pi / 100  # You can adjust this "period"
    #     factor = np.sin(omega * t)

    #     # Assume mode shape is stored in self.displacements[:, 0, :] (only the shape)
    #     simulated_displ = self.displacements[:, 0, :] * scale * factor
    #     displaced_pts = self.grid + simulated_displ
    #     self.scatter.setData(pos=displaced_pts[:, [0, 1]])

    #     if self.arrows_checkbox.isChecked():
    #         self.scatter.setVisible(False)

    #         # Arrow color represents displacement magnitude
    #         magnitudes = np.linalg.norm(simulated_displ, axis=1)
    #         norm = mcolors.Normalize(vmin=0, vmax=self.disp_max * scale)
    #         cmap = plt.colormaps[self.colormap]

    #         for shaft in self.arrow_shafts:
    #             self.viewbox.removeItem(shaft)
    #         self.arrow_shafts.clear()

    #         for pt0, pt1, mag in zip(self.grid, displaced_pts, magnitudes):
    #             color = cmap(norm(mag))
    #             color_rgb = tuple(int(255 * c) for c in color[:3])

    #             shaft = pg.PlotCurveItem(
    #                 x=[pt0[0], pt1[0]],
    #                 y=[pt0[1], pt1[1]],
    #                 pen=pg.mkPen(color_rgb, width=self.point_size_spin.value())
    #             )
    #             self.arrow_shafts.append(shaft)
    #             self.viewbox.addItem(shaft)
    #     else:
    #         self.scatter.setVisible(True)
    #         for shaft in self.arrow_shafts:
    #             self.viewbox.removeItem(shaft)
    #         self.arrow_shafts.clear()


def viewer(frames, displacements, points, fps=30, magnification=1, point_size=10, colormap="cool"):
    """Viewer for the videos and displacements.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (n_frames, height, width) containing the video frames.
    displacements : np.ndarray
        Array of shape (n_points, n_frames, 2) containing the displacements for
        each point in each frame. The directions of the last axis are the vertical and horizontal
        displacements, respectively.
    points : np.ndarray
        Array of shape (n_points, 2) containing the initial positions of the points. The first
        column is the vertical coordinate (y) and the second column is the horizontal coordinate (x).
    fps : int, optional
        Frames per second for the video playback, by default 30.
    magnification : int, optional
        Magnification factor for the displacements, by default 1.
    point_size : int, optional
        Size of the points in pixels, by default 10.
    colormap : str, optional
        Name of the colormap to use for the arrows, by default "cool".
    """
    points = points[:, ::-1]
    displacements = displacements[:, :, ::-1]

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    win = ResultViewer(frames, displacements, points, fps=fps, magnification=magnification, point_size=point_size, colormap=colormap)
    win.resize(800, 600)
    win.show()
    
    # Only call sys.exit if not in IPython
    if not hasattr(sys, 'ps1'):  # Not interactive
        sys.exit(app.exec())
    else:
        app.exec()  # Don't raise SystemExit in IPythonys


if __name__ == "__main__":
    n_frames, height, width = 200, 300, 400
    n_points = 100
    frames = np.random.randint(0, 255, size=(n_frames, height, width), dtype=np.uint8)
    displacements = 2 * (np.random.rand(n_points, n_frames, 2) - 0.5)
    grid = np.stack(np.meshgrid(np.linspace(50, 350, int(np.sqrt(n_points))),
                                np.linspace(50, 250, int(np.sqrt(n_points)))), axis=-1).reshape(-1, 2)[:n_points]
    grid = grid[:, ::-1]
    viewer(frames, displacements, grid)