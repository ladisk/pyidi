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
        # Set Fusion style (looks modern & compact)
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

        # Optional: make it a bit darker but safe
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
        QtWidgets.QApplication.setPalette(dark_palette)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        # Top Controls: keep horizontal as requested
        top_ctrl_layout = QtWidgets.QHBoxLayout()

        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 100)
        self.point_size_spin.setValue(self.points_size)
        self.point_size_spin.valueChanged.connect(self.update_point_size)
        top_ctrl_layout.addWidget(QtWidgets.QLabel("Point size (px):"))
        top_ctrl_layout.addWidget(self.point_size_spin)

        self.mag_spin = QtWidgets.QSpinBox()
        self.mag_spin.setRange(1, 10000)
        self.mag_spin.setValue(self.magnification)
        self.mag_spin.valueChanged.connect(self.update_frame)
        top_ctrl_layout.addWidget(QtWidgets.QLabel("Magnify:"))
        top_ctrl_layout.addWidget(self.mag_spin)

        self.arrows_checkbox = QtWidgets.QCheckBox("Show arrows")
        self.arrows_checkbox.stateChanged.connect(self.update_frame)
        top_ctrl_layout.addWidget(self.arrows_checkbox)

        top_ctrl_layout.addStretch()
        main_layout.addLayout(top_ctrl_layout)

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
        main_layout.addWidget(self.view, stretch=1)

        # === Playback Controls ===
        playback_layout = QtWidgets.QHBoxLayout()
        playback_layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QtWidgets.QPushButton("▶️ Play")
        self.play_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_button)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.video.shape[0] - 1)
        self.slider.valueChanged.connect(self.on_slider)
        playback_layout.addWidget(QtWidgets.QLabel("Frame:"))
        playback_layout.addWidget(self.slider)

        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.valueChanged.connect(self.on_fps_change)
        playback_layout.addWidget(QtWidgets.QLabel("FPS:"))
        playback_layout.addWidget(self.fps_spin)

        main_layout.addLayout(playback_layout)

        # === Finalize ===
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("Displacement Viewer")


    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶️ Play")
        else:
            self.set_timer_interval()
            self.timer.start()
            self.play_button.setText("⏹️ Pause")

    def set_timer_interval(self):
        self.fps = self.fps_spin.value()
        interval_ms = int(1000 / self.fps)
        self.timer.setInterval(interval_ms)

    def on_fps_change(self):
        if self.timer.isActive():
            self.set_timer_interval()

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