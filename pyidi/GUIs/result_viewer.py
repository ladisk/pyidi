import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

class ResultViewer(QtWidgets.QMainWindow):
    def __init__(self, video, displacements, points, fps=30, magnification=1, point_size=10, colormap="cool"):
        """
        The results from the pyidi analysis can directly be passed to this class:
        
        - ``video``: can be a ``VideoReader`` object (or numpy array of correct shape).
        - ``displacements``: directly the return from the ``get_displacements`` method or mode shapes.
        - ``points``: the points used for the analysis, which were passed to the ``set_points`` method.

        Parameters
        ----------
        video : np.ndarray or VideoReader
            Array of shape (n_frames, height, width) containing the video frames.
        displacements : np.ndarray
            Array of shape (n_frames, n_points, 2) for time-series displacements OR
            Array of shape (n_points, 2) for mode shapes.
        points : np.ndarray
            Array of shape (n_points, 2) containing the grid points.
        fps : int, optional
            Frames per second for the video playback, by default 30.
        magnification : int, optional
            Magnification factor for the displacements, by default 1.
        point_size : int, optional
            Size of the points in pixels, by default 10.
        colormap : str, optional
            Name of the colormap to use for the arrows, by default "cool".
        """
        # Create QApplication if it doesn't exist
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        
        super().__init__()
        
        # Coordinate transformation to match viewer function behavior
        from ..video_reader import VideoReader
        if isinstance(video, VideoReader):
            self.video = video.get_frames()
        else:
            self.video = video

        # Check if displacements are 2D (mode shapes) or 3D (time-series)
        if displacements.ndim == 2:
            # Mode shapes: shape (n_points, 2)
            self.is_mode_shape = True
            self.displacements = displacements[:, ::-1]  # Flip x,y coordinates
            self.time_per_period = 1.0 # Seconds
        else:
            # Time-series displacements: shape (n_frames, n_points, 2)
            self.is_mode_shape = False
            self.displacements = displacements[:, :, ::-1]  # Flip x,y coordinates

        self.grid = points[:, ::-1] + 0.5  # Flip x,y coordinates
        self.fps = fps
        self.magnification = magnification
        self.points_size = point_size
        self.current_frame = 0

        self.disp_max = np.max(np.abs(displacements))
        self.colormap = colormap

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.update_frame()
        
        # Start the GUI
        self.show()
        # Only call sys.exit if not in IPython
        if not hasattr(sys, 'ps1'):  # Not interactive
            sys.exit(app.exec())
        else:
            app.exec()  # Don't raise SystemExit in IPython

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
        point_size_layout = QtWidgets.QHBoxLayout()
        point_size_layout.addWidget(QtWidgets.QLabel("Point size:"))
        
        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 100)
        self.point_size_spin.setValue(self.points_size)
        self.point_size_spin.setSuffix("px")
        self.point_size_spin.setFixedWidth(80)
        self.point_size_spin.valueChanged.connect(self.update_point_size_from_spinbox)
        point_size_layout.addWidget(self.point_size_spin)
        
        point_size_layout.addStretch()  # Push everything to the left
        display_layout.addLayout(point_size_layout)
        
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(1, 20)
        self.point_size_slider.setValue(min(20, self.points_size))
        self.point_size_slider.valueChanged.connect(self.update_point_size_from_slider)
        display_layout.addWidget(self.point_size_slider)

        # Magnification control
        mag_layout = QtWidgets.QHBoxLayout()
        mag_layout.addWidget(QtWidgets.QLabel("Magnify:"))
        
        self.mag_spin = QtWidgets.QDoubleSpinBox()
        self.mag_spin.setRange(0.01, 999999)  # No practical upper limit
        self.mag_spin.setSingleStep(0.01)
        self.mag_spin.setValue(self.magnification)
        self.mag_spin.setSuffix("x")
        self.mag_spin.setFixedWidth(80)
        self.mag_spin.valueChanged.connect(self.update_mag_from_spinbox)
        mag_layout.addWidget(self.mag_spin)
        
        mag_layout.addStretch()  # Push everything to the left
        display_layout.addLayout(mag_layout)
        
        self.mag_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mag_slider.setRange(1, 1000)  # 0.1x to 10x (in percent: 10% to 1000%)
        self.mag_slider.setValue(int(self.magnification * 100))
        self.mag_slider.valueChanged.connect(self.update_mag_from_slider)
        display_layout.addWidget(self.mag_slider)

        # Show arrows checkbox
        self.arrows_checkbox = QtWidgets.QCheckBox("Show arrows")
        self.arrows_checkbox.stateChanged.connect(self.update_frame)
        display_layout.addWidget(self.arrows_checkbox)
        
        self.control_layout.addWidget(display_group)

        # Playback controls group
        playback_group = QtWidgets.QGroupBox("Playback Controls")
        playback_layout = QtWidgets.QVBoxLayout(playback_group)

        # FPS control
        fps_layout = QtWidgets.QHBoxLayout()
        fps_layout.addWidget(QtWidgets.QLabel("FPS:"))
        
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.setFixedWidth(80)
        self.fps_spin.valueChanged.connect(self.update_fps_from_spinbox)
        fps_layout.addWidget(self.fps_spin)
        
        fps_layout.addStretch()  # Push everything to the left
        playback_layout.addLayout(fps_layout)
        
        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 240)
        self.fps_slider.setValue(self.fps)
        self.fps_slider.valueChanged.connect(self.update_fps_from_slider)
        playback_layout.addWidget(self.fps_slider)
        
        self.control_layout.addWidget(playback_group)

        # Export controls group
        export_group = QtWidgets.QGroupBox("Export Video")
        export_layout = QtWidgets.QVBoxLayout(export_group)

        # Quality/FPS for export
        export_layout.addWidget(QtWidgets.QLabel("Export FPS:"))
        self.export_fps_spin = QtWidgets.QSpinBox()
        self.export_fps_spin.setRange(1, 120)
        self.export_fps_spin.setValue(30)
        export_layout.addWidget(self.export_fps_spin)

        # Export resolution
        export_layout.addWidget(QtWidgets.QLabel("Export Resolution:"))
        self.export_resolution_combo = QtWidgets.QComboBox()
        self.export_resolution_combo.addItems([
            "2x pixel scale",
            "4x pixel scale", 
            "6x pixel scale",
            "8x pixel scale",
        ])
        self.export_resolution_combo.setCurrentText("4x pixel scale")
        export_layout.addWidget(self.export_resolution_combo)

        # Duration for mode shapes
        if self.is_mode_shape:
            export_layout.addWidget(QtWidgets.QLabel("Duration (seconds):"))
            self.duration_spin = QtWidgets.QDoubleSpinBox()
            self.duration_spin.setRange(0.5, 60.0)
            self.duration_spin.setValue(2.0)
            self.duration_spin.setSingleStep(0.5)
            export_layout.addWidget(self.duration_spin)

        # Export button
        self.export_button = QtWidgets.QPushButton("Export Video")
        self.export_button.clicked.connect(self.export_video)
        export_layout.addWidget(self.export_button)

        # Progress bar
        self.export_progress = QtWidgets.QProgressBar() 
        self.export_progress.setVisible(False)
        export_layout.addWidget(self.export_progress)
        
        self.control_layout.addWidget(export_group)

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
        if self.is_mode_shape:
            self.slider.setRange(0, int(self.fps * self.time_per_period) - 1)
        else:
            self.slider.setRange(0, self.video.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider)
        playback_layout.addWidget(self.slider)

        self.frame_spinbox = QtWidgets.QSpinBox()
        if self.is_mode_shape:
            self.frame_spinbox.setRange(0, int(self.fps * self.time_per_period) - 1)
        else:
            self.frame_spinbox.setRange(0, self.video.shape[0] - 1)

        self.frame_spinbox.valueChanged.connect(self.on_slider)
        self.frame_spinbox.setValue(0)
        playback_layout.addWidget(self.frame_spinbox)

        main_layout.addLayout(playback_layout)

        # === Finalize ===
        self.setCentralWidget(central_widget)
        self.setWindowTitle("Displacement Viewer")
        self.resize(1200, 600)

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

        if self.is_mode_shape:
            self.slider.setMaximum(int(self.fps * self.time_per_period) - 1)
            self.frame_spinbox.setMaximum(int(self.fps * self.time_per_period) - 1)

    def update_fps_from_slider(self, value):
        self.fps = value
        self.fps_spin.blockSignals(True)  # Prevent recursive updates
        self.fps_spin.setValue(value)
        self.fps_spin.blockSignals(False)
        self.on_fps_change()

    def update_fps_from_spinbox(self, value):
        self.fps = value
        self.fps_slider.blockSignals(True)  # Prevent recursive updates
        self.fps_slider.setValue(value)
        self.fps_slider.blockSignals(False)
        self.on_fps_change()

    def update_mag_from_slider(self, value):
        # Convert slider value (1-1000) to magnification (0.01-10.0)
        magnification = value / 100.0
        self.magnification = magnification
        
        # Update spinbox without triggering its signal
        self.mag_spin.blockSignals(True)
        self.mag_spin.setValue(magnification)
        self.mag_spin.blockSignals(False)
        
        self.update_frame()

    def update_mag_from_spinbox(self, value):
        # Convert spinbox value to magnification
        magnification = value
        self.magnification = magnification
        
        # Update slider, clamping to its range and converting to int
        slider_value = int(max(1, min(1000, value * 100)))
        self.mag_slider.blockSignals(True)
        self.mag_slider.setValue(slider_value)
        self.mag_slider.blockSignals(False)
        
        self.update_frame()

    def update_point_size_from_slider(self, value):
        # Update the internal point size
        self.points_size = value
        
        # Update spinbox without triggering its signal
        self.point_size_spin.blockSignals(True)
        self.point_size_spin.setValue(value)
        self.point_size_spin.blockSignals(False)
        
        # Update the actual display
        self.scatter.setSize(value)

    def update_point_size_from_spinbox(self, value):
        # Update the internal point size
        self.points_size = value
        
        # Update slider, clamping to its range
        slider_value = min(20, max(1, value))
        self.point_size_slider.blockSignals(True)
        self.point_size_slider.setValue(slider_value)
        self.point_size_slider.blockSignals(False)
        
        # Update the actual display
        self.scatter.setSize(value)

    def update_point_size(self):
        # Keep this method for backward compatibility if needed
        size = self.point_size_spin.value()
        self.scatter.setSize(size)

    def next_frame(self):
        if self.is_mode_shape:
            self.current_frame = (self.current_frame + 1) % int(self.fps * self.time_per_period)
        else:
            self.current_frame = (self.current_frame + 1) % self.video.shape[0]
        
        self.slider.setValue(self.current_frame)

    def on_slider(self, val):
        self.current_frame = val
        self.frame_spinbox.setValue(val)
        self.slider.setValue(val)
        self.update_frame()

    def update_frame(self):
        scale = self.magnification

        if self.is_mode_shape:
            frame = self.video[0]
            self.img_item.setImage(frame.T)

            # Calculate time for sinusoidal motion
            t = self.current_frame / self.fps  # Convert frame to time in seconds
            
            # Calculate displacement amplitude using sinusoidal motion
            displ_raw = self.displacements
            amplitude = np.abs(displ_raw)
            phase = np.angle(displ_raw)

            # Calculate displacement using sinusoidal motion
            displ = scale * amplitude * np.sin(2 * np.pi * t - phase)

        else:
            # Regular displacement animation
            frame = self.video[self.current_frame]
            self.img_item.setImage(frame.T)

            displ = self.displacements[:, self.current_frame, :] * scale
        
        # Update scatter plot with displaced points
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

    def export_video(self):
        """Export the current visualization as a video file with pixel-perfect rendering."""
        try:
            import cv2
        except ImportError:
            QtWidgets.QMessageBox.warning(self, "Missing Dependency", 
                                        "OpenCV (cv2) is required for video export.\n"
                                        "Install it with: pip install opencv-python")
            return

        # Get export parameters
        export_fps = self.export_fps_spin.value()
        
        # Get pixel scaling factor from resolution selection
        resolution_text = self.export_resolution_combo.currentText()
        if "4x pixel scale" in resolution_text:
            pixel_scale = 4  # Each video pixel becomes 4x4 pixels in export
        elif "2x pixel scale" in resolution_text:
            pixel_scale = 2  # Each video pixel becomes 2x2 pixels in export
        elif "6x pixel scale" in resolution_text:
            pixel_scale = 6  # Each video pixel becomes 6x6 pixels in export
        else:  # 4K
            pixel_scale = 8  # Each video pixel becomes 8x8 pixels in export

        # Calculate export dimensions based on video dimensions and pixel scaling
        video_height, video_width = self.video[0].shape
        export_width = video_width * pixel_scale
        export_height = video_height * pixel_scale
        
        # Use MP4 with high quality settings
        default_ext = "mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # File dialog for save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Video", f"displacement_video.{default_ext}",
            f"MP4 files (*.{default_ext});;All files (*.*)"
        )
        
        if not file_path:
            return

        # Store current state to restore later
        original_frame = self.current_frame
        original_timer_active = self.timer.isActive()
        if original_timer_active:
            self.timer.stop()
        
        # Calculate total frames for export
        if self.is_mode_shape:
            duration = self.duration_spin.value()
            total_frames = int(export_fps * duration)
        else:
            total_frames = self.video.shape[0]

        # Initialize video writer
        writer = cv2.VideoWriter(file_path, fourcc, export_fps, (export_width, export_height))
        
        if not writer.isOpened():
            QtWidgets.QMessageBox.critical(self, "Export Error", 
                                         "Failed to create video writer. Check file path and format.")
            return

        # Show progress bar
        self.export_progress.setVisible(True)
        self.export_progress.setRange(0, total_frames)
        self.export_button.setText("Exporting...")
        self.export_button.setEnabled(False)

        try:
            # Get current visualization parameters
            scale = self.mag_spin.value()
            show_arrows = self.arrows_checkbox.isChecked()
            point_size = self.point_size_spin.value()
            
            for frame_idx in range(total_frames):
                # Update progress
                self.export_progress.setValue(frame_idx)
                QtWidgets.QApplication.processEvents()  # Keep UI responsive

                # Set the current frame
                if self.is_mode_shape:
                    self.current_frame = frame_idx % int(self.fps * self.time_per_period)
                    # Get base frame for mode shapes
                    base_frame = self.video[0]
                    
                    # Calculate time for sinusoidal motion
                    t = self.current_frame / self.fps
                    displ_raw = self.displacements
                    amplitude = np.abs(displ_raw)
                    phase = np.angle(displ_raw)
                    displ = scale * amplitude * np.sin(2 * np.pi * t - phase)
                else:
                    self.current_frame = frame_idx
                    base_frame = self.video[self.current_frame]
                    displ = self.displacements[:, self.current_frame, :] * scale

                # Create the export frame by scaling the video frame pixel-perfectly
                # Convert to RGB for proper color handling
                if len(base_frame.shape) == 2:  # Grayscale
                    frame_rgb = np.stack([base_frame, base_frame, base_frame], axis=2)
                else:
                    frame_rgb = base_frame
                
                # Scale up the frame without interpolation (nearest neighbor)
                export_frame = np.repeat(np.repeat(frame_rgb, pixel_scale, axis=0), pixel_scale, axis=1)
                
                # Calculate displaced points
                displaced_pts = self.grid + displ
                
                # Draw displacement visualization on the scaled frame
                if show_arrows:
                    # Draw arrows showing displacement
                    magnitudes = np.linalg.norm(displ, axis=1)
                    norm = mcolors.Normalize(vmin=0, vmax=self.disp_max*scale)
                    cmap = plt.colormaps[self.colormap]
                    
                    for i, (pt0, pt1, mag) in enumerate(zip(self.grid, displaced_pts, magnitudes)):
                        # Scale coordinates to export resolution
                        start_pt = (int(pt0[0] * pixel_scale), int(pt0[1] * pixel_scale))
                        end_pt = (int(pt1[0] * pixel_scale), int(pt1[1] * pixel_scale))
                        
                        # Get color for this magnitude
                        color = cmap(norm(mag))
                        color_bgr = tuple(int(255 * c) for c in color[2::-1])  # Convert RGB to BGR
                        
                        # Draw arrow line
                        cv2.line(export_frame, start_pt, end_pt, color_bgr, 
                                max(1, point_size * pixel_scale // 10))
                        
                        # Draw arrow head
                        cv2.circle(export_frame, end_pt, max(1, point_size * pixel_scale // 5), 
                                  color_bgr, -1)
                else:
                    # Draw points at displaced positions
                    for pt in displaced_pts:
                        center = (int(pt[0] * pixel_scale), int(pt[1] * pixel_scale))
                        cv2.circle(export_frame, center, max(1, point_size * pixel_scale // 5), 
                                  (0, 0, 255), -1)  # Red circles
                
                # Ensure the frame is in the correct format and size
                export_frame = np.clip(export_frame, 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if len(export_frame.shape) == 3:
                    export_frame_bgr = cv2.cvtColor(export_frame, cv2.COLOR_RGB2BGR)
                else:
                    export_frame_bgr = export_frame

                writer.write(export_frame_bgr)

            writer.release()
            
            QtWidgets.QMessageBox.information(self, "Export Complete", 
                                            f"Video exported successfully to:\n{file_path}\n"
                                            f"Resolution: {export_width}x{export_height} "
                                            f"(pixel scale: {pixel_scale}x)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", 
                                         f"An error occurred during export:\n{str(e)}")
        finally:
            # Restore original state
            self.current_frame = original_frame
            self.update_frame()
            if original_timer_active:
                self.timer.start()
            
            # Reset UI
            self.export_progress.setVisible(False)
            self.export_button.setText("Export Video")
            self.export_button.setEnabled(True)
            writer.release()



if __name__ == "__main__":
    n_frames, height, width = 200, 300, 400
    n_points = 100
    frames = np.random.randint(0, 255, size=(n_frames, height, width), dtype=np.uint8)
    
    # Test with regular time-series displacements
    displacements = 2 * (np.random.rand(n_points, n_frames, 2) - 0.5)
    
    # Test with mode shapes (2D array)
    grid = np.stack(np.meshgrid(np.linspace(50, 250, int(np.sqrt(n_points))),
                                np.linspace(50, 350, int(np.sqrt(n_points)))), axis=-1).reshape(-1, 2)[:n_points]
        
    # Create a simple mode shape (e.g., first bending mode)
    # displacements = np.zeros((n_points, 2))
    # for i, point in enumerate(grid):
    #     # Simple sinusoidal mode shape in y-direction
    #     displacements[i, 0] = 5 * np.sin(np.pi * point[0] / width)  # y displacement
    #     displacements[i, 1] = 0  # no x displacement
    
    # Test mode shape viewer
    ResultViewer(frames, displacements, grid)