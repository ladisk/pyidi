import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

class RegionSelectViewBox(pg.ViewBox):
    """Custom ViewBox that handles region selection with mouse events."""
    
    def __init__(self, parent_viewer):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.region_start = None
        self.region_current = None
        self.dragging = False
        
    def mousePressEvent(self, ev):
        if (self.parent_viewer.region_selection_active and 
            ev.button() == QtCore.Qt.MouseButton.LeftButton and
            ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            # Start region selection
            self.region_start = self.mapSceneToView(ev.scenePos())
            self.dragging = True
            ev.accept()
        else:
            super().mousePressEvent(ev)
            
    def mouseMoveEvent(self, ev):
        if self.dragging and self.parent_viewer.region_selection_active:
            # Update region selection
            self.region_current = self.mapSceneToView(ev.scenePos())
            self.parent_viewer.update_region_selection(self.region_start, self.region_current)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)
            
    def mouseReleaseEvent(self, ev):
        if (self.dragging and self.parent_viewer.region_selection_active and
            ev.button() == QtCore.Qt.MouseButton.LeftButton):
            # Finish region selection
            self.region_current = self.mapSceneToView(ev.scenePos())
            self.parent_viewer.finish_region_selection(self.region_start, self.region_current)
            self.dragging = False
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

class Viewer(QtWidgets.QMainWindow):
    def __init__(self, video, displacements=None, points=None, fps=30, magnification=1, point_size=10, colormap="cool"):
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

        # Check if displacements are provided
        if displacements is not None:
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
            self.disp_max = np.max(np.abs(displacements))
        else:
            self.displacements = None
            self.grid = None
            self.is_mode_shape = False
            self.disp_max = 0

        self.fps = fps
        self.magnification = magnification
        self.points_size = point_size
        self.current_frame = 0
        self.colormap = colormap

        # Region selection variables
        self.region_selection_active = False
        self.region_start_point = None
        self.region_end_point = None
        self.region_rect = None
        self.region_overlay = None
        self.selected_region = None  # (x, y, width, height) in image coordinates

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
        
        # Create custom viewbox for region selection
        self.viewbox = RegionSelectViewBox(self)
        self.view.addItem(self.viewbox)
        
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

        # Region selection controls
        export_layout.addWidget(QtWidgets.QLabel("Region Selection:"))
        
        region_layout = QtWidgets.QHBoxLayout()
        
        # Region selection button
        self.region_select_button = QtWidgets.QPushButton("Select Region")
        self.region_select_button.setCheckable(True)
        self.region_select_button.clicked.connect(self.toggle_region_selection)
        region_layout.addWidget(self.region_select_button)
        
        # Clear region button
        self.clear_region_button = QtWidgets.QPushButton("Clear")
        self.clear_region_button.clicked.connect(self.clear_region_selection)
        self.clear_region_button.setEnabled(False)
        region_layout.addWidget(self.clear_region_button)
        
        export_layout.addLayout(region_layout)
        
        # Region info label
        self.region_info_label = QtWidgets.QLabel("Full frame will be exported")
        self.region_info_label.setStyleSheet("font-size: 10px; color: #aaa;")
        export_layout.addWidget(self.region_info_label)

        # Frame range controls (only for non-mode shape videos)
        if not self.is_mode_shape:
            self.frame_range_label = QtWidgets.QLabel("Frame Range:")
            export_layout.addWidget(self.frame_range_label)
            
            frame_range_layout = QtWidgets.QHBoxLayout()
            
            # Start frame
            self.start_frame_spin = QtWidgets.QSpinBox()
            self.start_frame_spin.setRange(0, self.video.shape[0] - 1)
            self.start_frame_spin.setValue(0)
            self.start_frame_spin.setFixedWidth(80)
            self.start_frame_spin.valueChanged.connect(self.on_start_frame_changed)
            frame_range_layout.addWidget(self.start_frame_spin)
            
            # Stop frame
            self.stop_frame_spin = QtWidgets.QSpinBox()
            self.stop_frame_spin.setRange(0, self.video.shape[0] - 1)
            self.stop_frame_spin.setValue(self.video.shape[0] - 1)
            self.stop_frame_spin.setFixedWidth(80)
            self.stop_frame_spin.valueChanged.connect(self.on_stop_frame_changed)
            frame_range_layout.addWidget(self.stop_frame_spin)
            
            export_layout.addLayout(frame_range_layout)
            
            # Update the label with initial frame count
            self.update_frame_range_label()
            
            # Full range checkbox
            self.full_range_checkbox = QtWidgets.QCheckBox("Full Range")
            self.full_range_checkbox.setChecked(True)  # Initially checked since defaults are full range
            self.full_range_checkbox.stateChanged.connect(self.on_full_range_checkbox_changed)
            export_layout.addWidget(self.full_range_checkbox)

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

    def on_start_frame_changed(self, value):
        # Ensure start frame is not greater than stop frame
        if hasattr(self, 'stop_frame_spin') and value > self.stop_frame_spin.value():
            self.stop_frame_spin.setValue(value)
        
        # Update the frame range label
        self.update_frame_range_label()
        
        # Update checkbox state based on whether we have full range
        self.update_full_range_checkbox_state()

    def on_stop_frame_changed(self, value):
        # Ensure stop frame is not less than start frame
        if hasattr(self, 'start_frame_spin') and value < self.start_frame_spin.value():
            self.start_frame_spin.setValue(value)
        
        # Update the frame range label
        self.update_frame_range_label()
        
        # Update checkbox state based on whether we have full range
        self.update_full_range_checkbox_state()

    def update_frame_range_label(self):
        """Update the frame range label with current frame count."""
        if not self.is_mode_shape and hasattr(self, 'frame_range_label'):
            start_frame = self.start_frame_spin.value()
            stop_frame = self.stop_frame_spin.value()
            total_frames = stop_frame - start_frame + 1
            self.frame_range_label.setText(f"Frame Range: ({total_frames} frames)")

    def on_full_range_checkbox_changed(self, state):
        """Handle full range checkbox state changes."""
        if not self.is_mode_shape:
            if state == QtCore.Qt.CheckState.Checked.value:
                # Set to full range
                self.start_frame_spin.blockSignals(True)
                self.stop_frame_spin.blockSignals(True)
                self.start_frame_spin.setValue(0)
                self.stop_frame_spin.setValue(self.video.shape[0] - 1)
                self.start_frame_spin.blockSignals(False)
                self.stop_frame_spin.blockSignals(False)
                
                # Update the frame range label
                self.update_frame_range_label()

    def update_full_range_checkbox_state(self):
        """Update the checkbox state based on current spinbox values."""
        if not self.is_mode_shape and hasattr(self, 'full_range_checkbox'):
            is_full_range = (self.start_frame_spin.value() == 0 and 
                           self.stop_frame_spin.value() == self.video.shape[0] - 1)
            
            # Block signals to prevent recursive calls
            self.full_range_checkbox.blockSignals(True)
            self.full_range_checkbox.setChecked(is_full_range)
            self.full_range_checkbox.blockSignals(False)

    def set_full_range(self):
        """Set the frame range to cover the full video."""
        if not self.is_mode_shape:
            self.start_frame_spin.setValue(0)
            self.stop_frame_spin.setValue(self.video.shape[0] - 1)

    def toggle_region_selection(self):
        """Toggle region selection mode."""
        self.region_selection_active = self.region_select_button.isChecked()
        
        if self.region_selection_active:
            self.region_select_button.setText("Cancel Selection")
            self.region_select_button.setStyleSheet("background-color: #d73a00;")
            # Clear any existing region
            self.clear_region_graphics()
        else:
            self.region_select_button.setText("Select Region")
            self.region_select_button.setStyleSheet("")
            # Clear any temporary selection graphics
            self.clear_region_graphics()

    def clear_region_selection(self):
        """Clear the current region selection."""
        self.selected_region = None
        self.clear_region_graphics()
        self.clear_region_button.setEnabled(False)
        self.region_info_label.setText("Full frame will be exported")
        
        # Reset the selection button if it was active
        if self.region_selection_active:
            self.region_select_button.setChecked(False)
            self.toggle_region_selection()

    def clear_region_graphics(self):
        """Remove region selection graphics from the view."""
        if self.region_rect is not None:
            self.viewbox.removeItem(self.region_rect)
            self.region_rect = None
        if self.region_overlay is not None:
            self.viewbox.removeItem(self.region_overlay)
            self.region_overlay = None

    def update_region_selection(self, start_point, current_point):
        """Update the region selection rectangle during dragging."""
        if start_point is None or current_point is None:
            return
            
        # Clear previous rectangle
        if self.region_rect is not None:
            self.viewbox.removeItem(self.region_rect)
            
        # Create new rectangle
        x1, y1 = start_point.x(), start_point.y()
        x2, y2 = current_point.x(), current_point.y()
        
        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Create rectangle item
        self.region_rect = pg.RectROI([x_min, y_min], [x_max - x_min, y_max - y_min], 
                                      pen=pg.mkPen(color='red', width=2), 
                                      movable=False, removable=False)
        self.viewbox.addItem(self.region_rect)

    def finish_region_selection(self, start_point, end_point):
        """Finish region selection and apply overlay."""
        if start_point is None or end_point is None:
            return
            
        # Calculate region bounds
        x1, y1 = start_point.x(), start_point.y()
        x2, y2 = end_point.x(), end_point.y()
        
        # Ensure proper ordering and clip to image bounds
        video_height, video_width = self.video[0].shape
        x_min = max(0, min(x1, x2))
        x_max = min(video_width, max(x1, x2))
        y_min = max(0, min(y1, y2))
        y_max = min(video_height, max(y1, y2))
        
        # Store the selected region
        self.selected_region = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        
        # Update UI
        self.region_select_button.setChecked(False)
        self.toggle_region_selection()
        self.clear_region_button.setEnabled(True)
        self.region_info_label.setText(f"Region: {self.selected_region[2]}x{self.selected_region[3]} pixels")
        
        # Create overlay effect
        self.create_region_overlay()

    def create_region_overlay(self):
        """Create a semi-transparent overlay outside the selected region."""
        if self.selected_region is None:
            return
            
        # Clear existing overlay
        if self.region_overlay is not None:
            self.viewbox.removeItem(self.region_overlay)
            
        # Create overlay using ImageItem with alpha channel
        video_height, video_width = self.video[0].shape
        overlay = np.zeros((video_height, video_width, 4), dtype=np.uint8)
        
        # Set alpha to 128 (semi-transparent) for the entire overlay
        overlay[:, :, 3] = 128
        
        # Make the selected region fully transparent
        x, y, w, h = self.selected_region
        overlay[y:y+h, x:x+w, 3] = 0
        
        # Create ImageItem for overlay
        self.region_overlay = pg.ImageItem(overlay.transpose((1, 0, 2)))
        self.viewbox.addItem(self.region_overlay)

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
        # If no displacements, just show the video
        if self.displacements is None:
            frame = self.video[self.current_frame]
            self.img_item.setImage(frame.T)
            return

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

        # Ensure region overlay is on top if it exists
        if self.region_overlay is not None:
            self.viewbox.removeItem(self.region_overlay)
            self.viewbox.addItem(self.region_overlay)

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
        
        # Handle region selection
        if self.selected_region is not None:
            region_x, region_y, region_width, region_height = self.selected_region
            export_width = region_width * pixel_scale
            export_height = region_height * pixel_scale
        else:
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
            start_frame = 0
            stop_frame = total_frames - 1
        else:
            start_frame = self.start_frame_spin.value()
            stop_frame = self.stop_frame_spin.value()
            total_frames = stop_frame - start_frame + 1

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
                    # For regular videos, use the actual frame index within the specified range
                    actual_frame_idx = start_frame + frame_idx
                    self.current_frame = actual_frame_idx
                    base_frame = self.video[actual_frame_idx]
                    displ = self.displacements[:, actual_frame_idx, :] * scale

                # Create the export frame by scaling the video frame pixel-perfectly
                # Convert to RGB for proper color handling
                if len(base_frame.shape) == 2:  # Grayscale
                    frame_rgb = np.stack([base_frame, base_frame, base_frame], axis=2)
                else:
                    frame_rgb = base_frame
                
                # Apply region cropping if selected
                if self.selected_region is not None:
                    region_x, region_y, region_width, region_height = self.selected_region
                    frame_rgb = frame_rgb[region_y:region_y+region_height, region_x:region_x+region_width]
                
                # Scale up the frame without interpolation (nearest neighbor)
                export_frame = np.repeat(np.repeat(frame_rgb, pixel_scale, axis=0), pixel_scale, axis=1)
                
                # Calculate displaced points
                displaced_pts = self.grid + displ
                
                # Calculate region offset for coordinate transformation
                region_offset_x = 0
                region_offset_y = 0
                if self.selected_region is not None:
                    region_offset_x, region_offset_y = self.selected_region[0], self.selected_region[1]
                
                # Draw displacement visualization on the scaled frame
                if show_arrows:
                    # Draw arrows showing displacement
                    magnitudes = np.linalg.norm(displ, axis=1)
                    norm = mcolors.Normalize(vmin=0, vmax=self.disp_max*scale)
                    cmap = plt.colormaps[self.colormap]
                    
                    for i, (pt0, pt1, mag) in enumerate(zip(self.grid, displaced_pts, magnitudes)):
                        # Apply region offset and scale coordinates to export resolution
                        start_pt = (int((pt0[0] - region_offset_x) * pixel_scale), 
                                   int((pt0[1] - region_offset_y) * pixel_scale))
                        end_pt = (int((pt1[0] - region_offset_x) * pixel_scale), 
                                 int((pt1[1] - region_offset_y) * pixel_scale))
                        
                        # Check if points are within the export frame bounds
                        if (0 <= start_pt[0] < export_width and 0 <= start_pt[1] < export_height and
                            0 <= end_pt[0] < export_width and 0 <= end_pt[1] < export_height):
                            
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
                        center = (int((pt[0] - region_offset_x) * pixel_scale), 
                                 int((pt[1] - region_offset_y) * pixel_scale))
                        
                        # Check if point is within the export frame bounds
                        if (0 <= center[0] < export_width and 0 <= center[1] < export_height):
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
            
            # Create success message with frame range info
            if self.is_mode_shape:
                frame_info = f"Duration: {self.duration_spin.value():.1f}s"
            else:
                frame_info = f"Frames: {start_frame} to {stop_frame} ({total_frames} total)"
            
            # Add region info if applicable
            region_info = ""
            if self.selected_region is not None:
                region_info = f"Region: {self.selected_region[2]}x{self.selected_region[3]} pixels\n"
            
            QtWidgets.QMessageBox.information(self, "Export Complete", 
                                            f"Video exported successfully to:\n{file_path}\n"
                                            f"Resolution: {export_width}x{export_height} "
                                            f"(pixel scale: {pixel_scale}x)\n"
                                            f"{region_info}{frame_info}")

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

class ResultViewer(Viewer):
    pass

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
