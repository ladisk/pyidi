import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.path import Path

SELECTION_MODES = {
    'ROI grid': 0,
    'Deselect ROI polygon': 1,
    'Only polygon': 2,
    'Manual ROI select': 3
    }

MODE_DESCRIPTION = {
    0: 'Use SHIFT + LEFT CLICK\nto select a polygon where\na regular grid of ROIs will\nbe generated.',
    1: 'Use SHIFT + LEFT CLICK\nto select a polygon where\nthe ROIs will be removed.',
    2: 'Use SHIFT + LEFT CLICK\nto select a polygon.',
    3: 'Use SHIFT + LEFT CLICK\nto manually position ROIs.'
}

class SubsetSelection:
    def __init__(self, video=None, roi_size=(11, 11), noverlap=0, polygon=None):
        self.verbose = 0
        self.shift_is_held = False

        self.roi_size = roi_size
        self.noverlap = int(noverlap)
        self.cent_dist_0 = self.roi_size[0] - self.noverlap
        self.cent_dist_1 = self.roi_size[1] - self.noverlap

        if polygon is None:
            self.polygon = [[], []]
        else:
            self.polygon = polygon
        self.deselect_polygon = [[], []]
        self.points = [[], []]

        root = tk.Tk()
        root.title('Selection')

        self.show_box = tk.IntVar(value=1)

        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.geometry(f'{int(0.9*self.screen_width)}x{int(0.9*self.screen_height)}')

        # Create left frame for options
        left_frame = tk.Frame(root, width=int(0.2 * self.screen_width))
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        left_frame.grid_propagate(False)

        # Add options to the left frame
        self.options = SelectOptions(left_frame, self)

        # Create main frame for the canvas and controls
        main_frame = tk.Frame(root)
        main_frame.pack(side='right', fill='both', expand=1)

        button1 = ttk.Button(main_frame, text='Confirm selection', command=lambda: self.on_closing(root))
        button1.pack(side='top', pady=5)

        self.fig = Figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(False)
        self.ax.imshow(video.get_frame(0), cmap='gray')

        # Initiate polygon
        self.line, = self.ax.plot(self.polygon[1], self.polygon[0], 'C1.-')
        self.line_deselect, = self.ax.plot(self.deselect_polygon[1], self.deselect_polygon[0], 'k.-')
        self.line2, = self.ax.plot([], [], 'C0x')

        plt.show(block=False)

        # Embed figure in tkinter window
        canvas = FigureCanvasTkAgg(self.fig, main_frame)
        toolbar = NavigationToolbar2Tk(canvas, main_frame)
        toolbar.pack(side='top', fill='x') # First pack the toolbar (it should be on top)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1, padx=5, pady=5) # Then pack the canvas

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')

        # Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.update_variables()
        root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(root))
        tk.mainloop()


    def _mode_selection_polygon(self, get_rois=True):
        """Select polygon to compute the points based on ROI size and
        ROI overlap."""
        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    if self.polygon[0]:
                        del self.polygon[1][-1]
                        del self.polygon[0][-1]

                    self.polygon[1].append(int(np.round(event.xdata)))
                    self.polygon[0].append(int(np.round(event.ydata)))

                    if self.polygon[0]:
                        self.polygon[1].append(self.polygon[1][0])
                        self.polygon[0].append(self.polygon[0][0])
                        
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held:
                if self.verbose:
                    print('Deleted the last point...')
                del self.polygon[1][-2]
                del self.polygon[0][-2]

            self.line.set_xdata(self.polygon[1])
            self.line.set_ydata(self.polygon[0])
            self.fig.canvas.draw()

            if get_rois:
                self.plot_selection()
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def _mode_selection_deselect_polygon(self):
        """Select polygon to compute the points based on ROI size and
        ROI overlap."""
        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    if self.deselect_polygon[0]:
                        del self.deselect_polygon[1][-1]
                        del self.deselect_polygon[0][-1]

                    self.deselect_polygon[1].append(int(np.round(event.xdata)))
                    self.deselect_polygon[0].append(int(np.round(event.ydata)))

                    if self.deselect_polygon[0]:
                        self.deselect_polygon[1].append(self.deselect_polygon[1][0])
                        self.deselect_polygon[0].append(self.deselect_polygon[0][0])
                        
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held:
                if self.verbose:
                    print('Deleted the last point...')
                del self.deselect_polygon[1][-2]
                del self.deselect_polygon[0][-2]

            self.line_deselect.set_xdata(self.deselect_polygon[1])
            self.line_deselect.set_ydata(self.deselect_polygon[0])
            self.fig.canvas.draw()

            self.plot_selection()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def _mode_selection_manual_roi(self):
        """Select polygon to compute the points based on ROI size and
        ROI overlap."""
        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    self.points[0].append(int(np.round(event.ydata)))
                    self.points[1].append(int(np.round(event.xdata)))

            elif event.button == 3 and self.shift_is_held:
                del self.points[1][-1]
                del self.points[0][-1]

            self.fig.canvas.draw()
            self.plot_selection()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def on_key_press(self, event):
        """Function triggered on key press (shift)."""
        if event.key == 'shift':
            self.shift_is_held = True
    
    def on_key_release(self, event):
        """Function triggered on key release (shift)."""
        if event.key == 'shift':
            self.shift_is_held = False
    
    def update_variables(self):
        self.line2.set_xdata([])
        self.line2.set_ydata([])
        self.fig.canvas.draw()

        self.mode = self.options.combobox.get()
        if SELECTION_MODES[self.mode] == 0:
            self._disconnect_mpl_onclick()

            self._mode_selection_polygon()

            self.roi_size = [int(self.options.roi_entry_y.get()), int(self.options.roi_entry_x.get())]
            self.noverlap = int(self.options.noverlap_entry.get())

            self.cent_dist_0 = self.roi_size[0] - self.noverlap
            self.cent_dist_1 = self.roi_size[1] - self.noverlap

            self.plot_selection()
        
        elif SELECTION_MODES[self.mode] == 1:
            if len(self.points[0]) == 0:
                tk.messagebox.showwarning("Warning", "No points have been selected yet.")
            else:
                self._disconnect_mpl_onclick()

                self._mode_selection_deselect_polygon()
                self.plot_selection()

        elif SELECTION_MODES[self.mode] == 2:
            self._disconnect_mpl_onclick()
            self._mode_selection_polygon(get_rois=False)

        elif SELECTION_MODES[self.mode] == 3:
            self._disconnect_mpl_onclick()
            self._mode_selection_manual_roi()
            self.roi_size = [int(self.options.roi_entry_y.get()), int(self.options.roi_entry_x.get())]
            self.plot_selection()
        
        else:
            raise Exception('Non existing mode...')

        self.options.description.configure(text=MODE_DESCRIPTION[SELECTION_MODES[self.mode]])
    
    def _disconnect_mpl_onclick(self):
        try:
            self.fig.canvas.mpl_disconnect(self.cid)
        except:
            pass

    def plot_selection(self):
        if len(self.polygon[0]) > 2 and len(self.polygon[1]) > 2:

            self.points = get_roi_grid(self.polygon, self.roi_size, self.noverlap, self.deselect_polygon).T

        if len(self.points[0]) >= 1 and len(self.points[1]) >= 1:
            self.line2.set_xdata(np.array(self.points).T[:, 1])
            self.line2.set_ydata(np.array(self.points).T[:, 0])
            
            self.options.nr_points_label.configure(text=f'{len(np.array(self.points).T)}')

            # if SELECTION_MODES[self.mode] == 3:
            if self.show_box.get():
                [p.remove() for p in reversed(self.ax.patches)]
                self.rectangles = []
                for i, (p0, p1) in enumerate(zip(self.points[0], self.points[1])):
                    self.rectangles.append(patches.Rectangle((p1 - self.roi_size[1]/2, p0 - self.roi_size[0]/2), 
                                                    self.roi_size[1], self.roi_size[0], fill=False, color='C2', linewidth=2))
                    self.ax.add_patch(self.rectangles[-1])
            else:
                [p.remove() for p in reversed(self.ax.patches)]

            self.fig.canvas.draw()

    def clear_selection(self):
        self.polygon = [[], []]
        self.deselect_polygon = [[], []]
        self.points = [[], []]
        self.options.nr_points_label.configure(text='0')
        self.clear_plot()
    
    def clear_plot(self):
        self.line.set_xdata([])
        self.line.set_ydata([])
        self.line_deselect.set_xdata([])
        self.line_deselect.set_ydata([])
        self.line2.set_xdata([])
        self.line2.set_ydata([])
        [p.remove() for p in reversed(self.ax.patches)]
        self.fig.canvas.draw()

    def on_closing(self, root):
        self.points = np.array(self.points)
        if self.points.shape[0] == 2:
            self.points = self.points.T
        root.destroy()

    def __repr__(self):
        return f"SubsetSelection(roi_size={self.roi_size}, noverlap={self.noverlap}, n_points={len(self.points)})"

class SelectOptions:
    def __init__(self, parent_frame, parent: SubsetSelection):
        self.running_options = True
        self.parent = parent

        roi_x = tk.StringVar(parent_frame, value=str(parent.roi_size[1]))
        roi_y = tk.StringVar(parent_frame, value=str(parent.roi_size[0]))
        noverlap = tk.StringVar(parent_frame, value=str(parent.noverlap))

        row = 0
        ttk.Label(parent_frame, text='Selection mode:').grid(row=row, column=0, padx=5, pady=5, sticky='W')
        self.combobox = ttk.Combobox(parent_frame, values=list(SELECTION_MODES.keys()))
        self.combobox.current(0)
        self.combobox.grid(row=row, column=1, sticky='wens', padx=5, pady=5)
        self.combobox.bind("<<ComboboxSelected>>", self.apply) # Auto apply when changing mode

        row = 1
        ttk.Label(parent_frame, text='Horizontal ROI size').grid(row=row, column=0, sticky='E')
        self.roi_entry_x = tk.Entry(parent_frame, textvariable=roi_x)
        self.roi_entry_x.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 2
        ttk.Label(parent_frame, text='Vertical ROI size').grid(row=row, column=0, sticky='E')
        self.roi_entry_y = tk.Entry(parent_frame, textvariable=roi_y)
        self.roi_entry_y.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 3
        ttk.Label(parent_frame, text='Overlap pixels').grid(row=row, column=0, sticky='E')
        self.noverlap_entry = tk.Entry(parent_frame, textvariable=noverlap)
        self.noverlap_entry.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 4
        ttk.Label(parent_frame, text='Show ROI box').grid(row=row, column=0, sticky='E')
        self.show_box_checkbox = tk.Checkbutton(parent_frame, text='', variable=self.parent.show_box)
        self.show_box_checkbox.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 5
        apply_button = ttk.Button(parent_frame, text='Apply', command=parent.update_variables)
        apply_button.grid(row=row, column=0, sticky='we', padx=5, pady=5)

        clear_button = ttk.Button(parent_frame, text='Clear', command=parent.clear_selection)
        clear_button.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        row = 6
        ttk.Label(parent_frame, text='Number of selected points:').grid(row=row, column=0, sticky='E')
        self.nr_points_label = ttk.Label(parent_frame, text='0')
        self.nr_points_label.grid(row=row, column=1, sticky='W')

        row = 7
        ttk.Label(parent_frame, text=' ').grid(row=row, column=0)

        row = 8
        self.description = ttk.Label(parent_frame, text='Description')
        self.description.grid(row=row, column=0, columnspan=2, pady=5)

    def apply(self, *args):
        self.parent.update_variables()
    
    def on_closing(self):
        self.running_options = False
        self.parent.update_variables()
        self.root1.destroy()


def get_roi_grid(polygon_points, roi_size, noverlap, deselect_polygon):
    if len(roi_size) != 2:
        raise Exception(f'roi_size must be a tuple of length 2')

    cent_dist_0 = roi_size[0] - noverlap
    cent_dist_1 = roi_size[1] - noverlap

    points = np.array(polygon_points)
    if points.shape[0] == 2:
        points = points.T

    low_0 = np.min(points[:, 0])
    high_0 = np.max(points[:, 0])
    low_1 = np.min(points[:, 1])
    high_1 = np.max(points[:, 1])

    candidates_0 = np.arange(low_0, high_0, cent_dist_0)
    candidates_1 = np.arange(low_1, high_1, cent_dist_1)
    candidates = np.concatenate([_.flatten()[:, None] for _ in np.meshgrid(candidates_0, candidates_1)], axis=1)

    path = Path(points)
    mask = path.contains_points(candidates)

    if len(deselect_polygon[0]) and len(deselect_polygon[1]):
        path_deselect = Path(np.array(deselect_polygon).T)
        mask_deselect = path_deselect.contains_points(candidates)
        mask = np.logical_and(mask, np.logical_not(mask_deselect))

    return np.round(candidates[mask]).astype(int)




