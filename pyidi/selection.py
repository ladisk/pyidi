import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class ROISelect:
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

        root = tk.Tk()
        root.title('Selection')

        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.geometry(f'{int(0.9*self.screen_width)}x{int(0.9*self.screen_height)}')

        self.options = SelectOptions(root, self)
        button1 = tk.Button(root, text='Open options', command=lambda: self.open_options(root))
        button1.pack(side='top')

        button2 = tk.Button(root, text='Confirm selection', command=root.destroy)
        button2.pack(side='top')

        self.fig = Figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(False)
        self.ax.imshow(video.mraw[0], cmap='gray')

        # Initiate polygon
        self.line, = self.ax.plot(self.polygon[1], self.polygon[0], 'r.-')
        self.line2, = self.ax.plot([], [], 'bo')

        plt.show(block=False)

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(self.fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1, padx=5, pady=5)
        NavigationToolbar2Tk(canvas, root)
        
        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')

        # Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.update_variables()

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
        
        self.fig.canvas.mpl_connect('button_press_event', onclick)

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
        if self.mode == 'ROI grid':
            self._mode_selection_polygon()

            self.roi_size = [int(self.options.roi_entry_y.get()), int(self.options.roi_entry_x.get())]
            self.noverlap = int(self.options.noverlap_entry.get())

            self.cent_dist_0 = self.roi_size[0] - self.noverlap
            self.cent_dist_1 = self.roi_size[1] - self.noverlap

            self.plot_selection()

        elif self.mode == 'Only polygon':
            self._mode_selection_polygon(get_rois=False)
        else:
            raise Exception('Non existing mode...')
        
    def plot_selection(self):
        if len(self.polygon[0]) > 2 and len(self.polygon[1]) > 2:
            self.points = get_roi_grid(self.polygon, self.roi_size, self.noverlap)
            if len(self.points):
                self.line2.set_xdata(self.points[:, 1])
                self.line2.set_ydata(self.points[:, 0])
                self.fig.canvas.draw()

    def clear_selection(self):
        self.polygon = [[], []]
        self.points = [[], []]
        self.clear_plot()
    
    def clear_plot(self):
        self.line.set_xdata([])
        self.line.set_ydata([])
        self.line2.set_xdata([])
        self.line2.set_ydata([])
        self.fig.canvas.draw()

    def open_options(self, root):
        if not self.options.running_options:
            self.options = SelectOptions(root, self)
        else:
            self.options.root1.lift()

class SelectOptions:
    def __init__(self, root, parent):
        self.running_options = True
        self.parent = parent

        self.root1 = tk.Toplevel(root)
        self.root1.title('Selection options')
        self.root1.geometry(f'{int(0.2*parent.screen_width)}x{int(0.5*parent.screen_height)}')

        roi_x = tk.StringVar(self.root1, value=str(parent.roi_size[1]))
        roi_y = tk.StringVar(self.root1, value=str(parent.roi_size[0]))
        noverlap = tk.StringVar(self.root1, value=str(parent.noverlap))

        row = 0
        tk.Label(self.root1, text='Selection mode:').grid(row=row, column=0)
        self.combobox = ttk.Combobox(self.root1, values = [
            'ROI grid',
            'Only polygon', 
            ])
        self.combobox.current(0)
        self.combobox.grid(row=row, column=1, sticky='wens', padx=5, pady=5)

        row = 1
        tk.Label(self.root1, text='Horizontal ROI size').grid(row=row, column=0, sticky='E')
        self.roi_entry_x = tk.Entry(self.root1, textvariable=roi_x)
        self.roi_entry_x.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 2
        tk.Label(self.root1, text='Vertical ROI size').grid(row=row, column=0, sticky='E')
        self.roi_entry_y = tk.Entry(self.root1, textvariable=roi_y)
        self.roi_entry_y.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 3
        tk.Label(self.root1, text='Overlap pixels').grid(row=row, column=0, sticky='E')
        self.noverlap_entry = tk.Entry(self.root1, textvariable=noverlap)
        self.noverlap_entry.grid(row=row, column=1, padx=5, pady=5, sticky='W')

        row = 4
        apply_button = tk.Button(self.root1, text='Apply', command=parent.update_variables)
        apply_button.grid(row=row, column=0, sticky='we', padx=5, pady=5)

        clear_button = tk.Button(self.root1, text='Clear', command=parent.clear_selection)
        clear_button.grid(row=row, column=1, sticky='w', padx=5, pady=5)

        self.root1.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        self.running_options = False
        self.parent.update_variables()
        self.root1.destroy()



def get_roi_grid(polygon_points, roi_size, noverlap):
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
    
    rois = []
    for i in range(low_0+cent_dist_0, high_0, cent_dist_0):
        for j in range(low_1+cent_dist_1, high_1, cent_dist_1):
            if inside_polygon(i, j, points):
                rois.append([i, j])
    return np.asarray(rois)


def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / \
                            (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside