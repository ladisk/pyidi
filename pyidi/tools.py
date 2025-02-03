import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from multiprocessing import Pool
from tqdm import tqdm
import numba as nb

class ManualROI:
    """Manual ROI selection."""

    def __init__(self, video, roi_size, single=False, verbose=0):
        """Manually select region of interest.
        
        :param video: parent object
        :type video: object
        :param roi_size: size of region of interest (dy, dx)
        :type roi_size: tuple, list
        :param single: if True, ONLY ONE ROI can be selected, defaults to False
        :type single: bool, optional
        :param verbose: Show text, defaults to 0
        :type verbose: int, optional
        """
        self.roi_size = roi_size
        self.image = video.reader.mraw[0]
        self.verbose = verbose

        # Tkinter root and matplotlib figure
        root = tk.Tk()
        root.title('Pick point')
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.imshow(self.image, cmap='gray')
        plt.show()

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, root)

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')


        self.point = [[], []]
        line, = ax.plot(self.point[1], self.point[0], 'r.')
        
        self.rectangles = []
        self.rectangles.append(patches.Rectangle((0, 0), 10, 10, fill=False, alpha=0))
        ax.add_patch(self.rectangles[-1])        
        
        self.shift_is_held = False
        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    if single:
                        self.point[0] = [int(np.round(event.ydata))]
                        self.point[1] = [int(np.round(event.xdata))]
                    else:
                        self.point[0].append(int(np.round(event.ydata)))
                        self.point[1].append(int(np.round(event.xdata)))
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held and not single:
                if self.verbose:
                    print('Deleted the last point...')
                del self.point[1][-1]
                del self.point[0][-1]
                del self.rectangles[-1]

            line.set_xdata(self.point[1])
            line.set_ydata(self.point[0])
            
            if self.point[0]:
                [p.remove() for p in reversed(ax.patches)]
                self.rectangles = []
                for i, (p0, p1) in enumerate(zip(self.point[0], self.point[1])):
                    self.rectangles.append(patches.Rectangle((p1 - self.roi_size[1]//2, p0 - self.roi_size[0]//2), 
                                                    self.roi_size[1], self.roi_size[0], fill=False, color='r', linewidth=2))
                    ax.add_patch(self.rectangles[-1])

            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.points = np.asarray(self.point).T
            if self.verbose:
                for i, point in enumerate(self.polygon):
                    print(f'{i+1}. point: x ={point[1]:5.0f}, y ={point[0]:5.0f}')

        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()
    

class GridOfROI:
    """
    Automatic simple ROI grid generation.

    Different from RegularROIGrid in that it gets a regular grid and only
    then checks if all points are inside polygon. This yields a more regular
    and full grid. Does not contain sssig filter.
    """
    def __init__(self, video=None, roi_size=(7, 7), noverlap=0, verbose=0):
        """
        
        :param video: parent object of video
        :type video: object
        :param roi_size: Size of the region of interest (y, x), defaults to (7, 7)
        :type roi_size: tuple, list, optional
        :param noverlap: number of pixels that overlap between neighbouring ROIs
        :type noverlap: int, optional
        :param sssig_filter: minimum value of SSSIG that the roi must have, defaults to None
        :type sssig_filter: None, float, optional
        :param verbose: Show text, defaults to 1
        :type verbose: int, optional
        """
        self.roi_size = roi_size
        self.verbose = verbose

        self.noverlap = int(noverlap)

        self.cent_dist_0 = self.roi_size[0] - self.noverlap
        self.cent_dist_1 = self.roi_size[1] - self.noverlap
        
        if video is not None:
            self.image = video.reader.mraw[0]
            self.pick_window()
        else:
            print('set the polygon points in self.polygon and call the `get_roi_grid` method')

    def pick_window(self):
        # Tkinter root and matplotlib figure
        root = tk.Tk()
        root.title('Pick points')
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.imshow(self.image, cmap='gray')
        plt.show()

        # Embed figure in tkinter winodw
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, root)

        # Initiate polygon
        self.polygon = [[], []]
        line, = ax.plot(self.polygon[1], self.polygon[0], 'r.-')

        if self.verbose:
            print('SHIFT + LEFT mouse button to pick a pole.\nRIGHT mouse button to erase the last pick.')

        self.shift_is_held = False

        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            if event.button == 1 and self.shift_is_held:
                if event.xdata is not None and event.ydata is not None:
                    self.polygon[1].append(int(np.round(event.xdata)))
                    self.polygon[0].append(int(np.round(event.ydata)))
                    if self.verbose:
                        print(f'y: {np.round(event.ydata):5.0f}, x: {np.round(event.xdata):5.0f}')

            elif event.button == 3 and self.shift_is_held:
                if self.verbose:
                    print('Deleted the last point...')
                del self.polygon[1][-1]
                del self.polygon[0][-1]

            line.set_xdata(self.polygon[1])
            line.set_ydata(self.polygon[0])
            fig.canvas.draw()

        def handle_close(event):
            """On closing."""
            self.polygon = np.asarray(self.polygon).T
            if self.verbose:
                for i, point in enumerate(self.polygon):
                    print(f'{i+1}. point: x ={point[1]:5.0f}, y ={point[0]:5.0f}')
            
            self.points = self.get_roi_grid()

        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()
    
    def get_roi_grid(self):
        points = self.polygon
        
        low_0 = np.min(points[:, 0])
        high_0 = np.max(points[:, 0])
        low_1 = np.min(points[:, 1])
        high_1 = np.max(points[:, 1])
        
        rois = []
        for i in range(low_0+self.cent_dist_0, high_0-self.cent_dist_0, self.cent_dist_0):
            for j in range(low_1+self.cent_dist_1, high_1-self.cent_dist_1, self.cent_dist_1):
                if inside_polygon(i, j, self.polygon):
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


def update_docstring(target_method, doc_method=None, delimiter='---', added_doc=''):
    """
    Update the docstring in target_method with the docstring from doc_method.
    
    :param target_method: The method that waits for the docstring
    :type target_method: method
    :param doc_method: The method that holds the desired docstring
    :type doc_method: method
    :param delimiter: insert the desired docstring between two delimiters, defaults to '---'
    :type delimiter: str, optional
    """
    docstring = target_method.__doc__.split(delimiter)

    leading_spaces = len(docstring[1].replace('\n', '')) - len(docstring[1].replace('\n', '').lstrip(' '))
    
    if doc_method is not None:
        if doc_method.__doc__:
            docstring[1] = doc_method.__doc__
        else:
            docstring[1] = '\n' + ' '*leading_spaces + \
                'The selected method does not have a docstring.\n'
    else:
        docstring[1] = added_doc.replace('\n', '\n' + ' '*leading_spaces)

    target_method.__func__.__doc__ = delimiter.join(docstring)


def split_points(points, processes):
    """Split the array of points to different processes.
    
    :param points: Array of points (2d)
    :type points: numpy array
    :param processes: number of processes
    :type processes: int
    """
    points = np.asarray(points)
    step = points.shape[0]//processes
    rest = points.shape[0]%processes
    points_split = []
    
    last_point = 0
    for i in range(processes):
        this_step = step
        if i < rest:
            this_step += 1
        points_split.append(points[last_point:last_point+this_step])
        last_point += this_step

    return points_split

# @nb.njit
def get_gradient(image):
    """Fast gradient computation.
    
    Compute the gradient of image in both directions using central
    difference weights over 3 points.
    
    !!! WARNING:
    The edges are excluded from the analysis and the returned image
    is smaller then original.

    :param image: 2d numpy array
    :return: gradient in x and y direction
    """
    im1 = image[2:]
    im2 = image[:-2]
    Gy = (im1 - im2)/2

    im1 = image[:, 2:]
    im2 = image[:, :-2]
    Gx = (im1 - im2)/2
        
    return Gx[1:-1], Gy[:, 1:-1]



    
                
        