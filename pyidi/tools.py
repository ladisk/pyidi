import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class RegularROIGrid:
    def __init__(self, video, roi_size=(7, 7), noverlap=0, sssig_filter=None, verbose=1):
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
        if noverlap > np.min(np.asarray(self.roi_size)//2):
            print('!!!! WARNING: "noverlap" should be smaller than half of "roi_size"')

        self.noverlap = int(noverlap)

        self.sssig_filter = sssig_filter
        self.cent_dist_0 = self.roi_size[0] - self.noverlap
        self.cent_dist_1 = self.roi_size[1] - self.noverlap

        self.image = video.mraw[0]

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
            
            self.points, self.roi_quality = self.get_roi_grid()
            self.ROI_filter_SSSIG()


        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)
        # on closing the figure
        fig.canvas.mpl_connect('close_event', handle_close)

        root.mainloop()


    def get_roi_grid(self):
        """Get the points inside the polygon and SSSIG values that correspond to the ROI.
        """
        min_0 = np.min(self.polygon[:, 0])
        max_0 = np.max(self.polygon[:, 0])
        min_1 = np.min(self.polygon[:, 1])
        max_1 = np.max(self.polygon[:, 1])
        
        x = self.polygon[:, 1]
        y = self.polygon[:, 0]

        _polygon = np.asarray([x, y]).T
        
        sp_0 = []
        sp_1 = []
        roi_quality = []
        for p0 in range(min_0+1, max_0):
            for p1 in range(min_1+1, max_1):
                if inside_polygon(p0, p1, self.polygon):
                    if not sp_0 or (np.abs(p0 - sp_0[-1]) >= self.cent_dist_0 or np.abs(p1 - sp_1[-1] >= self.cent_dist_1)):
                        sp_0.append(p0)
                        sp_1.append(p1)
                        roi_quality.append(self._sssig(p0, p1))

        return np.column_stack((sp_0, sp_1)), np.asarray(roi_quality)


    def _sssig(self, p0, p1):
        """Determine SSSIG value for a ROI.
        """
        coor_0 = p0 + np.arange(-self.roi_size[0]//2, self.roi_size[0]//2)
        coor_1 = p1 + np.arange(-self.roi_size[1]//2, self.roi_size[1]//2)
        m0, m1 = np.meshgrid(coor_0, coor_1)
        g0, g1 = np.gradient(self.image[m0, m1])

        val = np.sqrt(np.sum(g0**2) + np.sum(g1**2)) / (self.roi_size[0]*self.roi_size[1])
        return val


    def ROI_filter_SSSIG(self):
        """Filter the points according to SSSIG value.
        """
        if self.sssig_filter is not None:
            f = float(self.sssig_filter)

            sel = (self.roi_quality > f)
            self.roi_quality = self.roi_quality[sel]
            self.points = self.points[sel]


    def show_grid(self, highlight=None, show_rois=True):
        """Show the image with ROIs and points.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.image, 'gray')

        polygon = np.concatenate((self.polygon, self.polygon[0:1]), axis=0)
        ax.plot(polygon[:, 1], polygon[:, 0], 'y')
        for i, p in enumerate(self.points):
            ax.plot(p[1], p[0], 'g.')
            if show_rois:
                ax.add_patch(patches.Rectangle(
                    (p[1]-self.roi_size[1]//2, p[0]-self.roi_size[0]//2), 
                    self.roi_size[1], self.roi_size[0], 
                    fill=False, color='g'))

        if highlight is not None:
            hL = int(highlight)
            ax.plot(self.points[hL, 1], self.points[hL, 0], 'ro')
            ax.add_patch(patches.Rectangle(
                (self.points[hL, 1]-self.roi_size[1]//2, self.points[hL, 0]-self.roi_size[0]//2), 
                self.roi_size[1], self.roi_size[0], 
                fill=False, color='r', linewidth=2))




def inside_polygon(x, y, points):
    """Return True if a coordinate (x, y) is inside a polygon defined by
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