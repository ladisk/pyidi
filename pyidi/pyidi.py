import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import pickle
import datetime
import json
import glob
import napari
from magicgui import magicgui
import warnings
warnings.simplefilter("default")

from .methods import IDIMethod, SimplifiedOpticalFlow, LucasKanade, LucasKanadeSc, LucasKanadeSc2, GradientBasedOpticalFlow, LucasKanade_1D
from .video_reader import VideoReader
from . import tools
from . import selection
from . import gui

available_method_shortcuts = [
    ('sof', SimplifiedOpticalFlow),
    ('lk', LucasKanade),
    ('lk_1D', LucasKanade_1D)
    # ('lk_scipy', LucasKanadeSc),
    # ('lk_scipy2', LucasKanadeSc2)
    # ('gb', GradientBasedOpticalFlow)
    ]


class pyIDI():
    """
    The pyIDI base class represents the video to be analysed.
    """
    def __init__(self, input_file, root=None):
        """Constructor of the pyIDI class.
        
        :param input_file: the video file to be analysed. Can be a name of the cih/cihx file, path to
            images directory, video file, or a 3D numpy array.
        :type input_file: str or np.ndarray
        :param root: root directory of the video file. Only used when the input file is a np.ndarray. Defaults to None.
        :type root: str
        """
        
        if type(input_file) in [str, np.ndarray]:
            self.reader = VideoReader(input_file, root=root)
            self.cih_file = input_file
        else:
            raise ValueError('`input_file` must be either a image/video/cih filename or a 3D array (N_time, height, width)')

        self.available_methods = dict([ 
            (key, {
                'IDIMethod': method,
                'description': method.__doc__,     
            })
            for key, method in available_method_shortcuts
        ])

        # Fill available methods into `set_method` docstring
        available_methods_doc = '\n' + '\n'.join([
            f"'{key}' ({method_dict['IDIMethod'].__name__}): {method_dict['description']}"
            for key, method_dict in self.available_methods.items()
            ])

        tools.update_docstring(self.set_method, added_doc=available_methods_doc)  


    def set_method(self, method, **kwargs):
        """
        Set displacement identification method on video.
        To configure the method, use `method.configure()`

        Available methods:
        ---
        [Available method names and descriptions go here.]
        ---

        :param method: the method to be used for displacement identification.
        :type method: IDIMethod or str
        """
        if isinstance(method, str) and method in self.available_methods.keys():
            self.method_name = method
            self.method = self.available_methods[method]['IDIMethod'](self, **kwargs)
        elif callable(method) and hasattr(method, 'calculate_displacements'):
            self.method_name = 'external_method'
            try:
                self.method = method(self, **kwargs)
            except:
                raise ValueError("The input `method` is not a valid `IDIMethod`.")
        else:
            raise ValueError("method must either be a valid name from `available_methods` or an `IDIMethod`.")
        
        # Update `get_displacements` docstring
        tools.update_docstring(self.get_displacements, self.method.calculate_displacements)
        # Update `show_points` docstring
        if hasattr(self.method, 'show_points'):
            try:
                tools.update_docstring(self.show_points, self.method.show_points)
            except:
                pass


    def set_points(self, points=None, method=None, **kwargs):
        """
        Set points that will be used to calculate displacements.
        If `points` is None and a `method` has aready been set on this `pyIDI` instance, 
        the `method` object's `get_point` is used to get method-appropriate points.
        """
        if points is None:
            if not hasattr(self, 'method'):
                if method is not None:
                    self.set_method(method)
                else:
                    raise ValueError("Invalid arguments. Please input points, or set the IDI method first.")
            self.method.get_points(self, **kwargs) # get_points sets the attribute video.points                
        else:
            self.points = np.asarray(points)


    def show_points(self, **kwargs):
        """
        Show selected points on image.
        """

        if hasattr(self, 'method') and hasattr(self.method, 'show_points'):
            self.method.show_points(self, **kwargs)
        else:
            figsize = kwargs.get('figsize', (15, 5))
            cmap = kwargs.get('cmap', 'gray')
            marker = kwargs.get('marker', '.')
            color = kwargs.get('color', 'r')
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(self.reader.get_frame(0).astype(float), cmap=cmap)
            ax.scatter(self.points[:, 1], self.points[:, 0], 
                marker=marker, color=color)
            plt.grid(False)
            plt.show()


    def show_field(self, field, scale=1., width=0.5):
        """
        Show displacement field on image.
        
        :param field: Field of displacements (number_of_points, 2)
        :type field: ndarray
        :param scale: scale the field, defaults to 1.
        :param scale: float, optional
        :param width: width of the arrow, defaults to 0.5
        :param width: float, optional
        """
        max_L = np.max(field[:, 0]**2 + field[:, 1]**2)

        fig, ax = plt.subplots(1)
        ax.imshow(self.reader.get_frame(0), 'gray')
        for i, ind in enumerate(self.points):
            f0 = field[i, 0]
            f1 = field[i, 1]
            alpha = (f0**2 + f1**2) / max_L
            if alpha < 0.2:
                alpha = 0.2
            plt.arrow(ind[1], ind[0], scale*f1, scale*f0, width=width, color='r', alpha=alpha)


    def get_displacements(self, autosave=True, **kwargs):
        """
        Calculate the displacements based on chosen method.

        Method docstring:
        ---
        Method is not set. Please use the `set_method` method.
        ---
        """
        if hasattr(self, 'method'):
            self.method.calculate_displacements(self, **kwargs)
            self.displacements = self.method.displacements
            
            # auto-save and clearing temp files
            if hasattr(self.method, 'process_number'):
                if self.method.process_number == 0:
                    
                    if autosave:
                        self.create_analysis_directory()
                        self.save(root=self.root_this_analysis)

                    self.method.clear_temp_files()
                    
            return self.displacements
        else:
            raise ValueError('IDI method has not yet been set. Please call `set_method()` first.')


    def close_video(self):
        """
        Close the .mraw video memmap.
        """
        self.reader.close()
    

    def create_analysis_directory(self):
        if type(self.cih_file) == str:
            cih_file_ = os.path.split(self.cih_file)[-1].split('.')[0]
        else:
            cih_file_ = 'ndarary_video'
        self.root_analysis = os.path.join(self.reader.root, f'{cih_file_}_pyidi_analysis')
        if not os.path.exists(self.root_analysis):
            os.mkdir(self.root_analysis)
        
        analyses = glob.glob(os.path.join(self.root_analysis, 'analysis_*/'))
        if analyses:
            last_an = sorted(analyses)[-1]
            print(last_an, last_an.split('\\')[-2])
            n = int(last_an.split('\\')[-2].split('_')[-1])
        else:
            n = 0
        self.root_this_analysis = os.path.join(self.root_analysis, f'analysis_{n+1:0>3.0f}')
        
        os.mkdir(self.root_this_analysis)

    
    def save(self, root=''):
        with open(os.path.join(root, 'results.pkl'), 'wb') as f:
            pickle.dump(self.displacements, f, protocol=-1)
        with open(os.path.join(root, 'points.pkl'), 'wb') as f:
            pickle.dump(self.points, f, protocol=-1)

        out = {
            'info': {
                'width': self.reader.image_width,
                'height': self.reader.image_height,
                'N': self.reader.N
            },
            'createdate': datetime.datetime.now().strftime("%Y %m %d    %H:%M:%S"),
            'cih_file': self.cih_file if type(self.cih_file) == str else 'ndarray',
            'settings': self.method.create_settings_dict(),
            'method': self.method_name
        }

        with open(os.path.join(root, 'settings.txt'), 'w') as f:
            json.dump(out, f, sort_keys=True, indent=2)

    
    def __repr__(self):
        
        rep = 'File name: ' + self.cih_file + ',\n' + \
        'Image width: ' + str(self.image_width) + ',\n' + \
        'Image height: ' + str(self.image_height) + ',\n' + \
        'Total frame: ' + str(self.N) + ',\n' + \
        'Record Rate(fps): ' + str(self.info['Record Rate(fps)'])
        
        if hasattr(self, 'method_name'):
            rep +=',\n' +  'Method: ' + self.method_name
                
            if hasattr(self.method, 'subset_size'):
                rep += ',\n' + 'Subset size: ' + str(self.method.subset_size)
                
            elif hasattr(self.method, 'roi_size'):
                 rep += ',\n' + 'ROI size: ' + str(self.method.roi_size)

        
        if hasattr(self, 'points'):
             rep +=',\n' + 'Number of points: ' + str(len(self.points))

        return rep
    
    def gui(self):
        self.gui_obj = gui.gui(self)

    @property
    def mraw(self):
        warnings.warn('`self.mraw` is deprecated and will be removed in the next version. Please use `self.reader.mraw` instead.', DeprecationWarning)
        return self.reader.mraw
    
    @property
    def info(self):
        #warnings.warn('`self.info` is deprecated and will be removed in the next version. Please use `self.reader.info` instead.', DeprecationWarning)
        return self.reader.info

#     def gui(self):
#         """Napari interface.
#         """
#         available_gui_methods = [
#             ('---', '---'),
#             ('Simplified Optical Flow', 'sof'),
#             ('Lucas-Kanade', 'lk'),
#             ]

#         if not hasattr(self, 'method_name'):
#             self.method_name = '---'

#         self.displacement_widget_shown = False

#         viewer = napari.Viewer(title='pyIDI interface') #launch viewer
#         layer = viewer.add_image(self.mraw) #add image layer
        
#         if hasattr(self, 'points'): #if points are given, add points layer
#             points_layer = viewer.add_points(self.points,size=1, edge_color='white', face_color='coral', symbol='cross', name='Points')
#             grid_points = self.points
            
#             if hasattr(self.method, 'roi_size') or hasattr(self.method, 'subset_size'): #if ROI is given, add ROI layer
#                 shapes_layer = viewer.add_shapes(tools.view_ROI(self), shape_type='rectangle', edge_width=0.1, edge_color='coral', face_color='#4169e164', opacity=0.8,name='ROI box')

#         else: #if there are no points given, launch point selector
#             points_layer=viewer.add_points(name='Points', size=1, face_color='coral', symbol='cross')
            
#         shapes_deselect = viewer.add_shapes(name='Area Deselection', edge_color='red', face_color='#ffffff00') #deselection layer
#         shapes = viewer.add_shapes(name='Area Selection', edge_color='red', face_color='#ffffff00') # selection layer

#         #Method selection widget
#         @magicgui(
#             call_button="Confirm method",
#             Method = {'choices': ['---'] + [_[0] for _ in available_gui_methods]})
#         def method_widget(Method=[_[0] for _ in available_gui_methods if _[1] == self.method_name][0]):
#             if Method != '---':
                
#                 if self.method_name != Method:
#                     self.set_method(dict(available_gui_methods)[Method])

#                     try:
#                         viewer.window.remove_dock_widget(self.ConfigWidget)
#                     except:
#                         pass
                    
#                     try:
#                         viewer.window.remove_dock_widget(self.DisplacementWidget)
#                     except:
#                         pass

#                     if self.method_name == 'sof':
#                         self.ConfigWidget = viewer.window.add_dock_widget(sof_config_widget, name='Method configuration - SOF')
#                     elif self.method_name == 'lk':
#                         self.ConfigWidget = viewer.window.add_dock_widget(lk_config_widget, name='Method configuration - LK')

#             else:
#                 warnings.warn('Select one of the methods first')
        
#         #sof configurator + point selection widget
#         @magicgui(call_button="Configure", Overlap_pixels={'min': -100 })
#         def sof_config_widget(
#             Subset_size: int=5,
#             Overlap_pixels: int=0,
#             Show_ROI_box: bool=False,
#             convert_from_px: float=1.,
#             mean_n_neighbours: int=0,
#             zero_shift: bool=False, 
#             reference_range_from: int=0,
#             reference_range_to: int=self.mraw.shape[0]
#             ):
            
#             #individual points selection
#             if viewer.layers['Area Selection'].data == []:
#                 grid_points = np.round(viewer.layers['Points'].data).astype(int)
            
#             #area selection for grid    
#             else:
#                 border = viewer.layers['Area Selection'].data[0].T #shape data
                    
#                 if viewer.layers['Area Deselection'].data == []:
#                     deselect_border = [[],[]]
#                 else:     
#                     deselect_border = viewer.layers['Area Deselection'].data[0].T #deselection shape data

#                 grid_points = selection.get_roi_grid(polygon_points=border, roi_size=(Subset_size,Subset_size),
#                      noverlap=Overlap_pixels, deselect_polygon=deselect_border) #get grid points

#             #method configuration   
#             self.method.configure(subset_size=Subset_size,
#                                 convert_from_px=convert_from_px,
#                                 mean_n_neighbours=mean_n_neighbours,
#                                 zero_shift=zero_shift,
#                                 reference_range=(reference_range_from,reference_range_to))

#             self.points = grid_points #export points data
#             if 'ROI box' in viewer.layers:
#                 viewer.layers.pop('ROI box') # refresh ROI layer

#             if Show_ROI_box is True: #Show ROI
#                 shapes_layer = viewer.add_shapes(tools.view_ROI(self), shape_type='rectangle', edge_width=0.1, edge_color='coral', face_color='#4169e164', opacity=0.8,name='ROI box')

#             viewer.layers.pop('Points') #refresh points layer
#             viewer.add_points(grid_points, size=1, face_color='coral', symbol='cross', name='Points')

#             if len(self.points) == 0: #delete points atribute if it is an empty array
#                 del self.points
            
#             if not self.displacement_widget_shown and self.points != []: #launch displacement widget
#                 viewer.window.add_dock_widget(displacement_widget, name='Displacements') 
#                 self.displacement_widget_shown = True
            
#         #lk configurator + point selection widget
#         @magicgui(
#             call_button="Confirm selection",
#             Overlap_pixels={'min': -1000},
#             Tolerance={"choices": [1e-3,1e-5,1e-8,1e-10]})
#         def lk_config_widget(
#             Horizontal_ROI_size: int=5,
#             Vertical_ROI_size: int=5,
#             Overlap_pixels: int=0,
#             Show_ROI_box: bool=False,
#             pad: int=2,
#             max_nfev: int=20,
#             Tolerance=1e-8,
#             int_order: int=3,
#             processes: int=1, 
#             #reference_range_from: int=0,
#             #reference_range_to: int=100,
#             mraw_range_full: bool=True,
#             mraw_range_from: int=0,
#             mraw_range_to: int=10,
#             mraw_range_step: int=1):

#             #individual points selection
#             if  viewer.layers['Area Selection'].data == []:
#                 grid_points = np.round(viewer.layers['Points'].data).astype(int)
            
#             #area selection
#             else:
#                 border = viewer.layers['Area Selection'].data[0].T #shape data
                    
#                 if viewer.layers['Area Deselection'].data == []:
#                     deselect_border = [[],[]]
#                 else:     
#                     deselect_border = viewer.layers['Area Deselection'].data[0].T #deselection shape data
                
#                 grid_points = selection.get_roi_grid(polygon_points=border, roi_size=(Vertical_ROI_size,Horizontal_ROI_size), 
#                     noverlap=Overlap_pixels, deselect_polygon=deselect_border) #get grid points
                
#             if mraw_range_full:
#                 mraw_range='full'
#             else:
#                 mraw_range=(mraw_range_from, mraw_range_to, mraw_range_step)

#             self.method.configure(roi_size=(Vertical_ROI_size, Horizontal_ROI_size),
#                                 pad=pad,
#                                 max_nfev=max_nfev,        
#                                 tol=Tolerance, 
#                                 int_order=int_order, 
#                                 processes=processes,   
#                                 #reference_image=(reference_range_from, reference_range_to),
#                                 mraw_range=mraw_range 
#                                 )

#             self.points = grid_points #export points data
#             if 'ROI box' in viewer.layers:
#                 viewer.layers.pop('ROI box') # refresh ROI layer

#             if Show_ROI_box is True: #Show ROI
#                 shapes_layer = viewer.add_shapes(tools.view_ROI(self), shape_type='rectangle', edge_width=0.1, edge_color='coral', 
#                     face_color='#4169e164', opacity=0.8,name='ROI box')

#             viewer.layers.pop('Points') #refresh grid layer
#             viewer.add_points(grid_points, size=1, face_color='coral', symbol='cross', name='Points')

#             if len(self.points) == 0:
#                 del self.points
            
#             if not self.displacement_widget_shown and self.points != []:
#                 self.DisplacementWidget=viewer.window.add_dock_widget(displacement_widget, name='Displacements') #launch displacement widget
#                 self.displacement_widget_shown = True

#         #Calculate displacement widget
#         @magicgui(call_button="Calculate displacements")
#         def displacement_widget():
#             self.get_displacements() #calculate displacements
#             napari_show_disp_field(self, viewer)

#         if hasattr(self, 'displacements'):
#             napari_show_disp_field(self, viewer)


#         # Show widgets
#         viewer.window.add_dock_widget(method_widget, name='Method selection')

#         # if self.method_name != '---':
#         #     viewer.window.add_dock_widget()


# def napari_show_disp_field(self, viewer):
#     if hasattr(self, 'displacements'):
#         vectors_all = np.empty((0,2,3))
#         for i in range(len(self.points)):
#             vectors = np.zeros((len(self.mraw),2,3), dtype=float)
#             vectors[:,0,0] = np.arange(len(self.mraw))
#             vectors[:,1,0] = np.arange(len(self.mraw))
#             vectors[:,0,1:] = self.points[i]
#             vectors[:,1,1:] = self.displacements[i]

#             vectors_all = np.append(vectors_all, vectors, axis=0)
        
#         if self.method_name == 'lk':
#                 scale = self.method.roi_size[0]/(2*np.max(self.displacements))

#         elif self.method_name == 'sof':
#                 scale = self.method.subset_size/(2*np.max(self.displacements))
#         try:
#             viewer.layers.pop('Displacement Field')
#         except:
#             pass
#         viewer.add_vectors(vectors_all, length=0, name='Displacement Field')
#         viewer.layers['Displacement Field'].length = scale
            
            