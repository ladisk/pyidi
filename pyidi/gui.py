from napari.components import viewer_model
import numpy as np
import napari
from magicgui import magicgui
import pathlib

import warnings
warnings.simplefilter("default")

from . import tools
from . import selection

NO_METHOD = '---'
add_vertical_stretch = True

available_gui_methods = [
    (NO_METHOD, NO_METHOD),
    ('Simplified Optical Flow', 'sof'),
    ('Lucas-Kanade', 'lk'),
    ]

# The default values that are shown in GUI are defined
# These are updated every time the gui is opened
default_values = {
    'method_name': NO_METHOD,
    'subset_size': 3,
    'vertical_subset_size': 9,
    'horizontal_subset_size': 9,
    'noverlap': 0,
    'show_subset_box': True,
    'convert_from_px': 1.0,
    'mean_n_neighbours': 0,
    'zero_shift': False,
    'progress_bar': True,
    'reference_range_from': 0,
    'reference_range_to': 100,
    'pad': 2,
    'max_nfev': 20,
    'tol': 1e-8,
    'int_order': 3,
    'verbose': 1,
    'show_pbar': True,
    'reference_image': 0,
    'mraw_range_full': True,
    'mraw_range_from': 0,
    'mraw_range_to': 100,
    'mraw_range_step': 1,
    'processes': 1
}


def gui(self):
    # Update default values
    for k in default_values:
        if k in self.__dict__.keys():
            default_values[k] = self.__dict__[k]
        try:
            if k in self.method.__dict__.keys():
                default_values[k] = self.method.__dict__[k]
        except:
            pass
        
    try:
        self.reader.mraw
    except:
        raise ValueError('Only `mraw` files are supported for GUI at the moment.')
    
    if hasattr(self, 'method'):
        if hasattr(self.method, 'mraw_range'):
            if type(self.method.mraw_range) == str:
                if self.method.mraw_range == 'full':
                    default_values['mraw_ranage_full'] = True
                    default_values['mraw_range_from'] = 0
                    default_values['mraw_range_to'] = self.reader.mraw.shape[0]
                    default_values['mraw_range_step'] = 1
            else:
                default_values['mraw_ranage_full'] = False
                default_values['mraw_range_from'] = self.method.mraw_range[0]
                default_values['mraw_range_to'] = self.method.mraw_range[1]
                if len(self.method.mraw_range) == 3:
                    default_values['mraw_range_step'] = self.method.mraw_range[2]
                else:
                    default_values['mraw_range_step'] = 1
            
    # Start gui
    viewer = napari.Viewer(title='pyIDI interface')
    image_layer = viewer.add_image(self.reader.mraw)

    if not hasattr(self, 'method_name'):
        self.method_name = NO_METHOD

    if hasattr(self, 'points'):
        points_layer = viewer.add_points(self.points, size=1, edge_color='white', face_color='coral', symbol='cross', name='Points')
    
        if hasattr(self.method, 'roi_size') or hasattr(self.method, 'subset_size'):
            subset_layer = viewer.add_shapes(view_ROI(self), shape_type='rectangle', edge_width=0.2, edge_color='coral', face_color='#4169e164', opacity=0.8, name='Subsets')

    else:
        points_layer = viewer.add_points(name='Points', size=1, face_color='coral', symbol='cross')

    deselect_layer = viewer.add_shapes(name='Area Deselection', edge_color='red', face_color='#ffffff00')
    select_layer = viewer.add_shapes(name='Area Selection', edge_color='red', face_color='#ffffff00')


    @magicgui(
        call_button="Confirm method",
        Method = {'choices': [_[0] for _ in available_gui_methods]})
    def set_method_widget(Method=[_[0] for _ in available_gui_methods if _[1] == default_values['method_name']][0]):
        if Method != NO_METHOD:
            if self.method_name != Method:
                self.set_method(dict(available_gui_methods)[Method])

                try:
                    viewer.window.remove_dock_widget(self.PointsWidget)
                except:
                    pass
                
                try:
                    viewer.window.remove_dock_widget(self.ConfigWidget)
                except:
                    pass
                    
                try:
                    viewer.window.remove_dock_widget(self.DisplacementWidget)
                except:
                    pass

                if self.method_name == 'sof':
                    self.PointsWidget = viewer.window.add_dock_widget(sof_set_points_widget, name='Set points - SOF', add_vertical_stretch=add_vertical_stretch)
                elif self.method_name == 'lk':
                    self.PointsWidget = viewer.window.add_dock_widget(lk_set_points_widget, name='Set points - LK', add_vertical_stretch=add_vertical_stretch)

        else:
            warnings.warn('Select one of the methods first')
    

    @magicgui(call_button='Set points')
    def sof_set_points_widget(subset_size:int=default_values['subset_size'], noverlap:int=default_values['noverlap'], show_subset_box:bool=default_values['show_subset_box']):
        self.method.subset_size = subset_size
        base_set_points_widget(self, viewer, (subset_size, subset_size), noverlap, show_subset_box)
        self.ConfigWidget = viewer.window.add_dock_widget(sof_config_widget, name='Configure - SOF', add_vertical_stretch=add_vertical_stretch)

    
    @magicgui(call_button='Set points')
    def lk_set_points_widget(vertical_subset_size:int=default_values['vertical_subset_size'], horizontal_subset_size:int=default_values['horizontal_subset_size'], 
        noverlap:int=default_values['noverlap'], show_subset_box:bool=default_values['show_subset_box']):

        self.method.roi_size = (vertical_subset_size, horizontal_subset_size)
        base_set_points_widget(self, viewer, (vertical_subset_size, horizontal_subset_size), noverlap, show_subset_box)
        self.ConfigWidget = viewer.window.add_dock_widget(lk_config_widget, name='Configure - LK', add_vertical_stretch=add_vertical_stretch)


    @magicgui(call_button="Configure")
    def sof_config_widget(
        convert_from_px: float=default_values['convert_from_px'],
        mean_n_neighbours: int=default_values['mean_n_neighbours'],
        zero_shift: bool=default_values['zero_shift'],
        reference_range_from: int=default_values['reference_range_from'],
        reference_range_to: int=default_values['reference_range_to']
        ):

        #method configuration   
        self.method.configure(subset_size=self.method.subset_size,
                            convert_from_px=convert_from_px,
                            mean_n_neighbours=mean_n_neighbours,
                            zero_shift=zero_shift,
                            reference_range=(reference_range_from,reference_range_to))
        
        try:
            viewer.window.remove_dock_widget(self.DisplacementWidget)
        except:
            pass
        self.DisplacementWidget = viewer.window.add_dock_widget(displacement_widget, name='Calculate displacements', add_vertical_stretch=add_vertical_stretch)

    

    @magicgui(call_button="Configure", Tolerance={"choices": [1e-3,1e-5,1e-8,1e-10]})
    def lk_config_widget(
        processes: int=default_values['processes'],
        pad: int=default_values['pad'],
        max_nfev: int=default_values['max_nfev'],
        Tolerance=default_values['tol'],
        int_order: int=default_values['int_order'],
        #reference_range_from: int=0,
        #reference_range_to: int=100,
        mraw_range_full: bool=default_values['mraw_range_full'],
        mraw_range_from: int=default_values['mraw_range_from'],
        mraw_range_to: int=default_values['mraw_range_to'],
        mraw_range_step: int=default_values['mraw_range_step']):
            
        if mraw_range_full:
            mraw_range='full'
        else:
            mraw_range=(mraw_range_from, mraw_range_to, mraw_range_step)

        self.method.configure(roi_size=self.method.roi_size,
                            pad=pad,
                            max_nfev=max_nfev,        
                            tol=Tolerance, 
                            int_order=int_order, 
                            processes=processes,   
                            #reference_image=(reference_range_from, reference_range_to),
                            mraw_range=mraw_range 
                            )
                            
        try:
            viewer.window.remove_dock_widget(self.DisplacementWidget)
        except:
            pass
        self.DisplacementWidget = viewer.window.add_dock_widget(displacement_widget, name='Calculate displacements', add_vertical_stretch=add_vertical_stretch)



    @magicgui(call_button="Calculate displacements")
    def displacement_widget():
        self.get_displacements() #calculate displacements
        napari_show_disp_field(self, viewer)
    
    
    self.SetMethodWidget = viewer.window.add_dock_widget(set_method_widget, name='Method selection', add_vertical_stretch=add_vertical_stretch)
    if self.method_name == 'sof':
        self.PointsWidget = viewer.window.add_dock_widget(sof_set_points_widget, name='Set points - SOF', add_vertical_stretch=add_vertical_stretch)
        if hasattr(self, 'points'):
            self.ConfigWidget = viewer.window.add_dock_widget(sof_config_widget, name='Configure - SOF', add_vertical_stretch=add_vertical_stretch)
    elif self.method_name == 'lk':
        self.PointsWidget = viewer.window.add_dock_widget(lk_set_points_widget, name='Set points - LK', add_vertical_stretch=add_vertical_stretch)
        if hasattr(self, 'points'):
            self.ConfigWidget = viewer.window.add_dock_widget(lk_config_widget, name='Configure - LK', add_vertical_stretch=add_vertical_stretch)

    return viewer



def base_set_points_widget(self, viewer, subset_size, noverlap, show_subset_box):
    #individual points selection

    if viewer.layers['Area Selection'].data == []:
        self.points = np.round(viewer.layers['Points'].data).astype(int)
    
    #area selection for grid
    else:
        border = viewer.layers['Area Selection'].data[0].T # shape data
            
        if viewer.layers['Area Deselection'].data == []:
            deselect_border = [[],[]]
        else:     
            deselect_border = viewer.layers['Area Deselection'].data[0].T # deselection shape data

        self.points = selection.get_roi_grid(
            polygon_points=border, 
            roi_size=subset_size,
            noverlap=noverlap, 
            deselect_polygon=deselect_border) # get grid points


    if 'Subsets' in viewer.layers:
        viewer.layers.pop('Subsets') # refresh ROI layer

    if show_subset_box is True: #Show ROIs
        shapes_layer = viewer.add_shapes(view_ROI(self), shape_type='rectangle', edge_width=0.1, edge_color='coral', 
            face_color='#4169e164', opacity=0.8, name='Subsets')

    viewer.layers.pop('Points') #refresh grid layer
    viewer.add_points(self.points, size=1, face_color='coral', symbol='cross', name='Points')

    if len(self.points) == 0:
        del self.points
    
    try:
        viewer.window.remove_dock_widget(self.ConfigWidget)
    except:
        pass
        
    try:
        viewer.window.remove_dock_widget(self.DisplacementWidget)
    except:
        pass


def view_ROI(self): # view ROI boxes
    if hasattr(self.method, 'subset_size') or hasattr(self.method, 'roi_size'): #subset/roi layer
        if hasattr(self.method, 'subset_size'):
            v_half_subset = h_half_subset = self.method.subset_size/2

        elif hasattr(self.method, 'roi_size'):
            v_half_subset = self.method.roi_size[0]/2 #vertical
            h_half_subset = self.method.roi_size[1]/2 #horizontal


        rectangles = np.empty(shape=(len(self.points), 4, 2))
        for i in range(len(self.points)):
            rectangle = np.array([[self.points[i,0]-v_half_subset, self.points[i,1]-h_half_subset],
                                [self.points[i,0]-v_half_subset, self.points[i,1]+h_half_subset],
                                [self.points[i,0]+v_half_subset, self.points[i,1]+h_half_subset],
                                [self.points[i,0]+v_half_subset, self.points[i,1]-h_half_subset]])
                            
            rectangles[i] = rectangle
        
        return rectangles


def napari_show_disp_field(self, viewer):
    if hasattr(self, 'displacements'):
        vectors_all = np.empty((0,2,3))
        for i in range(len(self.points)):
            vectors = np.zeros((len(self.reader.mraw),2,3), dtype=float)
            vectors[:,0,0] = np.arange(len(self.reader.mraw))
            vectors[:,1,0] = np.arange(len(self.reader.mraw))
            vectors[:,0,1:] = self.points[i]
            vectors[:,1,1:] = self.displacements[i]

            vectors_all = np.append(vectors_all, vectors, axis=0)
        
        if self.method_name == 'lk':
                scale = self.method.roi_size[0]/(2*np.max(self.displacements))

        elif self.method_name == 'sof':
                scale = self.method.subset_size/(2*np.max(self.displacements))
        try:
            viewer.layers.pop('Displacement Field')
        except:
            pass
        viewer.add_vectors(vectors_all, length=0, name='Displacement Field')
        viewer.layers['Displacement Field'].length = scale
