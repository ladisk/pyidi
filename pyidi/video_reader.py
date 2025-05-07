"""
Module for reading video files from high-speed video recordings.

@author: Ivan Tomac (itomac@fesb.hr), Klemen Zaletelj (klemen.zaletelj@fs.uni-lj.si), Janko Slavič (janko.slavic@fs.uni-lj.si)
"""

import os
import pyMRAW
import numpy as np
import imageio.v3 as iio
import warnings

PHORTRON_HEADER_FILE = ['cih', 'cihx']
SUPPORTED_IMAGE_FORMATS = ['png', 'tif', 'tiff', 'bmp', 'jpg', 'jpeg', 'gif']
PYAV_SUPPORTED_VIDEO_FORMATS = ['avi', 'mkv', 'mp4', 'mov', 'm4v', 'wmv', 'webm', 'flv', 'ogg', 'ogv']
CHANNELS = {'R': 0, 'G': 1, 'B': 2}

class VideoReader:
    """
    Manages reading of high-speed video recordings. The video recording can be any
    of the supported file formats which includes image streams, video files or memory 
    map for "mraw" file format.
    
    This applies to frames from image and video file formats:
    Reader returns the frame as a monochrome image. For colour images the "Y" (luma)
    is default channel, but other channels can be selected ("R", "G", "B", "Y"). 
    The reader returns image in 2D "numpy.array" ("height, width") of type 
    "numpy.uint8" or "numpy.uint16" depending on the bit depth of the image file,
    e.g. 12 bit depth images are returned as "numpy.array" of type "numpy.uint16".
    """
    def __init__(self, input_file, root=None, fps=None):
        """
        The video recording is initialized by providing the path to the image/video file, 
        "cih(x)" file from Photron, or "numpy.ndarray". For image stream it is enough to 
        provide the path to the any image file in the sequence. Images in stream must be 
        in the same directory and named in the way that can be sorted in the correct order,
        e.g. for stream of 10000 images file names should be: "im_0000.ext, ..., im_9999.ext".
        Image formats that support multiple images, such as "gif", "tif" are supported too.
        Upgrade is needed to enable higher bit depth then 8 bit for video file formats.

        :param input_file: path to the image/video or "cih(x)" file
        :type input_file: str
        :param root: root directory of the image/video file. Only used when the 
            input file is a "np.ndarray". Defaults to None.
        :type root: str
        :param fps: frames per second. If None and Photron file is passed, the fps 
            is read from the cih/cihx file. Defaults to None.
        :type fps: int or None
        """
        if fps:
            fps = int(fps)
            
        self.fps = fps


        if isinstance(input_file, np.ndarray):
            if root is None:
                raise ValueError('Root directory must be provided for np.ndarray input file!')
            
            self.root = root
            if not os.path.exists(self.root): # Create the folder if it does not exist
                os.mkdir(self.root)

            self.file_format = 'np.ndarray'
            self.input_file = 'ndarray'
            self.name = 'ndarray_video'

        elif isinstance(input_file, str):
            if not os.path.exists(input_file):
                raise FileNotFoundError(f'File "{input_file}" not found!')
            
            self.root, self.file = os.path.split(input_file)
            self.file_format = self.file.split('.')[-1].lower()
            self.input_file = input_file
            self.name = self.file.split('.')[0]


        if self.file_format in PHORTRON_HEADER_FILE:
            self.mraw, info = pyMRAW.load_video(input_file)
            self.N = info['Total Frame']
            self.image_width = info['Image Width']
            self.image_height = info['Image Height']
            if self.fps is None:
                self.fps = int(info['Record Rate(fps)'])
            self.info = info
        
        elif self.file_format in SUPPORTED_IMAGE_FORMATS:
            image_prop = iio.improps(input_file)
            self.image_meta = iio.immeta(input_file, plugin='pyav')
            if image_prop.n_images is None:
                self.is_n_images = False
                sc_dir = os.scandir(self.root)
                self.frame_files = [f.name for f in sc_dir \
                                    if f.name.endswith(self.file_format) \
                                        or f.name.endswith(self.file_format.upper())]
                self.N = len(self.frame_files)
                self.image_width = image_prop.shape[1]
                self.image_height = image_prop.shape[0]
            else:
                self.is_n_images = True
                self.N = image_prop.n_images
                self.image_width = image_prop.shape[2]
                self.image_height = image_prop.shape[1]
        
        elif self.file_format in PYAV_SUPPORTED_VIDEO_FORMATS:
            video_prop = iio.improps(input_file, plugin='pyav')
            self.N = video_prop.n_images
            self.image_width = video_prop.shape[2]
            self.image_height = video_prop.shape[1]

        elif self.file_format == 'np.ndarray':
            self.mraw = input_file
            self.N = input_file.shape[0]
            self.image_width = input_file.shape[2]
            self.image_height = input_file.shape[1]

        else:
            raise ValueError('Unsupported file format!')

        return None
    
    def get_frame(self, frame_number, *args):
        """
        Returns the "frame_number"-th frame from the video. Frames from image and video
        files are checked for the bit depth and converted to 8 or 16 bit depth if needed.
        The frames from "numpy.ndarray" and "mraw" files are returned as they are.

        :param frame_number: frame number
        :type frame_number: int
        :param args: additional arguments to be passed to the image readers to handle multiple channels in image
        :return: image (monochrome)
        """
        if not 0 <= frame_number < self.N:
            raise ValueError('Frame number exceeds total frame number!')

        if self.file_format in PHORTRON_HEADER_FILE or self.file_format == 'np.ndarray':
            image = self.mraw[frame_number]

        elif self.file_format in SUPPORTED_IMAGE_FORMATS:
            image = self._get_frame_from_image(frame_number, *args)

        elif self.file_format in PYAV_SUPPORTED_VIDEO_FORMATS:
            image = self._get_frame_from_video_file(frame_number, *args)

        return image

    def _get_frame_from_image(self, frame_number, use_channel='Y'):
        """Reads the frame from the image stream, or image file containing multiple images. 
        Colour images are assumed to be in "RGB(A)" format and they are automatically converted
        to "YUV" and "Y"(luma) channel is used. The 'use_channel' parameter can be used to select
        other the channels. The supported channels are R, G, B, Y (luma).

        :param frame_number: frame number
        :type frame_number: int
        :param use_channel: "R", "G", "B", "Y" (luma), defaults to "Y"
        :return: image (monochrome)
        """
        if self.is_n_images:
            input_file = os.path.join(self.root, self.file)
            image = iio.imread(input_file, index=frame_number, plugin='pyav', format=self.image_meta['video_format'])
        else:
            input_file = os.path.join(self.root, self.frame_files[frame_number])
            image = iio.imread(input_file, index=0, plugin='pyav', format=self.image_meta['video_format'])
        
        im_bit_depth = int(np.ceil(np.log2(image.max())))
        if im_bit_depth <= 8 and image.dtype != np.uint8:
            image = np.asarray(image, dtype=np.uint8)
        elif 8 < im_bit_depth <= 16 and image.dtype != np.uint16:
            image = np.asarray(image, dtype=np.uint16)
        elif im_bit_depth <= 8 and image.dtype == np.uint8:
            pass
        elif 8 < im_bit_depth <= 16 and image.dtype == np.uint16:
            pass
        else:
            raise ValueError('image format is not 8 or 16 bit depth! Image format: {}'.format(image.dtype))

        
        if len(image.shape) == 2:
            pass
        
        elif use_channel.upper() == 'Y':
            image = _rgb2luma(image[:, :, :3])
        
        elif use_channel.upper() in CHANNELS.keys():
            image = image[:, :, CHANNELS.get(use_channel.upper())]
        
        else:
            raise ValueError('Unsupported channel! Only R, G, B, Y are supported.')
        
        return image
    
    def _get_frame_from_video_file(self, frame_number, use_channel='Y'):
        """Reads the frame from the video file which is supported by the
        "imagio.v3" "pyav" plug-in.

        :param frame_number: frame number
        :type frame_number: int
        :param use_channel: "R", "G", "B", "Y" (luma), defaults to "Y"
        :return: monochrome image in 8 bit depth (note: needs upgrade to support higher bit depth)
        """
        input_file = os.path.join(self.root, self.file)
        if use_channel == 'Y':
            image = iio.imread(input_file, index=frame_number, plugin='pyav', format='yuv444p')
            image = image.transpose(1, 2, 0)
            image = image[:, :, 0]

        elif use_channel.upper() in CHANNELS.keys():
            image = iio.imread(input_file, index=frame_number, plugin='pyav')
            image = image[:, :, CHANNELS.get(use_channel.upper())]
        
        else:
            raise ValueError('Unsupported channel! Only R, G, B and Y are supported.')

        return image
    
    def close(self):
        """
        Close the video and clear the resources.
        In case of a MRAW video, closes the memory map for "mraw" file format.
        """
        if hasattr(self, 'mraw') and self.file_format in PHORTRON_HEADER_FILE:
            self.mraw._mmap.close()
            del self.mraw

    def gui(self):
        """Starts the GUI for pyIDI."""
        raise NotImplementedError('GUI is not implemented yet. Stay tuned!')
        # from . import gui
        # self.gui_obj = gui.gui(self)


def _rgb2luma(rgb_image):
    """Converts "RGB" image to "YUV" and returns only "Y" (luma) component.

    :param rgb_image: "RGB" image "(w, h, channels)"
    :type rgb_image: numpy.array
    :return: luma image
    """
    T = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    
    yuv_image = np.dot(rgb_image, T)
    y = np.asarray(np.around(yuv_image[:, :, 0]), dtype=rgb_image.dtype)

    return y