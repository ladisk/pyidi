"""
Module for reading video files from high-speed video recordings.

@author: Ivan Tomac (itomac@fesb.hr), Klemen Zaletelj (klemen.zaletelj@fs.uni-lj.si), Janko Slavič (janko.slavic@fs.uni-lj.si)
"""

import os
import warnings
import pyMRAW
import numpy as np
import imageio.v3 as iio
from . import slow_reader as _slow_reader

PHORTRON_HEADER_FILE = {"cih", "cihx"}
SLOW_FILE = {"slow"}
SUPPORTED_IMAGE_FORMATS = {"png", "tif", "tiff", "bmp", "jpg", "jpeg", "gif"}
PYAV_SUPPORTED_VIDEO_FORMATS = {
    "avi",
    "mkv",
    "mp4",
    "mov",
    "m4v",
    "wmv",
    "webm",
    "flv",
    "ogg",
    "ogv",
}
CHANNELS = {"R": 0, "G": 1, "B": 2}


class VideoReader:
    """
    Manages reading of high-speed video recordings. The video recording can be any
    of the supported file formats which includes image streams, video files or memory
    map for "mraw" file format.

    This applies to frames from image and video file formats:
    Reader returns the frame as a monochrome image. For colour images automatic conversion
    to grayscale (luma channel) is performed. 
    Other channels can be selected ("R", "G", "B") or custom weights can be applied.
    The reader returns image in 2D "numpy.array" ("height, width") of type
    "numpy.uint8" or "numpy.uint16" depending on the bit depth of the originalimage file.
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
            self.configure(fps=fps)

        if isinstance(input_file, np.ndarray):
            if root is None:
                raise ValueError(
                    "Root directory must be provided for np.ndarray input file!"
                )

            self.configure(root=root)

            self.file_format = "np.ndarray"
            self.input_file = "ndarray"
            self.name = "ndarray_video"

        elif isinstance(input_file, str):
            if not os.path.exists(input_file):
                raise FileNotFoundError(f'File "{input_file}" not found!')

            self.root, self.file = os.path.split(input_file)
            self.file_format = self.file.split(".")[-1].lower()
            self.input_file = input_file
            self.name = self.file.split(".")[0]

        if self.file_format in PHORTRON_HEADER_FILE:
            self._initialise_phortron_camera_files(input_file)

        elif self.file_format in SLOW_FILE:
            self._initialise_slow_files(input_file)

        elif self.file_format in SUPPORTED_IMAGE_FORMATS:
            self._initalise_image_files(input_file)

        elif self.file_format in PYAV_SUPPORTED_VIDEO_FORMATS:
            self._initialise_video_files(input_file)

        elif self.file_format == "np.ndarray":
            self._initialise_numpy_array(input_file)

        else:
            raise ValueError("Unsupported file format!")

    def configure(self, **kwargs):
        """Configure reader parameters after initialization.

        Supported keyword arguments:

        - ``fps`` *(int)*: Frames per second.
        - ``root`` *(str)*: Root directory for output or image sequence files.
          Used with "np.ndarray".The directory is created if it does not exist.
        - ``video_format`` *(str)*: PyAV pixel format string used when reading frames.
          (default: "gray", "gray16be", "gray16le"). For custom selection 
          of image channels, or using custom weights for conversion to monochrome,
          video format must be set to the RGB format, depending on the bit depth of 
          the original image file, e.g. "rgb24", "rgb48le", "rgb48be".
        - ``channel`` *(str)*: Colour channel to extract from RGB images.
          Must be one of "R", "G", "B".
        - ``channel_weights`` *(list or tuple)*: Three weights applied to the
          RGB channels to produce a monochrome image, e.g. luma coefficients
          ``[0.299, 0.587, 0.114]``.

        :param kwargs: Keyword arguments as described above.
        :type kwargs: dict
        :raises ValueError: If ``channel`` is not a string or
            ``channel_weights`` is not a list/tuple of length 3.
        """
        if "fps" in kwargs:
            self.fps = int(kwargs["fps"])

        if "root" in kwargs:
            if not isinstance(kwargs["root"], str):
                raise ValueError("Root must be a string!")
            self.root = kwargs["root"]
            if not os.path.exists(self.root):  # Create the folder if it does not exist
                os.mkdir(self.root)

        if "video_format" in kwargs:
            if not isinstance(kwargs["video_format"], str):
                raise ValueError("Video format must be a string!")
            self.video_format = kwargs["video_format"]

        if "channel" in kwargs:
            if not isinstance(kwargs["channel"], str):
                raise ValueError("Channel must be configured as a string!"
                                 " Only R, G, B are supported.")
            self.channel = kwargs["channel"].upper()
            if self.channel not in CHANNELS:
                raise ValueError("Unsupported channel! Only R, G and B are supported.")

        if "channel_weights" in kwargs:
            if not (
                isinstance(kwargs["channel_weights"], (list, tuple))
                and len(kwargs["channel_weights"]) == 3
            ):
                raise ValueError("Channel weights must be a list or tuple of length 3!")
            self.channel_weights = kwargs["channel_weights"]

    def get_frame(self, frame_number, *args, **kwargs):
        """
        Returns the "frame_number"-th frame from the video. Frames from image and video
        files are checked for the bit depth and converted to 8 or 16 bit depth if needed.
        The frames from "numpy.ndarray" and "mraw" files are returned as they are.

        :param frame_number: frame number
        :type frame_number: int
        :param args: additional arguments to be passed to the image readers to handle 
        multiple channels in image
        :param kwargs: additional keyword arguments forwarded to image/video reader methods
        :type kwargs: dict
        :return: image (monochrome)
        """
        if not 0 <= frame_number < self.N:
            raise ValueError("Frame number exceeds total frame number!")

        if (
            self.file_format in PHORTRON_HEADER_FILE
            or self.file_format in SLOW_FILE
            or self.file_format == "np.ndarray"
        ):
            image = self._frames[frame_number]

        elif self.file_format in SUPPORTED_IMAGE_FORMATS:
            image = self._get_frame_from_image(frame_number, *args, **kwargs)

        elif self.file_format in PYAV_SUPPORTED_VIDEO_FORMATS:
            image = self._get_frame_from_video_file(frame_number, *args, **kwargs)

        else:
            raise ValueError("Unsupported file format!")

        return image

    def get_frames(self, frame_range=None, *args, **kwargs):
        """Returns all the available frames.

        If "mraw" or "np.ndarray", it returns the frames as they are. If images
        or mp4, avi, etc., the ``get_frame`` method is called in a loop. In this
        case, the ``args`` are passed to the ``get_frame`` method (see the ``get_frame``
        method for details).

        :param frame_range: The range of the frames to return. If None, all frames are returned.
            If int, the frames from zero to ``frame_range`` are returned.
            If tuple, the frames from first to second index are returned.
        :type frame_range: tuple, list, int, None, optional
        :param args: positional arguments forwarded to ``get_frame``
        :param kwargs: keyword arguments forwarded to ``get_frame``
        :type kwargs: dict
        """
        if not isinstance(frame_range, (int, list, tuple, type(None))):
            raise ValueError(
                "Unsupported frame range! Supported types are int, list and tuple."
            )

        if isinstance(frame_range, (list, tuple)) and len(frame_range) != 2:
            raise ValueError("Length of the frame range must be 2!")

        if frame_range is None:
            frames_start = 0
            frames_end = self.N + 1
            n_frames = self.N
        elif isinstance(frame_range, int):
            frames_start = 0
            frames_end = frame_range
            n_frames = frame_range
        elif isinstance(frame_range, (list, tuple)):
            frames_start = frame_range[0]
            frames_end = frame_range[1]
            n_frames = frame_range[1] - frame_range[0]
        else:
            raise ValueError(
                "Unsupported frame range! Supported types are int, list and tuple."
            )

        if (
            self.file_format in PHORTRON_HEADER_FILE
            or self.file_format in SLOW_FILE
            or self.file_format == "np.ndarray"
        ):
            frames = self._frames[frames_start:frames_end]

        else:
            frames = np.zeros(
                (n_frames, self.image_height, self.image_width), dtype=int
            )
            for i in range(n_frames):
                frames[i] = self.get_frame(i + frames_start, *args, **kwargs)

        return frames

    def _get_frame_from_image(self, frame_number):
        """Reads the frame from the image stream, or image file containing multiple images.
        8bit and 16 bit images are supported. The bit depth is determined from the image
        properties. Color images are automatically converted to monochrome using a weighted
        sum with weights [0.299, 0.587, 0.114] by setting format to "gray" in ``iio.imread``.
        If the channel is configured as "R", "G" or "B", using a configuration method, 
        the corresponding channel is returned or if the custom channel weights are set
        using a configuration method, the weighted sum of the "RGB" channels is returned.

        :param frame_number: frame number
        :type frame_number: int
        :return: image (monochrome)
        """
        if self.is_n_images:
            input_file = os.path.join(self.root, self.file)
            image = iio.imread(
                input_file,
                index=frame_number,
                plugin="pyav",
                format=self.video_format
            )
        else:
            input_file = os.path.join(self.root, self.frame_files[frame_number])
            image = iio.imread(
                input_file,
                index=0,
                plugin="pyav",
                format=self.video_format
            )

        # im_bit_depth = int(np.ceil(np.log2(image.max())))
        # if im_bit_depth <= 8 and image.dtype != np.uint8:
        #     image = np.asarray(image, dtype=np.uint8)
        # elif 8 < im_bit_depth <= 16 and image.dtype != np.uint16:
        #     image = np.asarray(image, dtype=np.uint16)
        # elif im_bit_depth <= 8 and image.dtype == np.uint8:
        #     pass
        # elif 8 < im_bit_depth <= 16 and image.dtype == np.uint16:
        #     pass
        # else:
        #     raise ValueError(
        #         "image format is not 8 or 16 bit depth! Image format: {}".format(
        #             image.dtype
        #         )
        #     )

        if self.video_format in {'rgb24', 'rgb48le', 'rgb48be'}:
            if isinstance(
                getattr(self, "channel", None),
                str
                ):
                image = image[:, :, CHANNELS.get(self.channel)]

            elif isinstance(
                getattr(self, "channel_weights", None),
                (list, tuple)
                ):
                image = _rgb2mono(image, self.channel_weights)

            else:
                raise ValueError(
                    "Unsupported channel! Use configure method to set the channel "
                    "as a string (R, G, B) or channel weights as a list or tuple of length 3."
                )

        return image

    def _get_frame_from_video_file(self, frame_number):
        """Reads the frame from the video file which is supported by the
        "imagio.v3" "pyav" plug-in. Returns the frame as a monochrome image
        (sum with weights [0.299, 0.587, 0.114]). Custom channel selection 
        and custom weights for conversion to monochrome are supported by setting 
        ``channel`` or ``channel_weights`` using configuration method.

        :param frame_number: frame number
        :type frame_number: int
        :return: monochrome image in 8 bit depth (note: needs upgrade to support higher bit depth)
        """
        input_file = os.path.join(self.root, self.file)

        if isinstance(
                getattr(self, "channel", None),
                str
                ):
            image = iio.imread(
                input_file,
                index=frame_number,
                plugin="pyav"
            )
            image = image[:, :, CHANNELS.get(self.channel)]

        elif isinstance(
                getattr(self, "channel_weights", None),
                (list, tuple)
                ):
            image = iio.imread(
                input_file,
                index=frame_number,
                plugin="pyav"
            )
            image = _rgb2mono(image, self.channel_weights)

        else:
            image = iio.imread(
                input_file,
                index=frame_number,
                plugin="pyav",
                format="yuv444p"
            )
            image = image.transpose(1, 2, 0)
            image = image[:, :, 0]

        return image

    def _initialise_phortron_camera_files(self, input_file):
        """Initialise reader state for Photron ``cih/cihx`` sources.

        Loads frame data and metadata using ``pyMRAW`` and populates core
        attributes such as frame count, image size, fps and source metadata.

        :param input_file: Path to a Photron header file (``.cih`` or ``.cihx``)
        :type input_file: str
        """
        self._frames, info = pyMRAW.load_video(input_file)
        self.N = info["Total Frame"]
        self.image_width = info["Image Width"]
        self.image_height = info["Image Height"]
        if not getattr(self, "fps", False):
            self.configure(fps=info["Record Rate(fps)"])
        self.info = info

    def _initialise_slow_files(self, input_file):
        """Initialise reader state for ``.slow`` recordings.

        :param input_file: Path to a ``.slow`` file
        :type input_file: str
        """
        self._slow = _slow_reader.SlowFile(input_file)
        self._frames = self._slow.images
        self.N = self._slow.fr_cnt
        self.image_width = self._slow.width
        self.image_height = self._slow.height
        if not getattr(self, "fps", False):
            self.configure(fps=self._slow.fr_rate)
        self.info = self._slow.meta

    def _initalise_image_files(self, input_file):
        """Initialise reader state for image files and image sequences.

        Stores metadata from ``imageio`` in ``self.image_meta``. The
        ``self.image_meta[\"video_format\"]`` value is later used when reading
        frames via the ``pyav`` plugin to preserve the detected pixel format.

        :param input_file: Path to an image file from the sequence or a
            multi-image file
        :type input_file: str
        """
        image_prop = iio.improps(input_file)
        image_meta = iio.immeta(input_file, plugin="pyav")

        if image_prop.dtype == np.uint16:
            if image_meta.get("video_format") in {"gray16be", "gray16le"}:
                self.configure(video_format=image_meta.get("video_format"))
            else:
                self.configure(video_format="gray16be")
        elif image_prop.dtype == np.uint8:
            self.configure(video_format="gray")
        else:
            raise ValueError(
                f"Unsupported image bit depth {image_prop.dtype}! "
                "Only np.uint8 and np.uint16 formats are supported."
            )

        if image_prop.n_images is None:
            self.is_n_images = False
            sc_dir = os.scandir(self.root)
            self.frame_files = [
                f.name
                for f in sc_dir
                if f.name.endswith(self.file_format)
                or f.name.endswith(self.file_format.upper())
            ]
            self.frame_files.sort()
            self.N = len(self.frame_files)
            self.image_width = image_prop.shape[1]
            self.image_height = image_prop.shape[0]
        else:
            self.is_n_images = True
            self.N = image_prop.n_images
            self.image_width = image_prop.shape[2]
            self.image_height = image_prop.shape[1]

    def _initialise_video_files(self, input_file):
        """Initialise reader state for video containers handled by ``pyav``.

        :param input_file: Path to a supported video file
        :type input_file: str
        """
        video_prop = iio.improps(input_file, plugin="pyav")
        video_meta = iio.immeta(input_file, plugin="pyav")

        self.N = video_prop.n_images
        self.image_width = video_prop.shape[2]
        self.image_height = video_prop.shape[1]

        if not getattr(self, "fps", False):
            self.configure(fps=video_meta.get("fps", None))
        self.video_meta = video_meta

    def _initialise_numpy_array(self, input_file):
        """Initialise reader state when source frames are provided as ndarray.

        :param input_file: Frame stack with shape ``(N, H, W)``
        :type input_file: numpy.ndarray
        """
        self._frames = input_file
        self.N = input_file.shape[0]
        self.image_width = input_file.shape[2]
        self.image_height = input_file.shape[1]

    def close(self):
        """
        Close the video and clear the resources.
        In case of a MRAW video, closes the memory map for "mraw" file format.
        """
        if hasattr(self, "_frames") and self.file_format in PHORTRON_HEADER_FILE:
            self._frames._mmap.close()
            del self._frames
        elif hasattr(self, "_slow") and self.file_format in SLOW_FILE:
            del self._frames
            del self._slow

    def gui(self):
        """Starts the GUI for pyIDI."""
        raise NotImplementedError("GUI is not implemented yet. Stay tuned!")
        # from . import gui
        # self.gui_obj = gui.gui(self)

    @property
    def mraw(self):
        warnings.warn(
            'The "mraw" attribute has been deprecated. Use the "get_frames" method instead.'
        )
        return self.get_frames()


def _rgb2mono(rgb_image, weights):
    """Converts "RGB" image to sum weighted monochrome.

    :param rgb_image: "RGB" image "(w, h, channels)"
    :type rgb_image: numpy.array
    :param weights: conversion weights
    :type weights: tuple or list of length 3
    :return: weighted sum of the "RGB" channels
    """

    weighted_image = np.dot(rgb_image, weights)
    weighted_image_rounded = np.asarray(
        np.around(weighted_image),
        dtype=rgb_image.dtype
        )

    return weighted_image_rounded
