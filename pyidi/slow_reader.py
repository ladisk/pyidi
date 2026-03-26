"""
Reader for Pahrsighted .slow high-speed camera format.

File layout
-----------
Offset  Size  Type        Description
0x00       4  char[4]     Magic: "SLOW"
0x04       4  uint32 LE   Format version (2)
0x08       8  uint64 LE   Offset to JSON metadata (36)
0x10       8  uint64 LE   JSON metadata size in bytes
0x18       8  uint64 LE   Offset to frame data (4096)
0x20       4  uint32 LE   Unknown (5)
0x24       *  UTF-8       JSON metadata string
  (zero-padded to frame data offset)

Frame layout (repeated fr_cnt times)
-------------------------------------
Offset  Size  Type        Description
0x00       4  uint32 LE   Seconds since year_offset (add meta["year_offset"] for Unix time)
0x04       4  uint32 LE   Nanoseconds within the second
0x08       4  uint32 LE   Flags | exp_clks (low byte)
0x0C      20  bytes       Reserved (zeros)
0x20       *  uint8[W*H]  Raw image pixels, row-major, W×H from acq.res, acq.bpp bits/pixel
"""

import json
import struct
from pathlib import Path

import numpy as np

_FILE_HEADER_FMT = "<4sIQQQI"  # magic, version, json_off, json_size, data_off, unk
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_FMT)  # 36


def _build_frame_dtype(height: int, width: int, bpp: int) -> np.dtype:
    img_dtype = np.uint8 if bpp <= 8 else np.uint16
    meta_dtype = np.dtype([
        ("ts_s_raw", "<u4"),       # seconds since year_offset
        ("ts_ns",    "<u4"),       # nanoseconds within the second
        ("flags",    "<u4"),       # bits 7-0 = exp_clks; bit 30 = sync flag
        ("_reserved", "u1", 20),
    ])
    return np.dtype([
        ("meta",  meta_dtype),
        ("image", img_dtype, (height, width)),
    ])


class SlowFile:
    """
    Memory-mapped reader for .slow files.

    The entire frame data is backed by a single numpy memmap.
    Accessing frames is zero-copy: slices return views into the mmap.

    :param path: path to the .slow file
    :type path: str or Path
    :param mode: memmap mode passed to ``numpy.memmap``, defaults to ``"r"``
    :type mode: str
    """

    def __init__(self, path: str | Path, mode: str = "r"):
        self.path = Path(path)
        self._parse_header()
        frame_dtype = _build_frame_dtype(self.height, self.width, self.bpp)
        self.frames: np.memmap = np.memmap(
            self.path,
            dtype=frame_dtype,
            mode=mode,
            offset=self._data_off,
            shape=(self.fr_cnt,),
        )
        self.images: np.ndarray = self.frames["image"]

    def _parse_header(self):
        with open(self.path, "rb") as f:
            raw = f.read(_FILE_HEADER_SIZE)
            magic, version, json_off, json_size, data_off, _ = struct.unpack(_FILE_HEADER_FMT, raw)
            assert magic == b"SLOW", f"Not a .slow file (magic={magic!r})"
            assert version == 2, f"Unsupported version {version}"
            f.seek(json_off)
            self.meta = json.loads(f.read(json_size))

        self._data_off = data_off
        w, h = self.meta["acq"]["res"].split("x")
        self.width = int(w)
        self.height = int(h)
        self.bpp = self.meta["acq"]["bpp"]
        self.fr_cnt = self.meta["fr_cnt"]
        self.fr_rate = self.meta["acq"]["fr_rate"]
        self.year_offset = self.meta["meta"]["year_offset"]

    def timestamps_ns(self) -> np.ndarray:
        """Returns Unix timestamps in nanoseconds for every frame.

        :return: array of Unix timestamps in nanoseconds, shape (N,)
        :rtype: numpy.ndarray
        """
        ts_s = self.frames["meta"]["ts_s_raw"].astype(np.int64) + self.year_offset
        ts_ns = self.frames["meta"]["ts_ns"].astype(np.int64)
        return ts_s * 1_000_000_000 + ts_ns

    def __len__(self) -> int:
        return self.fr_cnt

    def __getitem__(self, index):
        """Returns image array for the given frame index or slice (zero-copy view).

        :param index: frame index or slice
        :type index: int or slice
        :return: image array of shape ``(H, W)`` for int index or ``(N, H, W)`` for slice
        :rtype: numpy.ndarray
        """
        return self.images[index]

    def __repr__(self) -> str:
        return (
            f"SlowFile({self.path.name!r}, "
            f"{self.width}x{self.height} @ {self.fr_rate} fps, "
            f"{self.fr_cnt} frames, {self.bpp} bpp)"
        )
