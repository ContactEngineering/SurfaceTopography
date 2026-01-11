#
# Copyright 2025 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Reader for Nanofocus NMS files.
"""

import struct

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase

# Header layout with fixed offsets
HEADER_SIZE = 3468
OFFSET_Z = 16  # zmin, zmax (2 doubles)
OFFSET_POINTS = 1368  # nx, ny (2 unsigned ints)
OFFSET_SPACING = 1376  # dx, dy (2 doubles)


class NMSReader(ReaderBase):
    _format = "nms"
    _mime_types = ["application/x-nanofocus-nms"]
    _file_extensions = ["nms"]

    _name = "Nanofocus"
    _description = """
This reader imports Nanofocus NMS data files.
"""

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        # NMS files don't have a clear magic signature
        # We need to try parsing and see if it makes sense
        return MagicMatch.MAYBE

    def __init__(self, fobj):
        self._fobj = fobj

        with OpenFromAny(fobj, "rb") as f:
            # Read z range
            f.seek(OFFSET_Z)
            zmin, zmax = struct.unpack("<2d", f.read(16))

            # Read grid dimensions
            f.seek(OFFSET_POINTS)
            nx, ny = struct.unpack("<2I", f.read(8))

            # Read spacing (in mm)
            f.seek(OFFSET_SPACING)
            dx_mm, dy_mm = struct.unpack("<2d", f.read(16))

            # Validate dimensions and spacing
            if nx == 0 or ny == 0:
                raise ValueError("Invalid grid dimensions in NMS file")
            if dx_mm <= 0 or dy_mm <= 0:
                raise ValueError("Invalid spacing in NMS file")
            if nx > 100000 or ny > 100000:
                raise ValueError("Grid dimensions too large for NMS file")

            # Validate file size matches expected data size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            expected_data_size = nx * ny * 2  # uint16 data
            expected_file_size = HEADER_SIZE + expected_data_size
            if file_size < expected_file_size:
                raise ValueError(
                    f"File too small for NMS format: {file_size} < {expected_file_size}"
                )

            # Convert spacing from mm to µm
            dx = dx_mm * 1000.0
            dy = dy_mm * 1000.0

            self._nx = nx
            self._ny = ny
            self._zmin = zmin
            self._zmax = zmax

            # Physical sizes in micrometers
            physical_size_x = dx * nx
            physical_size_y = dy * ny

            self._metadata = {
                "zmin": zmin,
                "zmax": zmax,
                "dx_mm": dx_mm,
                "dy_mm": dy_mm,
            }

            self._channels = [
                ChannelInfo(
                    self,
                    0,
                    name="Default",
                    dim=2,
                    nb_grid_pts=(nx, ny),
                    physical_sizes=(physical_size_x, physical_size_y),
                    uniform=True,
                    unit="µm",
                    info={"raw_metadata": self._metadata},
                )
            ]

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")
        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        channel = self._channels[channel_index]

        with OpenFromAny(self._fobj, "rb") as f:
            f.seek(HEADER_SIZE)
            raw_data = np.frombuffer(
                f.read(self._nx * self._ny * 2), dtype=np.uint16
            ).reshape(self._ny, self._nx)

        # Scale uint16 data to height values
        # The data is stored as uint16 scaled between zmin and zmax
        data = (
            raw_data.astype(np.float64) / (2**16 - 2) * (self._zmax - self._zmin)
            + self._zmin
        )

        # Transpose to get (nx, ny) ordering
        data = data.T

        _info = dict(self._metadata)
        _info.update(info)

        if height_scale_factor is not None:
            return Topography(
                data * height_scale_factor,
                channel.physical_sizes,
                unit=channel.unit,
                info=_info,
                periodic=periodic,
            )

        return Topography(
            data,
            channel.physical_sizes,
            unit=channel.unit,
            info=_info,
            periodic=periodic,
        )

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
