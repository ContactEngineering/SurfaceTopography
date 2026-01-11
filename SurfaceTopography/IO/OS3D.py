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
Reader for Digital Metrology OmniSurf3D OS3D files.

Format documentation: https://digitalmetrology.com/omnisurf3d-file-format/
"""

import struct

import numpy as np

from ..Exceptions import FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase

# Invalid value marker (minimum float32 value)
INVALID_VALUE_THRESHOLD = -3e38


class OS3DReader(ReaderBase):
    _format = "os3d"
    _mime_types = ["application/x-omnisurf3d"]
    _file_extensions = ["os3d"]

    _name = "Digital Metrology OmniSurf3D"
    _description = """
This reader imports Digital Metrology OmniSurf3D data files.
"""

    _MAGIC = b"OmniSurf3D"

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE
        if buffer.startswith(cls._MAGIC):
            return MagicMatch.YES
        return MagicMatch.NO

    def __init__(self, fobj):
        self._fobj = fobj

        with OpenFromAny(fobj, "rb") as f:
            # Read and verify magic
            magic = f.read(len(self._MAGIC))
            if magic != self._MAGIC:
                raise FileFormatMismatch(
                    f"Expected magic '{self._MAGIC.decode()}', got '{magic.decode()}'"
                )

            # Read version info
            major_version, minor_version = struct.unpack("<ii", f.read(8))

            # Read identification string (variable length)
            (id_length,) = struct.unpack("<i", f.read(4))
            identification = f.read(id_length).decode("utf-8", errors="replace")

            # Read datetime string (variable length)
            (datetime_length,) = struct.unpack("<i", f.read(4))
            datetime_str = f.read(datetime_length).decode("utf-8", errors="replace")

            # Read grid dimensions and spacing
            nx, ny = struct.unpack("<ii", f.read(8))
            step_x, step_y = struct.unpack("<dd", f.read(16))
            origin_x, origin_y = struct.unpack("<dd", f.read(16))

            # Store data offset for later reading
            self._data_offset = f.tell()
            self._nx = nx
            self._ny = ny

            self._metadata = {
                "major_version": major_version,
                "minor_version": minor_version,
                "identification": identification,
                "datetime": datetime_str,
                "origin_x": origin_x,
                "origin_y": origin_y,
            }

            # Physical sizes in micrometers
            physical_size_x = step_x * nx
            physical_size_y = step_y * ny

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
            f.seek(self._data_offset)
            data = np.frombuffer(
                f.read(self._nx * self._ny * 4), dtype=np.float32
            ).reshape(self._ny, self._nx)

        # Mark invalid values as NaN
        data = data.astype(np.float64)
        data[data < INVALID_VALUE_THRESHOLD] = np.nan

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
