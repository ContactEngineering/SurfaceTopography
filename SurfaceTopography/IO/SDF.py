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
Reader for ISO 25178-71 SDF (Surface Data File) format.

This format is defined in ISO 25178-71 and supports both ASCII and binary
variants.
"""

import struct

import numpy as np

from ..Exceptions import (
    CorruptFile,
    FileFormatMismatch,
    MetadataAlreadyFixedByFile,
    UnsupportedFormatFeature,
)
from ..Support.UnitConversion import get_unit_conversion_factor
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase

# Magic bytes for ASCII and binary formats
MAGIC_ASCII = b"aISO-1.0"
MAGIC_BINARY = b"bISO-1.0"

# Fixed unit in SDF files is meters
FIXED_UNIT = "m"

# Data type mapping (from ISO 25178-71)
DTYPE_MAP = {
    5: np.dtype("<i2"),  # INT16
    6: np.dtype("<i4"),  # INT32
    7: np.dtype("<f8"),  # DOUBLE
}

# Invalid value markers for each data type
INVALID_VALUE_MAP = {
    5: -(2**15),  # INT16 minimum
    6: -(2**31),  # INT32 minimum
    7: np.nan,  # NaN for doubles
}

# Binary header layout (ISO 25178-71)
# 8 bytes magic + 10 bytes ManufacID + 12 bytes CreateDate + 12 bytes ModDate
# + 2 bytes NumPoints + 2 bytes NumProfiles + 4x8 bytes (Xscale, Yscale, Zscale, Zres)
# + 3 bytes (Compression, DataType, CheckType)
BINARY_HEADER_FORMAT = "<8s10s12s12sHHddddbbb"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)


def _parse_ascii_header(content):
    """Parse the ASCII header section."""
    header = {}
    type_map = {
        "ManufacID": str,
        "CreateDate": str,
        "ModDate": str,
        "NumPoints": int,
        "NumProfiles": int,
        "Xscale": float,
        "Yscale": float,
        "Zscale": float,
        "Zresolution": float,
        "Compression": int,
        "DataType": int,
        "CheckType": int,
    }

    for line in content.strip().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name in type_map:
                header[name] = type_map[name](value)

    return header


def _read_ascii_data(data_section, num_points, num_profiles, z_scale):
    """Parse the ASCII data section."""
    # Replace BAD markers with NaN
    data_section = data_section.replace("BAD", "NAN")

    # Parse data
    data = np.fromstring(data_section, sep=" ", dtype=np.float64)
    data = data.reshape(num_profiles, num_points)
    data *= z_scale

    return data


class SDFReader(ReaderBase):
    _format = "sdf"
    _mime_types = ["application/x-iso25178-sdf"]
    _file_extensions = ["sdf"]

    _name = "ISO 25178-71 SDF"
    _description = """
This reader imports ISO 25178-71 Surface Data File (SDF) format files.
Both ASCII and binary variants are supported.
"""

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 8:
            return MagicMatch.MAYBE
        if buffer.startswith(MAGIC_ASCII) or buffer.startswith(MAGIC_BINARY):
            return MagicMatch.YES
        return MagicMatch.NO

    def __init__(self, fobj):
        self._fobj = fobj

        with OpenFromAny(fobj, "rb") as f:
            magic = f.read(8)

            if magic == MAGIC_ASCII:
                self._is_binary = False
                self._parse_ascii_file(f)
            elif magic == MAGIC_BINARY:
                self._is_binary = True
                self._parse_binary_file(f)
            else:
                raise FileFormatMismatch(
                    f"Invalid SDF magic: expected 'aISO-1.0' or 'bISO-1.0', "
                    f"got '{magic.decode('ascii', errors='replace')}'"
                )

    def _parse_ascii_file(self, f):
        """Parse ASCII format SDF file."""
        # Read entire file as text
        f.seek(0)
        content = f.read().decode("ascii")

        # Split into sections (header, data, trailer) separated by *
        sections = content.split("*")
        if len(sections) < 3:
            raise CorruptFile("Invalid ASCII SDF file: missing section delimiters")

        header_section = sections[0].lstrip()
        data_section = sections[1]

        # Parse header
        self._header = _parse_ascii_header(header_section)

        # Validate required fields
        required = ["NumPoints", "NumProfiles", "Xscale", "Yscale", "Zscale", "DataType"]
        for field in required:
            if field not in self._header:
                raise CorruptFile(f"Missing required field '{field}' in SDF header")

        if self._header["DataType"] not in DTYPE_MAP:
            raise UnsupportedFormatFeature(
                f"Unsupported DataType in SDF file: {self._header['DataType']}"
            )

        # Compute conversion factor from meters to micrometers
        self._unit_factor = get_unit_conversion_factor(FIXED_UNIT, "µm")

        # Store data section for later parsing
        self._data_section = data_section

        # Create channel info
        nx = self._header["NumPoints"]
        ny = self._header["NumProfiles"]
        step_x = self._header["Xscale"] * self._unit_factor
        step_y = self._header["Yscale"] * self._unit_factor

        self._channels = [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(step_x * nx, step_y * ny),
                uniform=True,
                unit="µm",
                info={"raw_metadata": self._header},
            )
        ]

    def _parse_binary_file(self, f):
        """Parse binary format SDF file."""
        f.seek(0)

        # Read binary header
        header_data = f.read(BINARY_HEADER_SIZE)
        unpacked = struct.unpack(BINARY_HEADER_FORMAT, header_data)

        self._header = {
            "ManufacID": unpacked[1].decode("ascii", errors="replace").strip("\x00"),
            "CreateDate": unpacked[2].decode("ascii", errors="replace").strip("\x00"),
            "ModDate": unpacked[3].decode("ascii", errors="replace").strip("\x00"),
            "NumPoints": unpacked[4],
            "NumProfiles": unpacked[5],
            "Xscale": unpacked[6],
            "Yscale": unpacked[7],
            "Zscale": unpacked[8],
            "Zresolution": unpacked[9],
            "Compression": unpacked[10],
            "DataType": unpacked[11],
            "CheckType": unpacked[12],
        }

        if self._header["DataType"] not in DTYPE_MAP:
            raise UnsupportedFormatFeature(
                f"Unsupported DataType in SDF file: {self._header['DataType']}"
            )

        # Store data offset
        self._data_offset = f.tell()

        # Compute conversion factor from meters to micrometers
        self._unit_factor = get_unit_conversion_factor(FIXED_UNIT, "µm")

        # Create channel info
        nx = self._header["NumPoints"]
        ny = self._header["NumProfiles"]
        step_x = self._header["Xscale"] * self._unit_factor
        step_y = self._header["Yscale"] * self._unit_factor

        self._channels = [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(step_x * nx, step_y * ny),
                uniform=True,
                unit="µm",
                info={"raw_metadata": self._header},
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
        nx = self._header["NumPoints"]
        ny = self._header["NumProfiles"]
        data_type = self._header["DataType"]
        z_scale = self._header["Zscale"] * self._unit_factor

        if self._is_binary:
            # Read binary data
            with OpenFromAny(self._fobj, "rb") as f:
                f.seek(self._data_offset)
                dtype = DTYPE_MAP[data_type]
                data = np.frombuffer(
                    f.read(nx * ny * dtype.itemsize), dtype=dtype
                ).reshape(ny, nx)

            # Mark invalid values
            invalid_value = INVALID_VALUE_MAP[data_type]
            data = data.astype(np.float64)
            if not np.isnan(invalid_value):
                data[data == invalid_value] = np.nan
            data *= z_scale
        else:
            # Parse ASCII data
            data = _read_ascii_data(self._data_section, nx, ny, z_scale)

        # Transpose to get (nx, ny) ordering
        data = data.T

        _info = dict(self._header)
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
