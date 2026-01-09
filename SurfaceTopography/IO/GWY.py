#
# Copyright 2022-2024 Lars Pastewka
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

#
# Reference information and implementations:
# http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/gwyfile.c
#

import io
import os
import re
from struct import calcsize, pack, unpack

import numpy as np

from ..Exceptions import FileFormatMismatch, MetadataAlreadyFixedByFile
from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import (
    get_unit_conversion_factor,
    is_length_unit,
    mangle_length_unit_utf8,
)
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase


def _read_null_terminated_string(f):
    return b"".join(iter(lambda: f.read(1), b"\x00")).decode("utf-8")


def _gwy_read_atomic(f, atomic_type):
    atomic_size = calcsize("<" + atomic_type)
    (value,) = unpack("<" + atomic_type, f.read(atomic_size))
    if atomic_type == "b":  # booleans are special
        return value != 0
    return value


_gwy_atomic_readers = {
    "b": lambda f, **kwargs: _gwy_read_atomic(f, "b"),
    "c": lambda f, **kwargs: _gwy_read_atomic(f, "c"),
    "i": lambda f, **kwargs: _gwy_read_atomic(f, "i"),
    "q": lambda f, **kwargs: _gwy_read_atomic(f, "q"),
    "d": lambda f, **kwargs: _gwy_read_atomic(f, "d"),
    "s": lambda f, **kwargs: _read_null_terminated_string(f),
}


def _gwy_read_array(f, atomic_type, skip_arrays=False):
    offset = f.tell()
    (nb_items,) = unpack("<L", f.read(4))
    type = np.dtype(atomic_type)
    if skip_arrays:
        # Skip reading this data
        f.seek(type.itemsize * nb_items, os.SEEK_CUR)
        return {
            "offset": offset,
            "type": atomic_type,
        }  # If we skip reading the array, return the file offset
    else:
        return np.frombuffer(f.read(nb_items * type.itemsize), dtype=type)


def _gwy_read_string_array(f):
    (nb_items,) = unpack("<L", f.read(4))
    return [_read_null_terminated_string(f) for i in range(nb_items)]


_gwy_array_readers = {
    "C": lambda f, **kwargs: _gwy_read_array(f, "c", **kwargs),
    "I": lambda f, **kwargs: _gwy_read_array(f, "i", **kwargs),
    "Q": lambda f, **kwargs: _gwy_read_array(f, "q", **kwargs),
    "D": lambda f, **kwargs: _gwy_read_array(f, "d", **kwargs),
    "S": lambda f, **kwargs: _gwy_read_string_array(f),
}


def _gwy_read_component(f, skip_arrays=False):
    """
    Read a single component from a GWY file.

    Parameters
    ----------
    f : stream-like
        The file stream to read from.
    skip_arrays : bool, optional
        Skip reading arrays to avoid reading image data.
        (Default: False)

    Returns
    -------
    data : dict
        Dictionary containing the decoded data.
    """
    name = _read_null_terminated_string(f)
    type = f.read(1).decode("ascii")
    return {name: _gwy_readers[type](f, skip_arrays=skip_arrays)}


def _gwy_read_object(f, skip_arrays=False):
    """
    Read a single object from a GWY file.

    Parameters
    ----------
    f : stream-like
        The file stream to read from.
    skip_arrays : bool, optional
        Skip reading arrays to avoid reading image data.
        (Default: False)

    Returns
    -------
    data : dict
        Dictionary containing the decoded data.
    """
    name = _read_null_terminated_string(f)
    (size,) = unpack("<L", f.read(4))
    start = f.tell()
    data = {}
    while f.tell() < start + size:
        data.update(_gwy_read_component(f, skip_arrays=skip_arrays))
    return {name: data}


def _gwy_read_object_array(f):
    nb_items = unpack("<L", f.read(4))
    return [_gwy_read_object(f) for i in range(nb_items)]


_gwy_readers = {
    **_gwy_atomic_readers,
    **_gwy_array_readers,
    "o": _gwy_read_object,
    "O": _gwy_read_object_array,
}


class GWYReader(ReaderBase):
    _format = "gwy"
    _mime_types = ["application/x-gwyddion-spm"]
    _file_extensions = ["gwy"]

    _name = "Gwyddion"
    _description = """
This reader imports the native file format of the open-source SPM
visualization and analysis software Gwyddion.
"""

    _MAGIC = b"GWYP"

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC):
            return MagicMatch.YES
        return MagicMatch.NO

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, "rb") as f:
            # Detect file magic
            magic = f.read(4)
            if magic != self._MAGIC:
                raise FileFormatMismatch(
                    "File magic does not match. This is not Gwyddion file."
                )

            # Read native metadata dictionary
            gwy = _gwy_read_object(f, skip_arrays=True)
            self._metadata = gwy["GwyContainer"]

            # Construct channels
            self._channels = {}
            self._masks = {}
            self._indices = []
            for key, value in self._metadata.items():
                if key.endswith("/data"):
                    index = int(re.match(r"\/([0-9])\/data", key)[1])
                    data = value["GwyDataField"]

                    # It's not height data if 'si_unit_z' is missing.
                    if "si_unit_z" in data:
                        # Get number of grid points
                        nb_grid_pts = [data["xres"]]
                        if "yres" in data:
                            nb_grid_pts += [data["yres"]]

                        # Get physical sizes
                        physical_sizes = [data["xreal"]]
                        if "yreal" in data:
                            physical_sizes += [data["yreal"]]

                        assert len(nb_grid_pts) == len(physical_sizes)

                        xyunit = data["si_unit_xy"]["GwySIUnit"]["unitstr"]
                        zunit = data["si_unit_z"]["GwySIUnit"]["unitstr"]

                        if is_length_unit(zunit):
                            # This is height data!
                            self._indices += [index]
                            self._channels[index] = ChannelInfo(
                                self,
                                len(self._channels),
                                name=self._metadata[f"/{index}/data/title"],
                                dim=len(nb_grid_pts),
                                nb_grid_pts=tuple(nb_grid_pts),
                                physical_sizes=tuple(physical_sizes),
                                unit=xyunit,
                                height_scale_factor=get_unit_conversion_factor(
                                    zunit, xyunit
                                ),
                                periodic=False,
                                uniform=True,
                                info={
                                    "raw_metadata": {
                                        key: value
                                        for key, value in self._metadata.items()
                                        if key.startswith(f"/{index}/")
                                    }
                                },
                                tags={"data": data["data"], "index": index},
                            )
                elif key.endswith("/mask"):
                    index = int(re.match(r"\/([0-9])\/mask", key)[1])
                    data = value["GwyDataField"]
                    self._masks[index] = data["data"]

    @property
    def channels(self):
        return [self._channels[i] for i in self._indices]

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=None,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        gwy_index = self._indices[channel_index]
        channel = self._channels[gwy_index]
        with OpenFromAny(self.file_path, "rb") as f:
            nx, ny = channel.nb_grid_pts
            f.seek(channel.tags["data"]["offset"])
            height_data = (
                _gwy_read_array(f, channel.tags["data"]["type"]).reshape((ny, nx)).T
            )
            if gwy_index in self._masks:
                f.seek(self._masks[gwy_index]["offset"])
                mask_data = (
                    _gwy_read_array(f, self._masks[gwy_index]["type"])
                    .reshape((ny, nx))
                    .T
                )
                height_data = np.ma.masked_array(height_data, mask=mask_data > 0.5)

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(
            height_data,
            channel.physical_sizes,
            unit=channel.unit,
            periodic=False if periodic is None else periodic,
            info=_info,
        )
        return topo.scale(channel.height_scale_factor)


# =============================================================================
# GWY Writer
# =============================================================================

def _write_null_terminated_string(f, s):
    """Write a null-terminated UTF-8 string."""
    f.write(s.encode("utf-8") + b"\x00")


def _gwy_write_atomic(f, value, atomic_type):
    """Write an atomic value."""
    f.write(pack("<" + atomic_type, value))


def _gwy_write_component(f, name, type_char, value):
    """Write a single component (name + type + value)."""
    _write_null_terminated_string(f, name)
    f.write(type_char.encode("ascii"))

    if type_char == "b":  # boolean
        _gwy_write_atomic(f, 1 if value else 0, "b")
    elif type_char == "c":  # char
        _gwy_write_atomic(f, value, "c")
    elif type_char == "i":  # int32
        _gwy_write_atomic(f, value, "i")
    elif type_char == "q":  # int64
        _gwy_write_atomic(f, value, "q")
    elif type_char == "d":  # double
        _gwy_write_atomic(f, value, "d")
    elif type_char == "s":  # string
        _write_null_terminated_string(f, value)
    elif type_char == "D":  # double array
        arr = np.asarray(value, dtype="<f8")
        f.write(pack("<L", arr.size))
        f.write(arr.tobytes())
    elif type_char == "o":  # object
        # value should be a tuple (object_name, object_data_dict)
        obj_name, obj_data = value
        _gwy_write_object(f, obj_name, obj_data)
    else:
        raise ValueError(f"Unknown type character: {type_char}")


def _gwy_write_object(f, name, data):
    """Write an object (name + size + components)."""
    _write_null_terminated_string(f, name)

    # Write object data to a buffer first to get the size
    buf = io.BytesIO()
    for comp_name, (type_char, value) in data.items():
        _gwy_write_component(buf, comp_name, type_char, value)

    # Write size and data
    obj_data = buf.getvalue()
    f.write(pack("<L", len(obj_data)))
    f.write(obj_data)


def write_gwy(
    self,
    fobj,
    name="Topography",
):
    """
    Write topography to a Gwyddion (GWY) file.

    GWY is the native file format of the open-source SPM visualization
    and analysis software Gwyddion.

    Parameters
    ----------
    self : :obj:`Topography`
        The topography to write.
    fobj : str or file-like object
        File path or file-like object to write to.
    name : str, optional
        Name/title for the channel. (Default: 'Topography')
    """
    if self.dim != 2:
        raise ValueError("GWY format only supports 2D topographies.")

    if self.communicator is not None and self.communicator.size > 1:
        raise RuntimeError("GWY writer does not support MPI parallelization.")

    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes

    # Get unit string - GWY uses SI units
    unit = self.unit if self.unit is not None else "m"
    unit_str = mangle_length_unit_utf8(unit)

    # Get height data
    heights = self.heights()
    has_mask = False
    if np.ma.isMaskedArray(heights):
        mask = np.ma.getmask(heights)
        if mask is not np.ma.nomask and np.any(mask):
            has_mask = True
            mask_data = mask.astype(np.float64)
        heights = np.ma.filled(heights, 0.0)

    # Heights stored in column-major order (transposed)
    heights_data = heights.T.astype("<f8").flatten()

    # Build GwySIUnit objects
    si_unit_xy = {
        "unitstr": ("s", unit_str),
    }

    si_unit_z = {
        "unitstr": ("s", unit_str),
    }

    # Build GwyDataField object
    data_field = {
        "xres": ("i", nx),
        "yres": ("i", ny),
        "xreal": ("d", float(sx)),
        "yreal": ("d", float(sy)),
        "xoff": ("d", 0.0),
        "yoff": ("d", 0.0),
        "si_unit_xy": ("o", ("GwySIUnit", si_unit_xy)),
        "si_unit_z": ("o", ("GwySIUnit", si_unit_z)),
        "data": ("D", heights_data),
    }

    # Build GwyContainer
    container = {
        "/0/data/title": ("s", name),
        "/0/data": ("o", ("GwyDataField", data_field)),
    }

    # Add mask if present
    if has_mask:
        mask_field = {
            "xres": ("i", nx),
            "yres": ("i", ny),
            "xreal": ("d", float(sx)),
            "yreal": ("d", float(sy)),
            "xoff": ("d", 0.0),
            "yoff": ("d", 0.0),
            "si_unit_xy": ("o", ("GwySIUnit", si_unit_xy)),
            "si_unit_z": ("o", ("GwySIUnit", {"unitstr": ("s", "")})),
            "data": ("D", mask_data.T.astype("<f8").flatten()),
        }
        container["/0/mask"] = ("o", ("GwyDataField", mask_field))

    # Write file
    def write_to_stream(f):
        # Write magic
        f.write(b"GWYP")
        # Write container object
        _gwy_write_object(f, "GwyContainer", container)

    if isinstance(fobj, str):
        with open(fobj, "wb") as f:
            write_to_stream(f)
    else:
        write_to_stream(fobj)


UniformTopographyInterface.register_function("to_gwy", write_gwy)
