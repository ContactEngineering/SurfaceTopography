#
# Copyright 2020-2026 Lars Pastewka
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
# Gwyddion OPD reader:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/opdfile.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, Cond, F, Tup, V
from .Reader import (
    Check,
    CompoundLayout,
    DeclarativeReaderBase,
    For,
    ForEach,
    If,
    Let,
    SizedChunk,
    Skip,
    Switch,
)

# Each directory entry is a 16-byte name followed by type, length and
# attribute fields
_BLOCK_HEADER_SIZE = 24

# Block types
_TYPE_ARRAY = 3
_TYPE_TEXT = 5
_TYPE_SHORT = 6
_TYPE_FLOAT = 7
_TYPE_DOUBLE = 8
_TYPE_LONG = 12

# Longest text block that is decoded into the metadata
_MAX_TEXT_LENGTH = 256

_block_header = [
    ("name", "16s"),
    ("type", "h"),
    ("length", "l"),
    ("attribute", "H"),
]

# Data raster: dimensions and element size, followed by the raw data.
# The block name is captured from the enclosing directory entry so that
# it can name the channel.
_raster = CompoundLayout(
    [
        Let({"block_name": C.__parent__.item.name}),
        BinaryStructure(
            [
                ("nb_grid_pts_x", "H", Validate(V > 0, CorruptFile)),
                ("nb_grid_pts_y", "H", Validate(V > 0, CorruptFile)),
                (
                    "itemsize",
                    "H",
                    Validate(V.isin(1, 2, 4), UnsupportedFormatFeature),
                ),
            ],
            byte_order="<",
            name="header",
        ),
        BinaryArray(
            "data",
            # The data is stored with the x index first
            Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
            Cond(
                C.header.itemsize == 1,
                F.dtype("u1"),
                Cond(
                    C.header.itemsize == 2, F.dtype("<i2"), F.dtype("<f4")
                ),
            ),
            # Flip the y direction (like other implementations of this
            # format)
            conversion_fun=F.flip(V, 1),
        ),
    ],
    name="raster",
)

# Names of array blocks that contain height data. (Arrays named "Image",
# "Intensity" or "SecArr_0" hold intensity data and are not read.)
_HEIGHT_ARRAYS = ["RAW DATA", "RAW_DATA", "OPD", "Raw", "SAMPLE_DATA"]

_block_length = C.__parent__.item.length


def _scalar(fmt):
    return BinaryStructure([("value", fmt)], byte_order="<")


_no_value = Let({"value": None})


def _named(layout):
    """Store the decoded value together with the block name"""
    return CompoundLayout(
        [Let({"name": C.__parent__.item.name}), layout], name="meta"
    )


# Scalar and text blocks are decoded into a (name, value) record; blocks
# of other types (e.g. serialized structures) are skipped. Some files
# declare inconsistent types for a few blocks (see Gwyddion's
# opdfile.c); the actual value size is inferred from the block length.
_metadata_entry = Switch(
    C.item.type,
    {
        _TYPE_TEXT: _named(
            Switch(
                _block_length,
                {n: _scalar(f"{n}s") for n in range(1, _MAX_TEXT_LENGTH + 1)},
                default=_no_value,
            )
        ),
        _TYPE_SHORT: _named(
            If(
                _block_length == 2,
                _scalar("h"),
                _block_length == 4,
                _scalar("i"),
                _no_value,
            )
        ),
        _TYPE_FLOAT: _named(
            If(
                _block_length == 2,
                _scalar("h"),
                _block_length >= 4,
                _scalar("f"),
                _no_value,
            )
        ),
        _TYPE_DOUBLE: _named(If(_block_length >= 8, _scalar("d"), _no_value)),
        _TYPE_LONG: _named(If(_block_length >= 4, _scalar("i"), _no_value)),
    },
    default=Skip(),
)

# Mapping from block names to decoded scalar and text values
_metadata = F.to_map(F.pluck(C.payloads, "meta"), "name", "value")

_wavelength = F.get(_metadata, "Wavelength", None)
_mult = F.get(_metadata, "Mult", 1)
_aspect = F.get(_metadata, "Aspect", 1.0)
_pixel_size = F.get(_metadata, "Pixel_size", None)

_date = F.get(_metadata, "Date", "")
_time = F.get(_metadata, "Time", "")

# Heuristic for undefined data points: points that are not finite or
# that exceed a data-type dependent maximum value (32766 for 16-bit
# integers, 1e38 for floats) are undefined
_undefined_mask = Cond(
    C.item.header.itemsize == 1,
    False,
    F.logical_not(
        F.isfinite(V)
        & (V < Cond(C.item.header.itemsize == 2, 32766, 1.0e38))
    ),
)


class OPDReader(DeclarativeReaderBase):
    _format = "opd"
    _mime_types = ["application/x-wyko-opd"]
    _file_extensions = ["opd"]

    _name = "Wyko OPD"
    _description = """
OPD files generated by the Vision software of Bruker Wyko white-light
interferometers.
"""

    # The 'Directory' block name follows a two-byte prelude
    _magic = [(2, b"Directory")]

    _file_layout = CompoundLayout(
        [
            Skip(2, comment="prelude"),
            BinaryStructure(
                [
                    ("name", "16s", Validate("Directory", FileFormatMismatch)),
                    ("type", "h"),
                    ("length", "l"),
                    ("attribute", "H"),
                ],
                byte_order="<",
                name="directory_header",
            ),
            Check(
                C.directory_header.length % _BLOCK_HEADER_SIZE == 0,
                CorruptFile,
                "Directory length is not a multiple of the block size.",
            ),
            For(
                C.directory_header.length // _BLOCK_HEADER_SIZE - 1,
                BinaryStructure(_block_header, byte_order="<"),
                name="directory",
            ),
            # The block payloads follow in directory order; each block
            # occupies exactly the length declared in the directory,
            # independent of how many bytes are actually interpreted.
            # Blocks with non-positive lengths have no payload.
            ForEach(
                C.directory,
                If(
                    C.item.length > 0,
                    SizedChunk(
                        C.item.length,
                        If(
                            C.item.type == _TYPE_ARRAY,
                            Switch(
                                C.item.name,
                                {name: _raster for name in _HEIGHT_ARRAYS},
                                default=Skip(),
                            ),
                            _metadata_entry,
                        ),
                        mode="skip-missing",
                    ),
                ),
                name="payloads",
            ),
            Check(
                F.logical_not(F.isnan(_wavelength)),
                CorruptFile,
                "File does not contain a 'Wavelength' block; cannot "
                "determine the height scale.",
            ),
        ]
    )

    _channel_bindings = [
        {
            # One channel per data raster block
            "foreach": F.pluck(C.payloads, "raster"),
            "name": C.item.block_name,
            "dim": 2,
            "nb_grid_pts": Tup(
                C.item.header.nb_grid_pts_x, C.item.header.nb_grid_pts_y
            ),
            # Without pixel size, the lateral calibration is unknown
            "physical_sizes": Cond(
                F.isnan(_pixel_size),
                None,
                Tup(
                    C.item.header.nb_grid_pts_x * _pixel_size,
                    C.item.header.nb_grid_pts_y * _pixel_size * _aspect,
                ),
            ),
            # Heights are in nm, widths in mm
            "height_scale_factor": _wavelength / _mult * 1e-6,
            "uniform": True,
            "unit": "mm",
            "info": {
                # Date is month/day/year
                "acquisition_time": Cond(
                    (F.len(_date) > 0) & (F.len(_time) > 0),
                    F.parse_datetime(_date + " " + _time),
                    None,
                ),
                "raw_metadata": _metadata,
            },
            "data": C.item.data,
            "mask": {"source": C.item.data, "rule": _undefined_mask},
        }
    ]
