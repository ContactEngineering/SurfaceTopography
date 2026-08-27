#
# Copyright 2025-2026 Lars Pastewka
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

from ..Exceptions import (
    CorruptFile,
    FileFormatMismatch,
    UnsupportedFormatFeature,
)
from .binary import (
    BinaryArray,
    BinaryStructure,
    TextHeader,
    TextMatrix,
    Validate,
)
from .expr import C, Cond, F, Tup, V
from .Reader import Check, CompoundLayout, DeclarativeReaderBase, If

# Magic strings for ASCII and binary variants
_MAGIC_ASCII = "aISO-1.0"
_MAGIC_BINARY = "bISO-1.0"

# Fixed unit in SDF files is meters; we report in micrometers
_FIXED_UNIT = "m"

# Data type mapping (from ISO 25178-71): 5 = INT16, 6 = INT32, 7 = DOUBLE
_DATA_TYPES = (5, 6, 7)
_dtype = Cond(
    C.header.DataType == 5,
    F.dtype("<i2"),
    Cond(C.header.DataType == 6, F.dtype("<i4"), F.dtype("<f8")),
)

# A pixel is undefined if it is NaN or, for the integer data types,
# carries the type's invalid marker (the type's minimum value). ASCII
# files with an integer DataType may also mark invalid points with the
# same numeric marker as the binary variant.
_invalid_mask = Cond(
    C.header.DataType == 7,
    F.isnan(V),
    F.isnan(V)
    | (V == Cond(C.header.DataType == 5, -(2**15), -(2**31))),
)

_binary_layout = CompoundLayout(
    [
        BinaryStructure(
            [
                ("ManufacID", "10s"),
                ("CreateDate", "12s"),
                ("ModDate", "12s"),
                ("NumPoints", "H", Validate(V > 0, CorruptFile)),
                ("NumProfiles", "H", Validate(V > 0, CorruptFile)),
                ("Xscale", "d"),
                ("Yscale", "d"),
                ("Zscale", "d"),
                ("Zresolution", "d"),
                # Note: `b` denotes raw bytes in the layout language; the
                # flag fields are read as unsigned `B` instead (the ISO
                # spec says signed, but the defined values are 0-7)
                ("Compression", "B"),
                (
                    "DataType",
                    "B",
                    Validate(V.isin(*_DATA_TYPES), UnsupportedFormatFeature),
                ),
                ("CheckType", "B"),
            ],
            byte_order="<",
            name="header",
        ),
        BinaryArray(
            "data",
            Tup(C.header.NumProfiles, C.header.NumPoints),
            _dtype,
            conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
        ),
    ]
)

_ascii_layout = CompoundLayout(
    [
        TextHeader(
            name="header",
            separator="=",
            terminator="*",
            converters={
                "NumPoints": F.int(V),
                "NumProfiles": F.int(V),
                "Xscale": F.float(V),
                "Yscale": F.float(V),
                "Zscale": F.float(V),
                "Zresolution": F.float(V),
                "Compression": F.int(V),
                "DataType": F.int(V),
                "CheckType": F.int(V),
            },
        ),
        Check(
            (F.get(C.header, "NumPoints", None) != None)  # noqa: E711
            & (F.get(C.header, "NumProfiles", None) != None)  # noqa: E711
            & (F.get(C.header, "Xscale", None) != None)  # noqa: E711
            & (F.get(C.header, "Yscale", None) != None)  # noqa: E711
            & (F.get(C.header, "Zscale", None) != None)  # noqa: E711
            & (F.get(C.header, "DataType", None) != None),  # noqa: E711
            CorruptFile,
            "Missing required field in SDF header",
        ),
        Check(
            C.header.DataType.isin(*_DATA_TYPES),
            UnsupportedFormatFeature,
            "Unsupported DataType in SDF file",
        ),
        TextMatrix(
            "data",
            Tup(C.header.NumProfiles, C.header.NumPoints),
            bad_markers=("BAD",),
            conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
        ),
    ]
)


class SDFReader(DeclarativeReaderBase):
    _format = "sdf"
    _mime_types = ["application/x-iso25178-sdf"]
    _file_extensions = ["sdf"]

    _name = "ISO 25178-71 SDF"
    _description = """
ISO 25178-71 Surface Data Files (SDF), a standardized exchange format for
surface topography data. Both the ASCII and binary variants are supported.
"""

    _magic = [
        (0, _MAGIC_ASCII.encode("ascii")),
        (0, _MAGIC_BINARY.encode("ascii")),
    ]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    (
                        "magic",
                        "8s",
                        Validate(
                            V.isin(_MAGIC_ASCII, _MAGIC_BINARY),
                            FileFormatMismatch,
                        ),
                    ),
                ],
                byte_order="<",
                name="prefix",
            ),
            If(
                C.prefix.magic == _MAGIC_BINARY,
                _binary_layout,
                _ascii_layout,
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(C.header.NumPoints, C.header.NumProfiles),
            # Lengths are stored in meters, we report micrometers
            "physical_sizes": Tup(
                C.header.Xscale
                * C.header.NumPoints
                * F.unit_conversion_factor(_FIXED_UNIT, "µm"),
                C.header.Yscale
                * C.header.NumProfiles
                * F.unit_conversion_factor(_FIXED_UNIT, "µm"),
            ),
            "height_scale_factor": C.header.Zscale
            * F.unit_conversion_factor(_FIXED_UNIT, "µm"),
            "uniform": True,
            "unit": "µm",
            "info": {"raw_metadata": C.header},
            "data": C.data,
            "mask": {"source": C.data, "rule": _invalid_mask},
        }
    ]
