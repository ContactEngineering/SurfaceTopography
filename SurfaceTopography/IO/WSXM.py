#
# Copyright 2023-2026 Lars Pastewka
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/wsxmfile.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, TextHeader, TextLine
from .expr import C, Cond, F, Lit, Tup, V
from .Reader import Check, CompoundLayout, DeclarativeReaderBase

_MAGIC = "WSxM file copyright"

# Binary data types
_DTYPES = {
    "short": "<i2",
    "integer": "<i2",
    "float": "<f4",
    "double": "<f8",
}

# The height scale is defined in terms of the z amplitude. For this we
# need to know the actual data range. The data range is stored in the
# metadata, but only up to fixed precision. This may lead to imprecise
# data conversion, but we should not open and scan through the whole file
# here. A perfectly flat scan has zero data range; any scale factor
# reproduces it.
_data_range = F.float(C.header.Miscellaneous.Maximum) - F.float(
    C.header.Miscellaneous.Minimum
)
_z_amplitude = F.float(
    F.split(C.header["General Info"]["Z Amplitude"], " ")[0]
)
_z_unit = F.mangle_length_unit(
    F.split(C.header["General Info"]["Z Amplitude"], " ")[-1]
)


def _lateral_size(amplitude):
    """Physical size along one direction, converted to the height unit."""
    return (
        F.float(F.split(amplitude, " ")[0])
        * F.unit_conversion_factor(F.split(amplitude, " ")[-1], _z_unit)
    )


class WSXMReader(DeclarativeReaderBase):
    _format = "wsxm"
    _mime_types = ["application/x-wsxm-spm"]
    _file_extensions = ["cur", "stp", "tom", "top"]

    _name = "WSxM"
    _description = """
Data files of WSxM, a standalone software package for scanning probe
microscopy available at http://www.wsxm.eu/.
"""

    _magic = [(0, _MAGIC.encode("ascii"))]

    _file_layout = CompoundLayout(
        [
            TextLine("copyright_line"),
            Check(
                C.copyright_line[: len(_MAGIC)] == _MAGIC,
                FileFormatMismatch,
                "This is not a WSxM file, because the first line does not "
                f"start with '{_MAGIC}'.",
            ),
            TextLine("filetype_line"),
            Check(
                C.filetype_line == "SxM Image file",
                FileFormatMismatch,
                "This is not a WSxM file, because the second line does not "
                "equal 'SxM Image file'.",
            ),
            TextLine("header_size_line"),
            Check(
                C.header_size_line[:18] == "Image header size:",
                CorruptFile,
                "This WSxM file appears corrupted, because the third line "
                "does not start with 'Image header size:'.",
            ),
            TextHeader(
                name="header",
                separator=":",
                sections="ini",
                terminator="[Header end]",
                strict=True,
            ),
            BinaryArray(
                "data",
                Tup(
                    F.int(C.header["General Info"]["Number of rows"]),
                    F.int(C.header["General Info"]["Number of columns"]),
                ),
                F.dtype(Lit(_DTYPES)[C.header["General Info"]["Image Data Type"]]),
                conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(
                F.int(C.header["General Info"]["Number of columns"]),
                F.int(C.header["General Info"]["Number of rows"]),
            ),
            "physical_sizes": Tup(
                _lateral_size(C.header.Control["X Amplitude"]),
                _lateral_size(C.header.Control["Y Amplitude"]),
            ),
            "height_scale_factor": Cond(
                _data_range > 0, _z_amplitude / _data_range, 1
            ),
            "uniform": True,
            "unit": _z_unit,
            "info": {
                "acquisition_time": F.parse_datetime(
                    C.header["General Info"]["Acquisition time"]
                ),
                "raw_metadata": C.header,
            },
            "data": C.data,
        }
    ]
