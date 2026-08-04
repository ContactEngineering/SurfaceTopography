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
# http://www.imagemet.com/WebHelp6/Default.htm#Reference_Guide/BCR_STM_File_Format.htm
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/bcrfile.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, TextHeader
from .expr import C, Cond, F, Tup, V
from .Reader import Check, CompoundLayout, DeclarativeReaderBase, If, Seek

_MAGIC = "fileformat = bcr"

# The header is either 2048 or 4096 characters; its exact length is given
# by the `headersize` key, which sits within the first 2048 characters
_MIN_HEADER_SIZE = 2048

# Files of the `bcrstm` flavor hold 16-bit integer data, `bcrf` files
# hold single-precision floating-point data. The `_unicode` suffix marks
# UTF-16-encoded headers.
_STM_FORMATS = ("bcrstm", "bcrstm_unicode")
_ALL_FORMATS = ("bcrstm", "bcrf", "bcrstm_unicode", "bcrf_unicode")

# Void (undefined) pixels are marked by the maximum value of the
# respective data type; `voidpixels` holds the *number* of void pixels in
# the file, not the marker value. If the key is missing, the file
# contains no void pixels.
_VOID_STM = 32767
_VOID_F = 3.4028e38


def _variant(encoding, bytes_per_char):
    """
    The BCR file structure for one header encoding: a probe pass over the
    first 2048 bytes locates the `headersize` key, a second pass parses
    the full header, and the raster follows at the byte offset given by
    the header size (which counts characters).
    """
    is_stm = C.header.fileformat.isin(*_STM_FORMATS)
    little_endian = F.int(C.header.intelmode) != 0
    return CompoundLayout(
        [
            TextHeader(
                name="headersize_probe",
                size=_MIN_HEADER_SIZE,
                encoding=encoding,
                comment_prefixes=("%", "#"),
            ),
            Check(
                F.get(C.headersize_probe, "headersize", None) != None,  # noqa: E711
                CorruptFile,
                "Could not find 'headersize' key in file metadata",
            ),
            Seek(0),
            TextHeader(
                name="header",
                size=F.int(C.headersize_probe.headersize) * bytes_per_char,
                encoding=encoding,
                comment_prefixes=("%", "#"),
            ),
            Check(
                C.header.fileformat.isin(*_ALL_FORMATS), FileFormatMismatch
            ),
            Check(
                C.header.xunit == C.header.yunit,
                CorruptFile,
                "x and y units differ",
            ),
            Check(
                F.unit_conversion_factor(C.header.zunit, "m") != None,  # noqa: E711
                UnsupportedFormatFeature,
                "This BCR/BCRF file reports data in units that are not "
                "height units as expected for topography data.",
            ),
            BinaryArray(
                "data",
                Tup(F.int(C.header.ypixels), F.int(C.header.xpixels)),
                Cond(
                    is_stm,
                    Cond(little_endian, F.dtype("<i2"), F.dtype(">i2")),
                    Cond(little_endian, F.dtype("<f4"), F.dtype(">f4")),
                ),
                conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
                mask_fun=Cond(
                    F.float(F.get(C.header, "voidpixels", 0)) > 0,
                    Cond(is_stm, V == _VOID_STM, V >= _VOID_F),
                    False,
                ),
            ),
        ]
    )


class BCRReader(DeclarativeReaderBase):
    _format = "bcr"
    _mime_types = ["application/x-bcr-spm", "application/x-bcrf-spm"]
    _file_extensions = ["bcr", "bcrf"]

    _name = "BCR-STM file format"
    _description = """
BCR-STM and BCRF file formats
"""

    _magic = [
        (0, _MAGIC.encode("latin-1")),
        (0, _MAGIC.encode("utf-16-le")),
        (0, b"\xff\xfe" + _MAGIC.encode("utf-16-le")),
        (0, b"\xfe\xff" + _MAGIC.encode("utf-16-be")),
    ]

    _file_layout = CompoundLayout(
        [
            # Probe the leading bytes to detect the header encoding, then
            # rewind; the file magic itself (the `fileformat` key) is
            # validated within the branch
            BinaryStructure(
                [("magic", f"{2 * len(_MAGIC) + 2}b")],
                name="encoding_probe",
            ),
            Seek(0),
            If(
                C.encoding_probe.magic[: len(_MAGIC)]
                == _MAGIC.encode("latin-1"),
                _variant("latin-1", 1),
                C.encoding_probe.magic[: 2 * len(_MAGIC)]
                == _MAGIC.encode("utf-16-le"),
                _variant("utf-16-le", 2),
                C.encoding_probe.magic[:2].isin(b"\xff\xfe", b"\xfe\xff"),
                _variant("utf-16", 2),
                Check(
                    False,
                    FileFormatMismatch,
                    "This is not a BCR-STM/BCRF data file",
                ),
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(
                F.int(C.header.xpixels), F.int(C.header.ypixels)
            ),
            "physical_sizes": Tup(
                F.float(C.header.xlength), F.float(C.header.ylength)
            ),
            "uniform": True,
            "unit": C.header.xunit,
            "height_scale_factor": F.float(C.header.bit2nm)
            * F.unit_conversion_factor(C.header.zunit, C.header.xunit),
            "info": {"raw_metadata": C.header},
            "data": C.data,
        }
    ]
