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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/psia.c
# https://gitlab.gwdg.de/ikuhlem/jpkfile
#

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, TIFFContainer
from .expr import C, Cond, F, Tup, V
from .Reader import (
    Check,
    CompoundLayout,
    DeclarativeReaderBase,
    If,
    Let,
    MagicMatch,
)

_MAGIC = 0x0E031301
_VERSION1 = 0x1000001
_VERSION2 = 0x1000002

_TAG_MAGIC = 50432
_TAG_VERSION = 50433
_TAG_DATA = 50434
_TAG_HEADER = 50435

_DATA_TYPE_INT16 = 0
_DATA_TYPE_INT32 = 1
_DATA_TYPE_FLOAT = 2

# The page record of the (single) TIFF page
_page = C.pages[0]


class PSReader(DeclarativeReaderBase):
    _format = "ps"
    _mime_types = ["image/tiff"]
    _file_extensions = ["tiff", "tif"]

    _name = "Park Systems TIFF"
    _description = """
TIFF-based file format of Park Systems instruments.
"""

    # TIFF magic bytes (little- and big-endian)
    _MAGIC_TIFF = (b"II*\x00", b"MM\x00*")

    # A TIFF magic can only tell that the file *may* be a Park Systems
    # TIFF; detection needs to try parsing
    _magic = None

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 4:
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC_TIFF):
            # TIFF file, could be a Park Systems TIFF
            return MagicMatch.MAYBE
        return MagicMatch.NO

    _file_layout = CompoundLayout(
        [
            TIFFContainer(
                tag_names={_TAG_MAGIC: "magic", _TAG_VERSION: "version"},
                tag_layouts={
                    # Vendor header; version 2 appends additional fields
                    _TAG_HEADER: CompoundLayout(
                        [
                            BinaryStructure(
                                [
                                    ("image_type", "I"),
                                    ("source_name", "32U"),
                                    ("image_mode", "8U"),
                                    ("lpf_strength", "d"),
                                    ("auto_flatten", "I"),
                                    ("ac_track", "I"),
                                    ("nb_grid_pts_x", "I"),
                                    ("nb_grid_pts_y", "I"),
                                    ("angle", "d"),
                                    ("sine_scan", "I"),
                                    ("overscan_rate", "d"),
                                    ("forward", "I"),
                                    ("scan_up", "I"),
                                    ("swap_xy", "I"),
                                    ("physical_size_x", "d"),
                                    ("physical_size_y", "d"),
                                    ("xoff", "d"),
                                    ("yoff", "d"),
                                    ("scan_rate", "d"),
                                    ("set_point", "d"),
                                    ("set_point_unit", "8U"),
                                    ("tip_bias", "d"),
                                    ("sample_bias", "d"),
                                    ("data_gain", "d"),
                                    ("data_scale_factor", "d"),
                                    ("data_offset", "d"),
                                    ("data_unit", "8U"),
                                    ("data_min", "i"),
                                    ("data_max", "i"),
                                    ("data_avg", "i"),
                                    ("compression", "I"),
                                    ("logscale", "I"),
                                    ("square", "I"),
                                ],
                                byte_order="<",
                            ),
                            If(
                                C.__parent__.tags.version == _VERSION2,
                                BinaryStructure(
                                    [
                                        ("data_servo_gain", "d"),
                                        ("data_scanner_range", "d"),
                                        ("xy_voltage_mode", "8U"),
                                        ("data_voltage_mode", "8U"),
                                        ("xy_servo_mode", "8U"),
                                        ("data_type", "I"),
                                        ("reserved1", "I"),
                                        ("reserved2", "I"),
                                        ("ncm_amplitude", "d"),
                                        ("ncm_frequency", "d"),
                                        ("cantilever_name", "16U"),
                                    ],
                                    byte_order="<",
                                ),
                                # Version 1 stores 16-bit integer data
                                Let({"data_type": _DATA_TYPE_INT16}),
                            ),
                        ],
                        name="header",
                    ),
                    # The data itself is not the TIFF image, but inside a
                    # tag; it is stored line by line, i.e. in C-order
                    # shape (ny, nx)
                    _TAG_DATA: BinaryArray(
                        "data",
                        Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
                        Cond(
                            C.header.data_type == _DATA_TYPE_INT16,
                            F.dtype("<i2"),
                            Cond(
                                C.header.data_type == _DATA_TYPE_INT32,
                                F.dtype("<i4"),
                                F.dtype("<f4"),
                            ),
                        ),
                        conversion_fun=F.transpose(V),  # To (nx, ny) order
                    ),
                },
            ),
            Check(
                F.len(C.pages) == 1,
                FileFormatMismatch,
                "More than one image in TIFF. This is not a Park Systems "
                "TIFF.",
            ),
            Check(
                F.get(_page.tags, "magic", 0) == _MAGIC,
                FileFormatMismatch,
                "This is not a Park Systems TIFF.",
            ),
            Check(
                _page.tags.version.isin(_VERSION1, _VERSION2),
                CorruptFile,
                "Only version 1 and 2 of Park Systems TIFFs are supported.",
            ),
            Check(
                _page.header.data_type.isin(
                    _DATA_TYPE_INT16, _DATA_TYPE_INT32, _DATA_TYPE_FLOAT
                ),
                CorruptFile,
                "Cannot handle data type of this Park Systems TIFF.",
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": _page.header.source_name,
            "dim": 2,
            "nb_grid_pts": Tup(
                _page.header.nb_grid_pts_x, _page.header.nb_grid_pts_y
            ),
            "physical_sizes": Tup(
                _page.header.physical_size_x, _page.header.physical_size_y
            ),
            "height_scale_factor": _page.header.data_scale_factor
            * _page.header.data_gain
            * F.unit_conversion_factor(_page.header.data_unit, "µm"),
            "uniform": True,
            "unit": "µm",
            "info": {"raw_metadata": _page.header},
            "data": _page.data,
        }
    ]
