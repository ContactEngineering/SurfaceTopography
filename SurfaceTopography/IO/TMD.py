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
Reader for TrueMap TMD files.
"""

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase

_MAGIC = "Binary TrueMap Data File v2.0"


class TMDReader(DeclarativeReaderBase):
    _format = "tmd"
    _mime_types = ["application/x-truemap-tmd"]
    _file_extensions = ["tmd"]

    _name = "TrueMap TMD"
    _description = """
TMD data files of the TrueGage TrueMap surface metrology software.
"""

    _magic = [(0, _MAGIC.encode("ascii"))]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    (
                        "magic",
                        "32s",
                        Validate(V[: len(_MAGIC)] == _MAGIC, FileFormatMismatch),
                    ),
                    ("comment", "24s"),
                    ("nb_grid_pts_x", "<I", Validate(V > 0, CorruptFile)),
                    ("nb_grid_pts_y", "<I", Validate(V > 0, CorruptFile)),
                    ("length_x", "<f", Validate(V > 0, CorruptFile)),
                    ("length_y", "<f", Validate(V > 0, CorruptFile)),
                    ("offset_x", "<f"),
                    ("offset_y", "<f"),
                ],
                name="header",
            ),
            BinaryArray(
                "data",
                Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
                F.dtype("<f4"),
                conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
            "physical_sizes": Tup(C.header.length_x, C.header.length_y),
            "height_scale_factor": 1.0,
            "uniform": True,
            "unit": "µm",
            "info": {
                "instrument": {"vendor": "TrueMap"},
                "comment": F.strip(C.header.comment),
                "raw_metadata": C.header,
            },
            "data": C.data,
        }
    ]
