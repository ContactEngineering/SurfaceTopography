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
Reader for Digital Metrology OmniSurf3D OS3D files.

Format documentation: https://digitalmetrology.com/omnisurf3d-file-format/
"""

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase

_MAGIC = "OmniSurf3D"

# Invalid value marker (minimum float32 value)
_INVALID_VALUE_THRESHOLD = -3e38


class OS3DReader(DeclarativeReaderBase):
    _format = "os3d"
    _mime_types = ["application/x-omnisurf3d"]
    _file_extensions = ["os3d"]

    _name = "Digital Metrology OmniSurf3D"
    _description = """
OS3D data files of Digital Metrology's OmniSurf3D surface analysis software.
"""

    _magic = [(0, _MAGIC.encode("ascii"))]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    ("magic", "10s", Validate(_MAGIC, FileFormatMismatch)),
                    ("major_version", "i"),
                    ("minor_version", "i"),
                    # 32-bit length-prefixed identification and datetime
                    # strings
                    ("identification", "T"),
                    ("datetime", "T"),
                    ("nb_grid_pts_x", "i", Validate(V > 0, CorruptFile)),
                    ("nb_grid_pts_y", "i", Validate(V > 0, CorruptFile)),
                    ("step_x", "d", Validate(V > 0, CorruptFile)),
                    ("step_y", "d", Validate(V > 0, CorruptFile)),
                    ("origin_x", "d"),
                    ("origin_y", "d"),
                ],
                byte_order="<",
                name="header",
            ),
            BinaryArray(
                "data",
                Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
                F.dtype("<f4"),
                conversion_fun=F.transpose(V),  # Transpose to (nx, ny) order
                # Invalid pixels are marked by the minimum float32 value
                mask_fun=V < _INVALID_VALUE_THRESHOLD,
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
            # Physical sizes in micrometers
            "physical_sizes": Tup(
                C.header.step_x * C.header.nb_grid_pts_x,
                C.header.step_y * C.header.nb_grid_pts_y,
            ),
            "uniform": True,
            "unit": "µm",
            "info": {"raw_metadata": C.header},
            "data": C.data,
        }
    ]
