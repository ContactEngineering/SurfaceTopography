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
Reader for Nanofocus NMS files.
"""

from ..Exceptions import FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate
from .common import OpenFromAny
from .expr import C, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase, Seek

# The header has a fixed size; the fields we interpret sit at fixed
# offsets within it
_HEADER_SIZE = 3468
_OFFSET_Z = 16  # zmin, zmax (2 doubles)
_OFFSET_POINTS = 1368  # nx, ny (2 unsigned ints), followed by dx, dy (2 doubles)

# NMS files carry no magic; the plausibility checks on the header fields
# and the file-size check in `_validate_metadata` are the only detection
# mechanism, hence validation failures report a format mismatch


class NMSReader(DeclarativeReaderBase):
    _format = "nms"
    _mime_types = ["application/x-nanofocus-nms"]
    _file_extensions = ["nms"]

    _name = "NanoFocus NMS"
    _description = """
NMS data files of NanoFocus confocal microscopes (e.g. the µsurf series).
"""

    # NMS files don't have a clear magic signature; detection needs to
    # try parsing (`_magic = None` answers "maybe")
    _magic = None

    _file_layout = CompoundLayout(
        [
            CompoundLayout(
                [
                    Seek(_OFFSET_Z, comment="z range"),
                    BinaryStructure(
                        [("zmin", "d"), ("zmax", "d")],
                        byte_order="<",
                    ),
                    Seek(_OFFSET_POINTS, comment="grid dimensions and spacing"),
                    BinaryStructure(
                        [
                            (
                                "nb_grid_pts_x",
                                "I",
                                Validate(
                                    (V > 0) & (V <= 100000), FileFormatMismatch
                                ),
                            ),
                            (
                                "nb_grid_pts_y",
                                "I",
                                Validate(
                                    (V > 0) & (V <= 100000), FileFormatMismatch
                                ),
                            ),
                            # Grid spacing in mm
                            ("dx_mm", "d", Validate(V > 0, FileFormatMismatch)),
                            ("dy_mm", "d", Validate(V > 0, FileFormatMismatch)),
                        ],
                        byte_order="<",
                    ),
                ],
                name="header",
            ),
            Seek(_HEADER_SIZE, comment="height data"),
            BinaryArray(
                "data",
                Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
                F.dtype("<u2"),
                # The data is stored as uint16 scaled between zmin and
                # zmax; rescale and transpose to (nx, ny) ordering
                conversion_fun=F.transpose(
                    V / (2**16 - 2) * (C.header.zmax - C.header.zmin)
                    + C.header.zmin
                ),
            ),
        ]
    )

    def _validate_metadata(self):
        # The file must be large enough to hold the full raster. Since the
        # format has no magic, this is the de-facto detection mechanism
        # and cannot be expressed in the file layout.
        header = self._metadata.header
        expected_file_size = (
            _HEADER_SIZE + 2 * header.nb_grid_pts_x * header.nb_grid_pts_y
        )
        with OpenFromAny(self.file_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
        if file_size < expected_file_size:
            raise FileFormatMismatch(
                f"File too small for NMS format: {file_size} < "
                f"{expected_file_size}"
            )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
            # Physical sizes in micrometers (spacing is in mm)
            "physical_sizes": Tup(
                C.header.dx_mm * 1000.0 * C.header.nb_grid_pts_x,
                C.header.dy_mm * 1000.0 * C.header.nb_grid_pts_y,
            ),
            "uniform": True,
            "unit": "µm",
            "info": {"raw_metadata": C.header},
            "data": C.data,
        }
    ]
