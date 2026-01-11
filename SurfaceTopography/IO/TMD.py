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

import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, MagicMatch


class TMDReader(DeclarativeReaderBase):
    _format = "tmd"
    _mime_types = ["application/x-truemap-tmd"]
    _file_extensions = ["tmd"]

    _name = "TrueMap"
    _description = """
This reader imports TrueMap TMD data files.
"""

    _MAGIC = b"Binary TrueMap Data File v2.0"

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE
        if buffer.startswith(cls._MAGIC):
            return MagicMatch.YES
        return MagicMatch.NO

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    (
                        "magic",
                        "32s",
                        Validate(
                            lambda x, context: x.startswith("Binary TrueMap Data File v2.0"),
                            FileFormatMismatch,
                        ),
                    ),
                    ("comment", "24s"),
                    (
                        "nb_grid_pts_x",
                        "<I",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "nb_grid_pts_y",
                        "<I",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "length_x",
                        "<f",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "length_y",
                        "<f",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    ("offset_x", "<f"),
                    ("offset_y", "<f"),
                ],
                name="header",
            ),
            BinaryArray(
                "data",
                lambda context: (
                    context.header.nb_grid_pts_y,
                    context.header.nb_grid_pts_x,
                ),
                lambda context: np.dtype("<f4"),
                conversion_fun=lambda x: x.T,  # Transpose to (nx, ny) order
            ),
        ]
    )

    @property
    def channels(self):
        header = self._metadata.header

        # Parse comment for metadata
        comment = header.comment
        if isinstance(comment, bytes):
            comment = comment.decode("utf-8", errors="replace").strip("\x00\r\n ")

        return [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=2,
                nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
                physical_sizes=(header.length_x, header.length_y),
                height_scale_factor=1.0,
                uniform=True,
                unit="µm",
                info={"comment": comment, "raw_metadata": header},
                tags={"reader": self._metadata.data},
            )
        ]
