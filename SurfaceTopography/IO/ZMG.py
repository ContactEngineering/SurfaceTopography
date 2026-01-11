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
Reader for KLA Zeta ZMG files.
"""

import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, RawBuffer, Validate
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, MagicMatch


class ZMGReader(DeclarativeReaderBase):
    _format = "zmg"
    _mime_types = ["application/x-zeta-zmg"]
    _file_extensions = ["zmg"]

    _name = "KLA Zeta"
    _description = """
This reader imports KLA Zeta ZMG data files.
"""

    _MAGIC = b"Zeta-Instruments"

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
                    ("magic", "16s", Validate("Zeta-Instruments", FileFormatMismatch)),
                    (None, "69s"),  # Skip to resolution fields
                    (
                        "nb_grid_pts_x",
                        "I",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "nb_grid_pts_y",
                        "I",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (None, "4s"),  # Reserved
                    (
                        "step_x",
                        "f",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "step_y",
                        "f",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    ("step_z", "f"),
                    (None, "8s"),  # Reserved
                    ("comment_size", "I"),
                    (None, "84s"),  # Reserved
                ],
                name="header",
            ),
            RawBuffer("comment", lambda context: context.header.comment_size),
            BinaryArray(
                "data",
                lambda context: (
                    context.header.nb_grid_pts_y,
                    context.header.nb_grid_pts_x,
                ),
                lambda context: np.dtype("<i2"),
                conversion_fun=lambda x: x.T,  # Transpose to (nx, ny) order
            ),
        ]
    )

    @property
    def channels(self):
        header = self._metadata.header

        # Physical sizes in micrometers
        physical_size_x = header.step_x * header.nb_grid_pts_x
        physical_size_y = header.step_y * header.nb_grid_pts_y

        return [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=2,
                nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
                physical_sizes=(physical_size_x, physical_size_y),
                height_scale_factor=header.step_z,
                uniform=True,
                unit="µm",
                info={"raw_metadata": header},
                tags={"reader": self._metadata.data},
            )
        ]
