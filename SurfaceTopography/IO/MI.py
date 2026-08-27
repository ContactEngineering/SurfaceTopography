#
# Copyright 2019-2026 Lars Pastewka
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

from ..Exceptions import FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, TextHeader
from .expr import C, Cond, DictExpr, F, Tup, V
from .Reader import Check, CompoundLayout, DeclarativeReaderBase, For

# The header consists of key-value lines with a fixed-width 14-character
# key column. Lines starting with `bufferLabel` open a new channel
# ("buffer") section; the line with the key `data` ends the header, and
# its value selects the encoding of the data buffers that follow.
_KEY_WIDTH = 14

# Data is normalized with the range of the data type
_type_range = Cond(C.header.data == "BINARY", 32768.0, 2147483648.0)

# Unit of the data of the current channel
_unit = F.mangle_length_unit(C.item.bufferUnit)

# `xLength` and `yLength` are always given in meters; they are converted
# to the data unit of the channel if that is a length unit. For channels
# whose data is not a height (e.g. deflection in V), the lateral sizes
# are reported in meters since there is no meaningful conversion.
_is_height_channel = F.is_length_unit(_unit)


class MIReader(DeclarativeReaderBase):
    _format = "mi"
    _mime_types = ["application/x-mi-spm"]
    _file_extensions = ["mi"]

    _name = "Molecular Imaging (MI)"
    _description = """
MI files of Agilent Technologies (Molecular Imaging) atomic force
microscopes. This format contains information on the physical size of the
topography map as well as its units.
"""

    # All MI files start with a `fileType` key line
    _magic = [(0, b"fileType")]

    _file_layout = CompoundLayout(
        [
            TextHeader(
                name="header",
                encoding="utf-8",
                key_width=_KEY_WIDTH,
                section_key="bufferLabel",
                sections_name="buffers",
                stop_key="data",
                comment_prefixes=(" ",),  # Skip indented (continuation) lines
            ),
            Check(
                F.get(C.header, "fileType", "").isin("Image", "Spectroscopy"),
                FileFormatMismatch,
                "This is not an MI file.",
            ),
            Check(
                C.header.fileType == "Image",
                UnsupportedFormatFeature,
                "Spectroscopy MI files are not supported.",
            ),
            Check(
                F.get(C.header, "data", "").isin("BINARY", "BINARY_32"),
                UnsupportedFormatFeature,
                "MI files with text (ASCII) data are not supported.",
            ),
            # The data buffers of all channels follow the header
            # back-to-back, stored line by line, i.e. in C-order shape
            # (ny, nx)
            For(
                F.len(C.header.buffers),
                BinaryArray(
                    "data",
                    Tup(F.int(C.header.yPixels), F.int(C.header.xPixels)),
                    Cond(
                        C.header.data == "BINARY",
                        F.dtype("<i2"),
                        F.dtype("<i4"),
                    ),
                    # If the scan direction is upwards, flip the height
                    # map vertically; transpose to (nx, ny) order
                    conversion_fun=F.transpose(
                        Cond(
                            F.get(C.header, "scanUp", "") == "TRUE",
                            F.flip(V, 0),
                            V,
                        )
                    ),
                ),
                name="data_buffers",
            ),
        ]
    )

    @property
    def info(self) -> dict:
        """Return all the available file-wide metadata as a dict."""
        return {
            key: value
            for key, value in self._metadata.header.items()
            if key not in ("buffers", "data")
        }

    _channel_bindings = [
        {
            # One channel per buffer section of the header
            "foreach": C.header.buffers,
            "name": C.item.bufferLabel,
            "dim": 2,
            "nb_grid_pts": Tup(
                F.int(C.header.xPixels), F.int(C.header.yPixels)
            ),
            "physical_sizes": Cond(
                _is_height_channel,
                Tup(
                    F.float(C.header.xLength)
                    * F.unit_conversion_factor("m", _unit),
                    F.float(C.header.yLength)
                    * F.unit_conversion_factor("m", _unit),
                ),
                Tup(F.float(C.header.xLength), F.float(C.header.yLength)),
            ),
            # Undo the normalization with the range of the data type and
            # scale with the range of the buffer
            "height_scale_factor": F.float(C.item.bufferRange) / _type_range,
            "uniform": True,
            "unit": _unit,
            "info": {
                # Reported metadata mirrors the historic reader: the
                # buffer keys are renamed (`label`, `range`, `name`), the
                # unit is reported at the channel level only
                "raw_metadata": F.merge(
                    F.omit(F.omit(C.header, "buffers"), "data"),
                    F.merge(
                        F.omit(
                            F.omit(
                                F.omit(C.item, "bufferUnit"), "bufferRange"
                            ),
                            "bufferLabel",
                        ),
                        DictExpr(
                            {
                                "name": C.item.bufferLabel,
                                "label": C.item.bufferLabel,
                                "range": C.item.bufferRange,
                            }
                        ),
                    ),
                ),
            },
            "data": C.data_buffers[C.item_index].data,
        }
    ]
