#
# Copyright 2022-2023 Lars Pastewka
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/surffile.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, RawBuffer, Validate
from .expr import C, Cond, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase


class SURReader(DeclarativeReaderBase):
    _format = "sur"
    _mime_types = ["application/x-surf-spm"]
    _file_extensions = ["sur"]

    _name = "Digital Surf SUR"
    _description = """
SUR data files of Digital Surf, the native format of the MountainsMap
software. Instruments of several vendors write this format directly.
"""

    _magic = [(0, b"DIGITAL SURF")]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    ("magic", "12s", Validate("DIGITAL SURF", FileFormatMismatch)),
                    ("format", "H"),
                    ("nb_objects", "H"),
                    ("version", "H"),
                    ("type", "H"),
                    ("object_name", "30s"),
                    ("instrument_name", "30s"),
                    ("material_code", "H"),
                    ("acquisition", "H"),
                    ("range", "H"),
                    ("special_points", "H"),
                    ("absolute", "H"),
                    (None, "I"),
                    (None, "I"),
                    (
                        "itemsize",
                        "H",
                        Validate(V.isin(16, 32), UnsupportedFormatFeature),
                    ),
                    ("zmin", "i"),
                    ("zmax", "i"),
                    ("nb_grid_pts_x", "i", Validate(V > 0, CorruptFile)),
                    ("nb_grid_pts_y", "i", Validate(V > 0, CorruptFile)),
                    (
                        "nb_points",
                        "I",
                        Validate(
                            V == C.nb_grid_pts_x * C.nb_grid_pts_y, CorruptFile
                        ),
                    ),
                    ("grid_spacing_x", "f", Validate(V > 0, CorruptFile)),
                    ("grid_spacing_y", "f", Validate(V > 0, CorruptFile)),
                    ("height_scale_factor", "f", F.float(V)),
                    ("name_x", "16s"),
                    ("name_y", "16s"),
                    ("data_name", "16s"),
                    ("delta_x_unit", "16s"),
                    ("delta_y_unit", "16s"),
                    ("delta_data_unit", "16s"),
                    ("x_unit", "16s"),
                    ("y_unit", "16s"),
                    ("data_unit", "16s"),
                    # The unit ratios validate that the respective unit
                    # pairs are convertible into each other
                    (
                        "x_unit_ratio",
                        "f",
                        Validate(
                            F.unit_conversion_factor(C.x_unit, C.delta_x_unit)
                            != None  # noqa: E711
                        ),
                    ),
                    (
                        "y_unit_ratio",
                        "f",
                        Validate(
                            F.unit_conversion_factor(C.y_unit, C.delta_y_unit)
                            != None  # noqa: E711
                        ),
                    ),
                    (
                        "data_unit_ratio",
                        "f",
                        Validate(
                            F.unit_conversion_factor(
                                C.data_unit, C.delta_data_unit
                            )
                            != None  # noqa: E711
                        ),
                    ),
                    ("imprint", "H"),
                    ("inversion", "H"),
                    ("leveling", "H"),
                    (None, "I"),
                    (None, "I"),
                    (None, "I"),
                    ("second", "H"),
                    ("minute", "H"),
                    ("hour", "H"),
                    ("day", "H"),
                    ("month", "H"),
                    ("year", "H"),
                    ("dayof", "H"),
                    ("measurement_duration", "f"),
                    (None, "I"),
                    (None, "I"),
                    (None, "H"),
                    ("comment_size", "H"),
                    ("private_size", "H"),
                    ("client_zone", "128s"),
                    ("x_offset", "f"),
                    ("y_offset", "f"),
                    ("data_offset", "f"),
                    (None, "34b"),
                ],
                name="header",
            ),
            RawBuffer("comment", C.header.comment_size),
            RawBuffer("private", C.header.private_size),
            BinaryArray(
                "data",
                Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
                Cond(
                    C.header.itemsize == 16, F.dtype("<i2"), F.dtype("<i4")
                ),
                conversion_fun=F.transpose(V),
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
            # Lateral lengths are converted to the unit of the height data
            "physical_sizes": Tup(
                F.unit_conversion_factor(
                    C.header.delta_x_unit, C.header.delta_data_unit
                )
                * C.header.grid_spacing_x
                * C.header.nb_grid_pts_x,
                F.unit_conversion_factor(
                    C.header.delta_y_unit, C.header.delta_data_unit
                )
                * C.header.grid_spacing_y
                * C.header.nb_grid_pts_y,
            ),
            "height_scale_factor": C.header.height_scale_factor,
            "uniform": True,
            "unit": C.header.delta_data_unit,
            "info": {
                "instrument": {
                    "vendor": "Digital Surf",
                    "name": Cond(
                        C.header.instrument_name != "",
                        C.header.instrument_name,
                        None,
                    ),
                },
                "acquisition_time": F.make_datetime(
                    C.header.year,
                    C.header.month,
                    C.header.day,
                    C.header.hour,
                    C.header.minute,
                    C.header.second,
                ),
                "raw_metadata": C.header,
            },
            "data": C.data,
        }
    ]
