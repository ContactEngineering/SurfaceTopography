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
# https://www.osti.gov/biblio/1419732
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/keyence.c
# https://github.com/torkian/vk4-python-driver
#

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate, ZipContainer
from .expr import C, Cond, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase, If, Seek, Skip

# VK3 and VK4 files carry the VK4 byte stream directly; VK6 and VK7 files
# wrap the same byte stream into a ZIP archive member named `Vk4File`
# behind a proprietary prefix (which `ZipFile` gracefully skips).
_VK4_MEMBER_NAME = "Vk4File"

# Layout of the VK3/VK4 byte stream. Offsets in the offset table are
# absolute positions within this stream.
_vk34_layout = CompoundLayout(
    [
        BinaryStructure(
            [
                (
                    "magic",
                    "4s",
                    Validate(V.isin("VK3_", "VK4_"), FileFormatMismatch),
                ),
                (None, "4b"),  # DLL version
                (
                    None,
                    "4b",
                    Validate(b"\x00\x00\x00\x00", FileFormatMismatch),
                ),
            ],
            byte_order="<",
            name="prefix",
        ),
        BinaryStructure(
            [
                ("setting", "I"),
                ("color_peak", "I"),
                ("color_light", "I"),
                ("light1", "I"),
                ("light2", "I"),
                ("light3", "I"),
                ("height1", "I"),
                ("height2", "I"),
                ("height3", "I"),
                ("color_peak_thumbnail", "I"),
                ("color_thumbnail", "I"),
                ("light_thumbnail", "I"),
                ("height_thumbnail", "I"),
                ("assemble", "I"),
                ("line_measure", "I"),
                ("line_thickness", "I"),
                ("string_data", "I"),
            ],
            byte_order="<",
            name="offset_table",
        ),
        # File version 4 (and 6, 7, which wrap version 4) have an
        # additional offset entry here
        If(
            C.prefix.magic == "VK4_",
            Skip(4, comment="additional offset table entry"),
        ),
        BinaryStructure(
            [
                ("size", "I"),
                ("year", "I"),
                ("month", "I"),
                ("day", "I"),
                ("hour", "I"),
                ("minute", "I"),
                ("second", "I"),
                ("diff_utc_by_minutes", "i"),
                ("image_attributes", "I"),
                ("user_interface_mode", "I"),
                ("color_composite_mode", "I"),
                ("num_layer", "I"),
                ("run_mode", "I"),
                ("peak_mode", "I"),
                ("sharpening_level", "I"),
                ("speed", "I"),
                ("distance", "I"),
                ("pitch", "I"),
                ("optical_zoom", "I"),
                ("num_line", "I"),
                ("line0_pos", "I"),
                ("_", "I"),
                ("_", "I"),
                ("_", "I"),
                ("lens_mag", "I"),
                ("pmt_gain_mode", "I"),
                ("pmt_gain", "I"),
                ("pmt_offset", "I"),
                ("nd_filter", "I"),
                ("_", "I"),
                ("persist_count", "I"),
                ("shutter_speed_mode", "I"),
                ("shutter_speed", "I"),
                ("white_balance_mode", "I"),
                ("white_balance_red", "I"),
                ("white_balance_blue", "I"),
                ("camera_gain", "I"),
                ("plane_compensation", "I"),
                ("length_unit", "I"),
                ("height_unit", "I"),
                ("xy_decimal_place", "I"),
                ("height_decimal_place", "I"),
                ("x_length_per_pixel", "I"),
                ("y_length_per_pixel", "I"),
                ("height_scale_factor", "I"),
                ("_", "I"),
                ("_", "I"),
                ("_", "I"),
                ("_", "I"),
                ("_", "I"),
                ("light_filter_type", "I"),
                ("_", "I"),
                ("gamma_reverse", "I"),
                ("gamma", "I"),
                ("gamma_offset", "I"),
                ("ccd_bw_offset", "I"),
                ("numerical_aperture", "I"),
                ("head_type", "I"),
                ("pmt_gain2", "I"),
                ("omit_color_image", "I"),
                ("lens_id", "I"),
                ("light_lut_mode", "I"),
                ("light_lut_in0", "I"),
                ("light_lut_out0", "I"),
                ("light_lut_in1", "I"),
                ("light_lut_out1", "I"),
                ("light_lut_in2", "I"),
                ("light_lut_out2", "I"),
                ("light_lut_in3", "I"),
                ("light_lut_out3", "I"),
                ("light_lut_in4", "I"),
                ("light_lut_out4", "I"),
                ("upper_position", "I"),
                ("lower_position", "I"),
                ("light_effective_itemsize", "I"),
                ("height_effective_itemsize", "I"),
            ],
            byte_order="<",
            name="header",
        ),
        # Right now, we are assuming that there is only a single (height)
        # channel per VK4 file. Not sure if this is correct.
        Seek(C.offset_table.height1, comment="height image"),
        CompoundLayout(
            [
                BinaryStructure(
                    [
                        ("width", "I", Validate(V > 0, CorruptFile)),
                        ("height", "I", Validate(V > 0, CorruptFile)),
                        (
                            "itemsize",
                            "I",
                            Validate(V.isin(8, 16, 32), CorruptFile),
                        ),
                        ("compression", "I"),
                        (
                            "byte_size",
                            "I",
                            Validate(
                                V == C.width * C.height * C.itemsize // 8,
                                CorruptFile,
                            ),
                        ),
                        ("palette_range_min", "I"),
                        ("palette_range_max", "I"),
                    ],
                    byte_order="<",
                    name="header",
                ),
                Skip(768, comment="palette"),
                BinaryArray(
                    "data",
                    Tup(C.header.height, C.header.width),
                    Cond(
                        C.header.itemsize == 8,
                        F.dtype("<u1"),
                        Cond(
                            C.header.itemsize == 16,
                            F.dtype("<u2"),
                            F.dtype("<u4"),
                        ),
                    ),
                    conversion_fun=F.transpose(V),  # Transpose to (nx, ny)
                ),
            ],
            name="height_data",
        ),
    ]
)


class VKReader(DeclarativeReaderBase):
    _format = "vk"
    _mime_types = [
        "application/x-keyence-vk3",
        "application/x-keyence-vk4",
        "application/x-keyence-vk5",
        "application/x-keyence-vk6",
    ]
    _file_extensions = ["vk3", "vk4", "vk6", "vk7"]

    _name = "Keyence VK"
    _description = """
VK3, VK4, VK6 and VK7 file formats of the Keyence laser confocal microscope.
"""

    # VK3/VK4 have a 4-byte magic, VK6/VK7 a 3-byte magic
    _magic = [(0, b"VK3_"), (0, b"VK4_"), (0, b"VK6"), (0, b"VK7")]

    _file_layout = CompoundLayout(
        [
            # Read the leading three bytes to decide between the direct
            # VK3/VK4 byte stream and the ZIP-wrapped VK6/VK7 variant,
            # then rewind; the full magic is validated within the branch
            BinaryStructure(
                [
                    (
                        "magic",
                        "3s",
                        Validate(
                            V.isin("VK3", "VK4", "VK6", "VK7"),
                            FileFormatMismatch,
                        ),
                    ),
                ],
                byte_order="<",
                name="outer",
            ),
            Seek(0, comment="rewind after magic detection"),
            If(
                C.outer.magic.isin("VK3", "VK4"),
                _vk34_layout,
                ZipContainer([(_VK4_MEMBER_NAME, _vk34_layout)]),
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(
                C.height_data.header.width, C.height_data.header.height
            ),
            # Note: the physical size is the number of pixels times the
            # pixel size (pixel convention, like everywhere else in this
            # library and in other implementations of this file format),
            # not the distance between the first and last pixel centers
            "physical_sizes": Tup(
                F.float(C.height_data.header.width)
                * C.header.x_length_per_pixel,
                F.float(C.height_data.header.height)
                * C.header.y_length_per_pixel,
            ),
            "height_scale_factor": F.float(C.header.height_scale_factor),
            "uniform": True,
            "unit": "pm",
            "info": {
                "acquisition_time": F.make_datetime(
                    C.header.year,
                    C.header.month,
                    C.header.day,
                    C.header.hour,
                    C.header.minute,
                    C.header.second,
                    C.header.diff_utc_by_minutes,
                ),
                "instrument": {"vendor": "Keyence"},
                "raw_metadata": C.header,
            },
            "data": C.height_data.data,
        }
    ]
