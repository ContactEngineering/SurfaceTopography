#
# Copyright 2022-2026 Lars Pastewka
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/microprof.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, TLVContainer, Validate
from .expr import C, Cond, DictExpr, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase, For, If, SizedChunk, Skip

_MAGIC = "FRTM_GLIDERV"

# Undefined data is marked by the value 1
_UNDEFINED_DATA = 1

# Tag of the block holding the topography data
_TOPOGRAPHY_TAG = 0x000B


def _height_array(name, nb_grid_pts_x, nb_grid_pts_y, bits_per_pixel):
    """
    Height data: a raster of 16-bit unsigned or 32-bit signed integers,
    stored row by row.
    """
    return BinaryArray(
        name,
        Tup(nb_grid_pts_y, nb_grid_pts_x),
        Cond(bits_per_pixel == 16, F.dtype("<u2"), F.dtype("<i4")),
        # Transpose to (nx, ny) order. The first stored row is the bottom
        # line of the image; flip it to match the orientation of Gwyddion
        # (`microprof.c` flips both the binary and the text variant of the
        # format) and of the other readers.
        conversion_fun=F.flip(F.transpose(V), 1),
        # Undefined data is marked by the value 1
        mask_fun=V == _UNDEFINED_DATA,
    )


# Image data types of the multi-image block 0x007D (bit flags). Only the
# height-like types are reported as channels; the remaining ones (e.g.
# intensity, camera, phase) are no topography data.
_IMAGE_TYPE_MASK = 0x0000FFDF
_IMAGE_TYPE_NAMES = DictExpr(
    {
        "4": "Topography",
        "128": "Thickness",
        "2048": "Sample thickness",
        "4096": "AFM",
    }
)
_HEIGHT_IMAGE_TYPES = 0x0004 | 0x0080 | 0x0800 | 0x1000
_BOTTOM_SENSOR = 0x10000000
_BUFFER_COUNTER_MASK = 0x0F000000


def _block(fields, name=None, mode="read-once"):
    """
    A metadata block of the FRT tag list: a fixed field structure wrapped
    into a size check against the block size declared in the tag header.
    Blocks with (undocumented) trailing data use mode 'skip-missing'.
    """
    return SizedChunk(
        C._block_size,
        BinaryStructure(fields, byte_order="<"),
        mode=mode,
        name=name,
    )


_tag_map = {
    # Comment and identification text blocks
    0x0065: "text",
    0x0077: "text",
    0x0066: _block(
        [
            ("nb_grid_pts_x", "I", Validate(V > 0, CorruptFile)),
            ("nb_grid_pts_y", "I", Validate(V > 0, CorruptFile)),
            (
                "bytes_per_pixel",
                "I",
                Validate(V.isin(16, 32), UnsupportedFormatFeature),
            ),
        ],
        # Named, so that the topography data block can reference the grid
        # dimensions
        name="grid",
    ),
    0x0067: _block(
        [
            ("physical_size_x", "d"),
            ("physical_size_y", "d"),
            ("offset_x", "d"),
            ("offset_y", "d"),
            ("factor_y", "d"),
            ("scan_direction", "I"),
        ]
    ),
    0x0068: _block(
        [
            ("xspeed", "d"),
            ("yspeed", "d"),
            ("override_speed", "I"),
            ("check_sensor_error", "I"),
            ("scan_back_meas", "I"),
            ("sensor_delay", "I"),
            ("sensor_error_time", "I"),
        ]
    ),
    0x0069: _block(
        [
            ("range_unit_type", "I"),
            ("offset_unit_type", "I"),
            ("xspeed_unit_type", "I"),
            ("yspeed_unit_type", "I"),
        ]
    ),
    0x006A: _block(
        [
            ("step_xcount", "I"),
            ("step_ycount", "I"),
            ("xstep", "d"),
            ("ystep", "d"),
            ("step_delay", "I"),
            ("back_scan_step", "I"),
        ]
    ),
    0x006B: _block(
        [
            ("wait_at_start_of_line", "I"),
            ("display_start_box", "I"),
            ("do_hysteresis_corr", "I"),
            ("back_scan_delay", "I"),
        ]
    ),
    0x006C: _block(
        [
            ("meas_range", "I"),
            ("height_scale_factor", "d"),
        ]
    ),
    0x006D: _block(
        [
            ("zrange", "d"),
            ("use_percentage", "d"),
            ("display_correction", "I"),
            ("palette_type", "I"),
            ("display_size", "I"),
            ("autorange", "I"),
        ]
    ),
    0x006E: _block(
        [
            ("sensor_type", "I"),
            ("xytable_type", "I"),
            ("ztable_type", "I"),
        ]
    ),
    0x006F: _block(
        [
            ("do_integrate", "I"),
            ("integrate_over", "I"),
            ("sensor_was_piezo", "I"),
            ("sensor_was_full", "I"),
        ]
    ),
    0x0070: _block(
        [
            ("first_valid", "I"),
            ("last_valid", "I"),
        ]
    ),
    0x0071: _block(
        [
            ("zoffset", "d"),
        ]
    ),
    0x0072: _block(
        [
            ("meas_started", "I"),
            ("meas_ended", "I"),
            ("meas_time", "I"),
        ]
    ),
    0x0073: _block(
        [
            ("dio_type", "I"),
        ]
    ),
    0x0074: _block(
        [
            ("dllver1", "I"),
            ("dllver2", "I"),
        ]
    ),
    # There is a (drift correction) data block at the end of this section
    0x0075: _block(
        [
            ("nb_values", "I"),
            ("is_applied", "I"),
            ("do_drift_corr_scan", "I"),
            ("data_available", "I"),
            ("line_not_row", "I"),
        ],
        mode="skip-missing",
    ),
    0x0076: _block(
        [
            ("xstart", "d"),
            ("ystart", "d"),
            ("xend", "d"),
            ("yend", "d"),
        ]
    ),
    0x0079: _block(
        [
            ("xdispoffset", "d"),
            ("ydispoffset", "d"),
        ]
    ),
    0x007A: _block(
        [
            ("meas_rate", "I"),
            ("min_intensity", "I"),
        ]
    ),
    0x007B: _block(
        [
            ("sensor_subtype", "I"),
            ("xytable_subtype", "I"),
        ]
    ),
    0x007C: _block(
        [
            ("speed_control", "I"),
        ]
    ),
    0x007E: _block(
        [
            ("max_xrange", "d"),
            ("max_yrange", "d"),
        ]
    ),
    0x007F: _block(
        [
            ("calibration", "255s"),
            ("is_calibrated", "?"),
        ]
    ),  # check
    0x0080: _block(
        [
            ("is_z_motor_ctrl_on", "I"),
        ]
    ),
    0x0081: _block(
        [
            ("nb_layers", "I"),
            ("range1", "d"),
            ("range_rest", "d"),
        ]
    ),
    0x0082: _block(
        [
            ("motion_type", "I"),
        ]
    ),
    0x0083: _block(
        [
            ("data_type", "I"),
        ]
    ),
    0x0084: _block(
        [
            ("use_std_schichthohe", "I"),
        ]
    ),
    0x0085: _block(
        [
            ("volt_range", "I"),
            ("val_channel", "I"),
            ("int_channel", "I"),
            ("val_range", "d"),
            ("int_range", "I"),
            ("min_valid_val", "d"),
            ("max_valid_val", "d"),
            ("min_valid_intens", "d"),
            ("max_valid_intens", "d"),
            ("unit1", "16s"),
            ("unit2", "16s"),
            ("unit3", "16s"),
            ("unit4", "16s"),
            ("unit5", "16s"),
            ("unit6", "16s"),
            ("unit7", "16s"),
            ("unit8", "16s"),
            ("selected_unit", "I"),
        ]
    ),
    0x0086: _block(
        [
            ("product_id", "H"),
            ("series_no", "H"),
        ]
    ),
    0x0087: _block(
        [
            ("use_frt_offset", "I"),
        ]
    ),
    0x0088: _block(
        [
            ("volt_range", "I"),
            ("val_channel", "I"),
            ("int_channel", "I"),
            ("int_range", "I"),
            ("min_valid_val", "d"),
            ("max_valid_val", "d"),
            ("min_valid_intens", "d"),
            ("max_valid_intens", "d"),
            ("unit1", "16s"),
            ("unit2", "16s"),
            ("unit3", "16s"),
            ("unit4", "16s"),
            ("unit5", "16s"),
            ("unit6", "16s"),
            ("unit7", "16s"),
            ("unit8", "16s"),
            ("selected_unit", "I"),
            ("min_valid_unit_value", "d"),
            ("max_valid_unit_value", "d"),
        ]
    ),
    0x0089: _block(
        [
            ("auto_approach", "I"),
            ("auto_retract", "I"),
        ]
    ),
    0x008A: _block(
        [
            ("zmotor_drive_allowed", "I"),
            ("zmotor_drive_way", "d"),
        ]
    ),
    0x008B: _block(
        [
            ("do_wait", "I"),
        ]
    ),
    0x008C: _block(
        [
            ("tv_range", "d"),
            ("tv_offset", "d"),
            ("set_tv_offset", "B"),
            ("set_tv_automatic", "B"),
            ("tv_range_percent", "f"),
        ]
    ),
    0x008D: _block(
        [
            ("meas_mode", "I"),
            ("height_edit", "d"),
            ("topo_edit", "d"),
            ("pref_mode", "I"),
            ("freq_edit", "d"),
            ("hf_edit", "I"),
            ("nf_edit", "I"),
            ("phase_edit", "d"),
            ("nf_mode", "I"),
            ("topo_scale", "d"),
        ]
    ),
    0x008E: _block(
        [
            ("serial_number", "T"),
            ("day", "B"),
            ("month", "B"),
            ("year", "H"),
            ("was_created", "I"),
            ("nb_values", "I"),
        ]
    ),
    0x008F: _block(
        [
            ("tracking_mode_activated", "I"),
        ]
    ),
    0x0090: _block(
        [
            ("despike_do_it", "I"),
            ("despike_threshold", "d"),
            ("filter_meas_do_it", "I"),
            ("filter_meas_type", "I"),
            ("filter_meas_param", "d"),
            ("tip_simul_do_it", "I"),
            ("tip_simul_angle", "d"),
            ("tip_simul_radius", "d"),
        ]
    ),
    # There are subblocks at the end of this section
    0x0091: SizedChunk(
        C._block_size,
        CompoundLayout(
            [
                BinaryStructure(
                    [
                        ("topography", "I"),
                        ("differential", "I"),
                        ("topo_edit", "d"),
                        ("height_edit", "d"),
                        ("topo_scale", "d"),
                        ("nb_subblocks", "I"),
                    ],
                    byte_order="<",
                ),
                For(
                    C.nb_subblocks,
                    BinaryStructure(
                        [
                            ("active", "I"),
                            ("frequency", "f"),
                            ("ac_dB", "f"),
                            ("low_pass", "f"),
                            ("high_pass", "f"),
                            ("out_gain", "f"),
                            ("pre_gain", "f"),
                        ],
                        byte_order="<",
                    ),
                    name="subblocks",
                ),
            ]
        ),
    ),
    # 0x0092:
    0x0093: _block(
        [
            ("invalid_values", "I"),
            ("lower_values", "I"),
            ("upper_values", "I"),
        ]
    ),
    # The rest of this block appears to be empty
    0x0094: _block(
        [
            ("min_teach", "d"),
            ("max_teach", "d"),
            ("min_norm_teach", "I"),
            ("max_norm_teach", "I"),
            ("name_of_teach", "T"),
        ],
        mode="skip-missing",
    ),
    0x0095: _block(
        [
            ("thickness_mode", "I"),
            ("kind_of_thickness", "I"),
            ("refractive_index", "d"),
        ]
    ),
    0x0096: _block(
        [
            ("thickness_lints_on", "I"),
            ("low_limit", "d"),
            ("high_limit", "d"),
        ]
    ),
    0x0097: _block(
        [
            ("laser_power", "I"),
            ("laser_power_fine", "I"),
            ("laser_frequency", "I"),
            ("intensity", "I"),
            ("min_valid_intens", "I"),
        ]
    ),
    0x0098: _block(
        [
            ("meas_z_position", "d"),
        ]
    ),
    0x0099: _block(
        [
            ("is_dual_scan", "I"),
            ("scan_frequency", "d"),
            ("duty", "f"),
        ]
    ),
    0x009A: _block(
        [
            ("is_ttv", "I"),
            ("meas_rate2", "I"),
            ("intensity2", "I"),
            ("zoffsets1", "d"),
            ("zoffsets2", "d"),
            ("scale1", "d"),
            ("scale2", "d"),
        ]
    ),
    0x009B: _block(
        [
            ("is_roundness", "I"),
            ("is_sample_used", "I"),
            ("radius", "d"),
            ("max_xrange", "d"),
            ("max_yrange", "d"),
        ]
    ),
    0x009C: _block(
        [
            ("do_despike", "I"),
            ("do_interpolate", "I"),
        ]
    ),
    0x009D: _block(
        [
            ("subtract_sinus", "I"),
        ]
    ),
    0x009E: _block(
        [
            ("layer_info", "I"),
            ("fit_threshold", "d"),
        ]
    ),
    # There is a string 'zunit' after the text length
    0x009F: _block(
        [
            ("textlen", "H"),
        ],
        mode="skip-missing",
    ),
    0x00A0: _block(
        [
            ("brightness", "H"),
            ("eval_method", "H"),
            ("focus", "H"),
            ("gain", "H"),
            ("meas_zrange", "H"),
            ("objective", "H"),
            ("shutter", "H"),
            ("zresolution", "d"),
        ]
    ),
    0x00A1: _block(
        [
            ("min_quality", "H"),
            ("focus", "d"),
        ]
    ),
    0x00A2: _block(
        [
            ("volt_range", "I"),
            ("val_channel", "I"),
            ("int_channel", "I"),
            ("int_range", "I"),
            ("min_valid_val", "d"),
            ("max_valid_val", "d"),
            ("min_valid_intens", "d"),
            ("max_valid_intens", "d"),
            ("unit1", "16s"),
            ("unit2", "16s"),
            ("unit3", "16s"),
            ("unit4", "16s"),
            ("unit5", "16s"),
            ("unit6", "16s"),
            ("unit7", "16s"),
            ("unit8", "16s"),
            ("selected_unit", "I"),
            ("min_valid_unit_value", "d"),
            ("max_valid_unit_value", "d"),
        ]
    ),
    0x00A3: _block(
        [
            ("cfm_objective", "H"),
            ("cfm_shutter", "H"),
            ("start_pos", "d"),
            ("end_pos", "d"),
            ("cfm_zresolution", "d"),
            ("lower_reflect_threshold", "d"),
            ("upper_reflect_threshold", "d"),
        ]
    ),
    0x00A4: _block(
        [
            ("angle", "d"),
            ("I_zfb", "d"),
            ("P_zfb", "d"),
            ("retract_time", "d"),
            ("xoffset", "d"),
            ("yoffset", "d"),
            ("zgain", "d"),
        ]
    ),
    0x00A5: _block(
        [
            ("external_timing", "I"),
        ]
    ),
    # There is a string 'objective_name' after the text length
    0x00A6: _block(
        [
            ("textlen", "I"),
        ],
        mode="skip-missing",
    ),
    0x00A9: _block(
        [
            ("xaxis_subtracted", "I"),
            ("yaxis_subtracted", "I"),
        ]
    ),
    0x00AA: _block(
        [
            ("sensor_ini_path", "259s"),
            ("start_pos", "d"),
            ("end_pos", "d"),
            ("zspeed", "d"),
            ("presampling_zlength", "d"),
            ("postsampling_zlength", "d"),
            ("pos_after_zscan", "I"),
            ("preprocessor", "I"),
            ("postprocessor", "I"),
        ]
    ),
    # 0x00ac user name and description
    0x00AD: _block(
        [
            ("nb_subblocks", "I"),
        ]
    ),
    0x00AE: _block(
        [
            ("signal", "I"),
            ("filter", "I"),
            ("reference_type", "I"),
            ("layer_stack_id", "I"),
            ("reference_material_id", "i"),
            ("reference_constant", "d"),
            ("material_thickness", "d"),
        ]
    ),
    0x00AF: _block(
        [
            ("auto_focus", "I"),
            ("auto_brightness", "I"),
            ("focus_search_length", "d"),
            ("max_brightness", "I"),
            ("move_back_after_meas", "I"),
            ("move_back_below_scan_range", "I"),
        ]
    ),
    # Topography data. The grid dimensions come from block 0x0066, which
    # precedes the data in the tag list and is visible in the context as
    # `grid`; the size check guards against dimension mismatch.
    _TOPOGRAPHY_TAG: SizedChunk(
        C._block_size,
        _height_array(
            "data",
            C.grid.nb_grid_pts_x,
            C.grid.nb_grid_pts_y,
            # Despite its name, this field holds the number of bits
            C.grid.bytes_per_pixel,
        ),
        name="height_data",
    ),
    # Multi-image block: four buffer ids followed by a sequence of images,
    # each with its own header (data type flags, grid dimensions and bits
    # per pixel). Newer files store the topography (duplicated in block
    # 0x000B) and further images such as intensity or a second sensor here.
    0x007D: SizedChunk(
        C._block_size,
        CompoundLayout(
            [
                BinaryStructure(
                    [
                        ("currbuf_id1", "H"),
                        ("currbuf_id2", "H"),
                        ("currbuf_id3", "H"),
                        ("currbuf_id4", "H"),
                    ],
                    byte_order="<",
                    name="buffers",
                ),
                SizedChunk(
                    C.__parent__._block_size - 8,
                    CompoundLayout(
                        [
                            BinaryStructure(
                                [
                                    ("data_type", "I"),
                                    (
                                        "nb_grid_pts_x",
                                        "I",
                                        Validate(V > 0, CorruptFile),
                                    ),
                                    (
                                        "nb_grid_pts_y",
                                        "I",
                                        Validate(V > 0, CorruptFile),
                                    ),
                                    (
                                        "bits_per_pixel",
                                        "I",
                                        Validate(
                                            V.isin(16, 32),
                                            UnsupportedFormatFeature,
                                        ),
                                    ),
                                ],
                                byte_order="<",
                                name="header",
                            ),
                            _height_array(
                                "data",
                                C.header.nb_grid_pts_x,
                                C.header.nb_grid_pts_y,
                                C.header.bits_per_pixel,
                            ),
                        ]
                    ),
                    mode="loop",
                    name="images",
                ),
            ]
        ),
        name="multi_image",
    ),
}


def _tag_list(size_format):
    return TLVContainer(
        _tag_map,
        name="blocks",
        tag_format="<H",
        size_format=size_format,
        count=C.prefix.nb_blocks,
        store_by_name=False,
        # Skip unknown blocks without reading them
        default=Skip(),
        # The entries end up in raw_metadata, which must survive a JSON
        # round trip: store them under hex string keys like '0x66'
        hex_tag_keys=True,
    )


# Start of the measurement as a POSIX timestamp (block 0x0072, which may be
# absent); 0 and 0xFFFFFFFF mark an unset time
_meas_started = F.get(F.get(C.blocks, "0x72", {}), "meas_started", 0)
_acquisition_time = Cond(
    _meas_started.isin(0, 0xFFFFFFFF), None, F.from_timestamp(_meas_started)
)


_MULTI_IMAGE_TAG = 0x007D

_physical_sizes = Tup(
    C.blocks["0x67"].physical_size_x,
    C.blocks["0x67"].physical_size_y,
)

_info = {
    "instrument": {
        "vendor": "FRT",
        # The serial number block may be absent
        "serial": F.get(F.get(C.blocks, "0x8e", {}), "serial_number", None),
    },
    "acquisition_time": _acquisition_time,
    # The lazy height data handles are not metadata
    "raw_metadata": F.omit(
        F.omit(C.blocks, hex(_TOPOGRAPHY_TAG)), hex(_MULTI_IMAGE_TAG)
    ),
}


class FRTReader(DeclarativeReaderBase):
    _format = "frt"
    _mime_types = ["application/x-microprof"]
    _file_extensions = ["frt"]

    _name = "FRT MicroProf"
    _description = """
FRT files of MicroProf optical profilometers by FRT (Fries Research &
Technology, now part of FormFactor).
"""

    _magic = [(0, _MAGIC.encode("ascii"))]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    ("magic", "12s", Validate(_MAGIC, FileFormatMismatch)),
                    (
                        "version",
                        "4s",
                        Validate(
                            V.isin("1.00", "1.01"), UnsupportedFormatFeature
                        ),
                    ),
                    ("nb_blocks", "H"),
                ],
                byte_order="<",
                name="prefix",
            ),
            # Version 1.00 uses 32-bit block sizes, 1.01 uses 64-bit
            If(
                C.prefix.version == "1.00",
                _tag_list("<I"),
                _tag_list("<Q"),
            ),
        ]
    )

    def _validate_metadata(self):
        blocks = self._metadata.blocks
        if hex(_TOPOGRAPHY_TAG) not in blocks and hex(_MULTI_IMAGE_TAG) not in blocks:
            raise CorruptFile(
                "File does not appear to contain topography data."
            )

    _channel_bindings = [
        # The topography of block 0x000B
        {
            "name": "default",
            "where": F.get(C.blocks, hex(_TOPOGRAPHY_TAG), None) != None,  # noqa: E711
            "dim": 2,
            "nb_grid_pts": Tup(
                C.blocks["0x66"].nb_grid_pts_x,
                C.blocks["0x66"].nb_grid_pts_y,
            ),
            "physical_sizes": _physical_sizes,
            "height_scale_factor": C.blocks["0x6c"].height_scale_factor,
            "unit": "m",  # All units m
            "periodic": False,
            "uniform": True,
            "info": _info,
            "data": C.blocks[hex(_TOPOGRAPHY_TAG)].data,
        },
        # Files without block 0x000B: the height-like images of the
        # multi-image block 0x007D (as in Gwyddion's `microprof.c`)
        {
            "foreach": F.get(
                F.get(C.blocks, hex(_MULTI_IMAGE_TAG), {}), "images", []
            ),
            "where": (F.get(C.blocks, hex(_TOPOGRAPHY_TAG), None) == None)  # noqa: E711
            & ((C.item.header.data_type & _HEIGHT_IMAGE_TYPES) != 0),
            "name": F.get(
                _IMAGE_TYPE_NAMES,
                F.str(C.item.header.data_type & _IMAGE_TYPE_MASK),
                "Unknown",
            )
            + Cond(
                (C.item.header.data_type & _BOTTOM_SENSOR) != 0,
                " (bottom sensor) ",
                " (top sensor) ",
            )
            + F.str(
                ((C.item.header.data_type & _BUFFER_COUNTER_MASK) >> 24) + 1
            ),
            "dim": 2,
            "nb_grid_pts": Tup(
                C.item.header.nb_grid_pts_x, C.item.header.nb_grid_pts_y
            ),
            "physical_sizes": _physical_sizes,
            "height_scale_factor": C.blocks["0x6c"].height_scale_factor,
            "unit": "m",
            "periodic": False,
            "uniform": True,
            "info": _info,
            "data": C.item.data,
        },
    ]
