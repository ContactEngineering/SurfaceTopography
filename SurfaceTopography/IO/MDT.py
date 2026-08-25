#
# Copyright 2026 Lars Pastewka
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/nt-mdt.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, Cond, F, Tup, V
from .Reader import (
    Check,
    CompoundLayout,
    DeclarativeReaderBase,
    For,
    If,
    SizedChunk,
    Skip,
)

_FRAME_HEADER_SIZE = 22

# Frame types (`MDTFrameType`); only scanned-data frames carry raster
# topography images in the classic layout. MDA frames (types 105/106)
# use a self-describing multidimensional layout that is not supported.
_FRAME_SCANNED = 0

# Unit codes of the axis scales (`MDTUnit`), keyed by their decimal
# string representation (JSON mappings have string keys). Non-length
# units (nA, V, deg, ...) are mapped to the empty string; channels whose
# axes do not carry length units are not emitted.
_UNIT_BY_CODE = {
    "-5": "m",
    "-4": "cm",
    "-3": "mm",
    "-2": "µm",
    "-1": "nm",
    "0": "Å",
}

# Signal names by ADC channel index (`MDTADCMode`)
_SIGNAL_BY_ADC_INDEX = {
    "0": "Height",
    "1": "DFL",
    "2": "Lateral F",
    "3": "Bias V",
    "4": "Current",
    "5": "FB-Out",
    "6": "MAG",
    "7": "MAG*Sin",
    "8": "MAG*Cos",
    "9": "RMS",
    "10": "CalcMag",
    "11": "Phase1",
    "12": "Phase2",
    "13": "CalcPhase",
    "14": "Ex1",
    "15": "Ex2",
    "16": "HvX",
    "17": "HvY",
    "18": "Snap Back",
}

# The variables section of a scanned-data frame starts with the axis
# scales (30 bytes) followed by the scan variables, of which we parse the
# leading six bytes
_PARSED_VARS_SIZE = 36

# Body of a scanned-data frame (frame type 0), positioned directly after
# the frame header. The variables section occupies `var_size` bytes; the
# frame-mode block, an optional dot (point spectroscopy) section and the
# raster image follow.
_scanned_frame = CompoundLayout(
    [
        Check(
            C.__parent__.header.var_size >= _PARSED_VARS_SIZE,
            CorruptFile,
            "The variables section of a scanned-data frame is too short.",
        ),
        # Axis scales: offset (r0), step (r) and unit code per axis
        BinaryStructure(
            [
                ("x_offset", "f"),
                ("x_step", "f"),
                ("x_unit", "h"),
                ("y_offset", "f"),
                ("y_step", "f"),
                ("y_unit", "h"),
                ("z_offset", "f"),
                ("z_step", "f"),
                ("z_unit", "h"),
            ],
            byte_order="<",
            name="axis_scales",
        ),
        # Leading scan variables; the ADC channel index determines the
        # recorded signal and thereby the channel name
        BinaryStructure(
            [
                ("adc_index", "B"),
                ("mode", "B"),
                ("nb_grid_pts_x", "H"),
                ("nb_grid_pts_y", "H"),
            ],
            byte_order="<",
            name="scan_vars",
        ),
        Skip(
            C.__parent__.header.var_size - _PARSED_VARS_SIZE,
            comment="remaining scan variables",
        ),
        # The frame-mode block carries the actual raster dimensions,
        # which can differ from the scan variables (e.g. for partial
        # scans)
        BinaryStructure(
            [
                ("mode", "H"),
                ("nb_grid_pts_x", "H"),
                ("nb_grid_pts_y", "H"),
                ("nb_dots", "H"),
            ],
            byte_order="<",
            name="frame_mode",
        ),
        If(
            C.frame_mode.nb_dots > 0,
            Skip(
                14 + 16 * C.frame_mode.nb_dots,
                comment="dot (point spectroscopy) section",
            ),
        ),
        BinaryArray(
            "data",
            # The raster image is stored row by row (y index first)
            Tup(C.frame_mode.nb_grid_pts_y, C.frame_mode.nb_grid_pts_x),
            F.dtype("<i2"),
            # Report heights in the unit of the z axis scale, like the
            # reference implementation (a zero or undefined step means
            # unscaled DAC values). The result is indexed (x, y) per the
            # array convention.
            conversion_fun=F.transpose(V)
            * Cond(
                F.isnan(C.axis_scales.z_step) | (C.axis_scales.z_step == 0),
                1.0,
                C.axis_scales.z_step,
            )
            + Cond(F.isnan(C.axis_scales.z_offset), 0.0, C.axis_scales.z_offset),
        ),
    ]
)

# Per-frame expressions used by the channel bindings
_is_scanned = C.item.header.type == _FRAME_SCANNED
_xy_unit = F.get(_UNIT_BY_CODE, F.str(C.item.axis_scales.x_unit), "")
_z_unit = F.get(_UNIT_BY_CODE, F.str(C.item.axis_scales.z_unit), "")


class MDTReader(DeclarativeReaderBase):
    _format = "mdt"
    _mime_types = ["application/x-nt-mdt"]
    _file_extensions = ["mdt"]

    _name = "NT-MDT"
    _description = """
File format of NT-MDT (now NT-MDT Spectrum Instruments) atomic force
microscopes, e.g. the Solver and Ntegra series running the Nova
software. A file is a container of frames; this reader supports the
classic scanned-data frames (raster images). Spectroscopy, text and
multidimensional-array (MDA) frames are skipped.
"""

    _magic = [(0, b"\x01\xb0\x93\xff")]

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    # The magic bytes 01 B0 93 FF as little-endian uint32
                    ("magic", "I", Validate(0xFF93B001, FileFormatMismatch)),
                    ("frame_data_size", "I"),
                    (None, "4s"),  # reserved
                    ("last_frame_index", "H"),
                    (None, "18s"),  # reserved
                    # The header is documented as 32 bytes but is
                    # actually followed by one more unused byte
                    (None, "B"),
                ],
                byte_order="<",
                name="header",
            ),
            For(
                C.header.last_frame_index + 1,
                CompoundLayout(
                    [
                        BinaryStructure(
                            [
                                (
                                    "size",
                                    "I",
                                    Validate(V >= _FRAME_HEADER_SIZE, CorruptFile),
                                ),
                                ("type", "H"),
                                # Stored big-endian within the otherwise
                                # little-endian structure
                                ("version", ">H"),
                                ("year", "H"),
                                ("month", "H"),
                                ("day", "H"),
                                ("hour", "H"),
                                ("minute", "H"),
                                ("second", "H"),
                                ("var_size", "H"),
                            ],
                            byte_order="<",
                            name="header",
                        ),
                        # Each frame occupies exactly the size declared in
                        # its header; content following the parsed
                        # structures (title, XML comment) and frames of
                        # unsupported types are skipped.
                        SizedChunk(
                            C.header.size - _FRAME_HEADER_SIZE,
                            If(C.header.type == _FRAME_SCANNED, _scanned_frame),
                            mode="skip-missing",
                        ),
                    ]
                ),
                name="frames",
            ),
            Check(
                F.len(F.match_records(C.frames, ("header", "type"), _FRAME_SCANNED))
                > 0,
                UnsupportedFormatFeature,
                "File contains no scanned-data frames. (Spectroscopy, text "
                "and MDA frames are not supported.)",
            ),
        ]
    )

    _channel_bindings = [
        {
            # One channel per scanned-data frame whose axes carry length
            # units and that contains a raster image (dots-only frames
            # have zero-size images)
            "foreach": C.frames,
            "where": Cond(
                _is_scanned,
                F.is_length_unit(_xy_unit)
                & F.is_length_unit(_z_unit)
                & (C.item.frame_mode.nb_grid_pts_x > 0)
                & (C.item.frame_mode.nb_grid_pts_y > 0),
                False,
            ),
            "checks": [
                {
                    # NaN-valued steps are sanitized to null by the
                    # decoder; `isnan(null)` is true
                    "condition": Cond(
                        F.isnan(C.item.axis_scales.x_step)
                        | F.isnan(C.item.axis_scales.y_step),
                        False,
                        (C.item.axis_scales.x_step > 0)
                        & (C.item.axis_scales.y_step > 0),
                    ),
                    "error": "corrupt_file",
                    "message": "A scanned-data frame has a non-positive "
                    "lateral step size.",
                }
            ],
            "name": F.get(
                _SIGNAL_BY_ADC_INDEX, F.str(C.item.scan_vars.adc_index), "Unknown"
            ),
            "dim": 2,
            "nb_grid_pts": Tup(
                C.item.frame_mode.nb_grid_pts_x, C.item.frame_mode.nb_grid_pts_y
            ),
            "physical_sizes": Tup(
                C.item.frame_mode.nb_grid_pts_x * C.item.axis_scales.x_step,
                C.item.frame_mode.nb_grid_pts_y * C.item.axis_scales.y_step,
            ),
            "unit": _xy_unit,
            # The data is reported in the z-axis unit; convert to the
            # lateral unit
            "height_scale_factor": F.unit_conversion_factor(_z_unit, _xy_unit),
            "uniform": True,
            "data": C.item.data,
            "info": {
                "acquisition_time": F.make_datetime(
                    C.item.header.year,
                    C.item.header.month,
                    C.item.header.day,
                    C.item.header.hour,
                    C.item.header.minute,
                    C.item.header.second,
                ),
                "instrument": {"vendor": "NT-MDT"},
                "raw_metadata": {
                    "header": C.item.header,
                    "axis_scales": C.item.axis_scales,
                    "scan_vars": C.item.scan_vars,
                    "frame_mode": C.item.frame_mode,
                },
            },
        }
    ]
