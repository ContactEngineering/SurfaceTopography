#
# Copyright 2021-2026 Lars Pastewka
#           2021 Michael Röttger
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
Reader for Keyence ZON files.
"""

import dateutil.parser
import numpy as np
import zstandard

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, Validate, XMLStructure, ZipContainer
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, MagicMatch

# Thanks to @mcmalburg (https://github.com/mcmalburg) for reverse engineering the
# format. See discussion here https://github.com/gabeguss/Keyence/issues/2

# ZON files start with a magic, followed by a BMP thumbnail of the measurement,
# followed by a ZIP archive that holds the actual data. KPK1 files additionally
# wrap the content of each archive member into a zstandard compression layer.
_MAGIC0 = "KPK0"
_MAGIC1 = "KPK1"
_MAGIC = {_MAGIC0, _MAGIC1}

# The files within the ZIP archive are named using UUIDs. Some of these
# UUIDs are fixed and contain the same information in each of these files.

# This file contains height data
_HEIGHT_DATA_UUID = "4cdb0c75-5706-48cc-a9a1-adf395d609ae"

# This file contains a per-pixel validity mask for the height data
_VALID_PIXEL_UUID = "9e7ca6dd-0e6a-4d7b-8986-f0a7b773c916"

# This contains information on unit conversion
_UNIT_UUID = "686613b8-27b5-4a29-8ffc-438c2780873e"

# Additional user calibration information
_CALIBRATION_UUID = "7d5ea102-58f4-4c0f-b5ba-c8b71be6a4ae"

# This contains an inventory of *image* data
_INVENTORY_UUID = "772e6d38-40aa-4590-85d3-b041fa243570"

# Device information
_DEVICE_UUID = "cbc1d39b-215f-4d56-a20e-cf8e5f570755"

# Scan information
_SCAN_UUID = "1d35862b-3b48-4df2-9235-833ef590b043"

# The per-pixel validity mask carries two kinds of information (reverse
# engineered from VR-3200 and VR-6200 files, no official documentation):
#
# The low two bits encode why a pixel carries no valid height information:
#   0x01: out of measurement range / no measurement; the stored height is
#         garbage and often sits at the sensor floor (around -590 um)
#   0x02: excluded from measurement by the operator ("do not measure"); the
#         stored height is a placeholder, typically zero or a constant
#
# The high bits (0x40 and 0x80 have been observed, also in combination with
# each other and with 0x01) appear to be quality annotations on pixels that
# still carry plausible heights: their height distribution is statistically
# indistinguishable from that of 0x00 pixels. Their exact meaning is unknown;
# candidates are interpolated/filled values, low-confidence measurements or
# stitching seams. (In one sample file, a 0x80 pixel holds exactly the same
# placeholder constant as the 0x02 regions, hinting that at least some of
# these pixels were filled in by post-processing rather than measured.)
#
# We therefore treat a pixel as undefined only if one of the low two bits is
# set and keep quality-flagged pixels. The raw mask remains accessible via
# `reader.metadata.valid_pixel_data.data(stream)` for more aggressive
# filtering downstream.
_INVALID_PIXEL_BITS = 0x03


def _array_member(name, dtype, element_size):
    """
    Layout of the binary array members of the ZON archive: a 16-byte header
    followed by row-major data with rows padded to `row_bytes` bytes.
    """
    dtype = np.dtype(dtype)
    return CompoundLayout(
        [
            BinaryStructure(
                [
                    (
                        "nb_grid_pts_x",
                        "i",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "nb_grid_pts_y",
                        "i",
                        Validate(lambda x, context: x > 0, CorruptFile),
                    ),
                    (
                        "element_size",
                        "i",
                        Validate(element_size, UnsupportedFormatFeature),
                    ),
                    (
                        "row_bytes",
                        "i",
                        Validate(
                            lambda x, context: x % dtype.itemsize == 0
                            and x >= context.nb_grid_pts_x * element_size,
                            CorruptFile,
                        ),
                    ),
                ],
                byte_order="<",
                name="header",
            ),
            BinaryArray(
                "data",
                lambda context: (
                    context.header.nb_grid_pts_y,
                    context.header.row_bytes // dtype.itemsize,
                ),
                lambda context: dtype,
                # Crop the row padding and transpose to (x, y) indexing
                conversion_fun=lambda arr, context: arr[
                    :, : context.header.nb_grid_pts_x
                ].T,
            ),
        ],
        name=name,
    )


class ZONReader(DeclarativeReaderBase):
    _format = "zon"
    _mime_types = ["application/x-keyence-zon"]
    _file_extensions = ["zon"]

    _name = "Keyence ZON"
    _description = """
This reader open ZON files that are written by some Keyence instruments.
"""

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 4:
            return MagicMatch.MAYBE
        if buffer[:4].decode("latin1") in _MAGIC:
            return MagicMatch.YES
        return MagicMatch.NO

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    ("magic", "4s", Validate(lambda x, context: x in _MAGIC, FileFormatMismatch)),
                    ("bmp_size", "L"),  # Size of the BMP thumbnail, which we skip
                ],
                byte_order="<",
                name="header",
            ),
            ZipContainer(
                [
                    (_DEVICE_UUID, XMLStructure(name="device")),
                    (
                        _SCAN_UUID,
                        XMLStructure(
                            name="scan",
                            converters={"ScanDateTime": dateutil.parser.parse},
                        ),
                    ),
                    (
                        _UNIT_UUID,
                        XMLStructure(
                            name="calibration",
                            converters={"MeterPerPixel": float, "MeterPerUnit": float},
                        ),
                    ),
                    (_HEIGHT_DATA_UUID, _array_member("height_data", "<i4", 4)),
                    # The validity mask may be missing from the archive
                    (_VALID_PIXEL_UUID, _array_member("valid_pixel_data", "u1", 1), True),
                ],
                stream_filter=lambda stream_obj, context: (
                    zstandard.ZstdDecompressor().stream_reader(stream_obj)
                    if context.header.magic == _MAGIC1
                    else stream_obj
                ),
            ),
        ]
    )

    def _validate_metadata(self):
        mask_data = self._metadata.get("valid_pixel_data")
        if mask_data is not None:
            height_header = self._metadata.height_data.header
            mask_header = mask_data.header
            if (mask_header.nb_grid_pts_x, mask_header.nb_grid_pts_y) != (
                height_header.nb_grid_pts_x,
                height_header.nb_grid_pts_y,
            ):
                raise CorruptFile(
                    "Dimensions of the validity mask do not match the height data."
                )

    def _read_height_data(self, stream_obj):
        height_data = self._metadata.height_data.data(stream_obj)
        mask_data = self._metadata.get("valid_pixel_data")
        if mask_data is not None:
            mask = mask_data.data(stream_obj)
            height_data = np.ma.masked_array(
                height_data, mask=(mask & _INVALID_PIXEL_BITS) != 0
            )
        return height_data

    @property
    def channels(self):
        header = self._metadata.height_data.header
        calibration = self._metadata.calibration
        meter_per_pixel = calibration.XYCalibration.MeterPerPixel
        meter_per_unit = calibration.ZCalibration.MeterPerUnit

        return [
            ChannelInfo(
                self,
                0,
                name="default",
                dim=2,
                nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
                physical_sizes=(
                    header.nb_grid_pts_x * meter_per_pixel,
                    header.nb_grid_pts_y * meter_per_pixel,
                ),
                height_scale_factor=meter_per_unit,
                unit="m",
                uniform=True,
                info={
                    "acquisition_time": self._metadata.scan.ScanDateTime,
                    "instrument": {
                        "name": self._metadata.device.DeviceModelName,
                        "vendor": "Keyence",
                        "serial": self._metadata.device.DeviceSerialId,
                    },
                    "raw_metadata": {
                        "data_uuid": _HEIGHT_DATA_UUID,
                        "meter_per_pixel": meter_per_pixel,
                        "meter_per_unit": meter_per_unit,
                    },
                },
                tags={"reader": self._read_height_data},
            )
        ]
