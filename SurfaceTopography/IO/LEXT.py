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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/lextfile.c
#

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import TIFFContainer
from .expr import C, F, Tup
from .Reader import Check, CompoundLayout, DeclarativeReaderBase, MagicMatch

# The height data lives in the page marked HEIGHT; the measurement
# metadata is XML embedded in the image description (and the Exif
# device-setting description) of the overview page
_height_page = F.match_records(
    C.pages, ["tags", "ImageDescription"], "HEIGHT"
)[0]
_metadata_page = F.match_records_containing(
    C.pages, ["tags", "ImageDescription"], "TiffTagDescData"
)[0]

_image_desc = F.parse_xml(_metadata_page.tags.ImageDescription)[
    "TiffTagDescData"
]
_data_desc = F.parse_xml(
    _metadata_page.tags.ExifTag.DeviceSettingDescription
)["ExifTagDescData"]

_height_info = _image_desc.HeightInfo
_calibration = _data_desc.ImageCommonSettingsInfo

_nb_grid_pts_x = _metadata_page.shape[0]
_nb_grid_pts_y = _metadata_page.shape[1]


def _physical_size(nb_grid_pts, data_per_pixel, calibration_value):
    # All lengths appear to be picometers; we report micrometers
    return (
        1e-12
        * nb_grid_pts
        * F.float(_height_info[data_per_pixel])
        * F.float(_calibration[calibration_value])
    )


class LEXTReader(DeclarativeReaderBase):
    _format = "lext"
    _mime_types = ["image/tiff"]
    _file_extensions = ["lext"]

    _name = "Olympus LEXT"
    _description = """
TIFF-based file format of Olympus instruments.
"""

    # TIFF magic bytes (little- and big-endian)
    _MAGIC_TIFF = (b"II*\x00", b"MM\x00*")

    # A TIFF magic can only tell that the file *may* be a LEXT file;
    # detection needs to try parsing
    _magic = None

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 4:
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC_TIFF):
            # TIFF file, could be an Olympus LEXT file
            return MagicMatch.MAYBE
        return MagicMatch.NO

    _file_layout = CompoundLayout(
        [
            TIFFContainer(),
            Check(
                F.len(
                    F.match_records(
                        C.pages, ["tags", "ImageDescription"], "HEIGHT"
                    )
                )
                == 1,
                FileFormatMismatch,
                "TIFF file does not contain a HEIGHT page, so this is not "
                "an Olympus LEXT file.",
            ),
            Check(
                F.len(
                    F.match_records_containing(
                        C.pages, ["tags", "ImageDescription"],
                        "TiffTagDescData",
                    )
                )
                == 1,
                FileFormatMismatch,
                "TIFF file does not contain LEXT measurement metadata, so "
                "this is not an Olympus LEXT file.",
            ),
            Check(
                (
                    F.int(_height_info.HeightDataUnitX)
                    == F.int(_height_info.HeightDataUnitZ)
                )
                & (
                    F.int(_height_info.HeightDataUnitY)
                    == F.int(_height_info.HeightDataUnitZ)
                ),
                CorruptFile,
                "Units in x-, y- and height direction appear to differ.",
            ),
        ]
    )

    _channel_bindings = [
        {
            "name": "default",
            "dim": 2,
            "nb_grid_pts": Tup(_nb_grid_pts_x, _nb_grid_pts_y),
            "physical_sizes": Tup(
                _physical_size(
                    _nb_grid_pts_x,
                    "HeightDataPerPixelX",
                    "MakerCalibrationValueX",
                ),
                _physical_size(
                    _nb_grid_pts_y,
                    "HeightDataPerPixelY",
                    "MakerCalibrationValueY",
                ),
            ),
            "height_scale_factor": 1e-12
            * F.float(_height_info.HeightDataPerPixelZ)
            * F.float(_calibration.MakerCalibrationValueZ),
            "uniform": True,
            "unit": "µm",
            "info": {
                "raw_metadata": {
                    "pages": F.pluck(C.pages, "tags"),
                    "measurement": _image_desc,
                    "device_settings": _data_desc,
                },
            },
            "data": _height_page.image,
        }
    ]
