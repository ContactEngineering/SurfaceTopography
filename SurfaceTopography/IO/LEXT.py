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

# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/lextfile.c

import enum

import dateutil.parser
import xmltodict

from tiffile import TiffFile, TiffFileError

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class LEXTReader(ReaderBase):
    _format = "lext"
    _mime_types = ["image/tiff"]
    _file_extensions = ["lext"]

    _name = "Olympus LEXT"
    _description = """
TIFF-based file format of Olympus instruments.
"""

    _tag_readers = {306: lambda value: str(dateutil.parser.parse(value))}

    @classmethod
    def _tag_to_dict(cls, key, value):
        # Try dedicated tag readers first
        try:
            return {value.name: cls._tag_readers[key](value.value)}
        except KeyError:
            pass

        # Generic treatment if failed
        if isinstance(value.value, enum.Enum):
            return {value.name: [value.value.name, value.value.value]}
        else:
            name = value.name
            value = value.value
            try:
                if value.startswith(r"<?xml"):
                    value = xmltodict.parse(value)
            except AttributeError:
                pass
            if isinstance(value, tuple):
                value = list(value)
            return {name: value}

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self._file_path = file_path

        # All units appear to be picometers, also some files report 'INCH' as resolution unit
        self._unit = "µm"

        with OpenFromAny(self._file_path, "rb") as f:
            try:
                with TiffFile(f) as t:
                    # Go through all pages and see what is in there
                    metadata = []
                    for i, p in enumerate(t.pages):
                        page_metadata = {}
                        for key, value in p.tags.items():
                            tag_metadata = self._tag_to_dict(key, value)
                            if "ExifTag" in tag_metadata:
                                updated_metadata = {}
                                for key, value in tag_metadata["ExifTag"].items():
                                    try:
                                        if value.startswith("<?xml"):
                                            updated_metadata[key] = xmltodict.parse(value)
                                    except AttributeError:
                                        pass
                                tag_metadata["ExifTag"].update(updated_metadata)
                            page_metadata.update(tag_metadata)

                        if page_metadata["ImageDescription"] == "HEIGHT":
                            # This is the actual height data
                            self._height_page_index = i
                        elif "TiffTagDescData" in page_metadata["ImageDescription"]:
                            # Get image dimension
                            try:
                                nx, ny, _ = p.shape
                            except ValueError:
                                nx, ny = p.shape

                            # This is some metadata
                            image_desc = page_metadata["ImageDescription"]["TiffTagDescData"]
                            data_desc = page_metadata["ExifTag"]["DeviceSettingDescription"]["ExifTagDescData"]
                            self._height_scale_factor = (
                                    1e-12
                                    * float(image_desc["HeightInfo"]["HeightDataPerPixelZ"])
                                    * float(data_desc["ImageCommonSettingsInfo"]["MakerCalibrationValueZ"])
                            )
                            self._nb_grid_pts = (nx, ny)
                            self._physical_sizes = (
                                1e-12 * nx
                                * float(image_desc["HeightInfo"]["HeightDataPerPixelX"])
                                * float(data_desc["ImageCommonSettingsInfo"]["MakerCalibrationValueX"]),
                                1e-12 * ny
                                * float(image_desc["HeightInfo"]["HeightDataPerPixelY"])
                                * float(data_desc["ImageCommonSettingsInfo"]["MakerCalibrationValueY"])
                            )
                            self._x_unit_index = int(image_desc["HeightInfo"]["HeightDataUnitX"])
                            self._y_unit_index = int(image_desc["HeightInfo"]["HeightDataUnitY"])
                            self._height_unit_index = int(image_desc["HeightInfo"]["HeightDataUnitZ"])

                            if self._x_unit_index != self._height_unit_index or \
                                    self._y_unit_index != self._height_unit_index:
                                raise CorruptFile("Units in x-, y- and height direction appear to differ")

                        metadata += [page_metadata]

                    self._info = {"raw_metadata": metadata}
            except TiffFileError:
                raise FileFormatMismatch("This is not a TIFF file, so it cannot be an Olympus LEXT file.")

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name="default",
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=self._height_scale_factor,
                            uniform=True,
                            info=self._info,
                            unit=self._unit)]

    def topography(
            self,
            channel_index=None,
            physical_sizes=None,
            height_scale_factor=None,
            unit=None,
            info={},
            periodic=None,
            subdomain_locations=None,
            nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(f"There is only a single channel. Channel index must be {self._default_channel_index}.")

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        with OpenFromAny(self._file_path, "rb") as f:
            with TiffFile(f) as t:
                height_data = t.pages[self._height_page_index].asarray()
                assert height_data.shape == self._nb_grid_pts

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(
            height_data,
            self._physical_sizes,
            unit=self._unit,
            periodic=False if periodic is None else periodic,
            info=_info,
        )
        return topo.scale(self._height_scale_factor)
