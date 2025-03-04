#
# Copyright 2018-2023 Lars Pastewka
#           2018-2021 Michael Röttger
#           2019-2020 Antoine Sanner
#           2019 Kai Haase
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

import re
from collections import defaultdict

import numpy as np

from ..Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import length_units
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from . import ReaderBase
from .common import CHANNEL_NAME_INFO_KEY, OpenFromAny, text
from .FromFile import make_wrapped_reader
from .Reader import ChannelInfo


@text()
def read_matrix(
    fobj, physical_sizes=None, unit=None, height_scale_factor=None, periodic=False
):
    """
    Reads a surface profile from a text file and presents in in a
    SurfaceTopography-conformant manner. No additional parsing of
    meta-information is carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    arr = np.loadtxt(fobj)
    if physical_sizes is None:
        surface = Topography(arr, arr.shape, periodic=periodic, unit=unit)
    else:
        surface = Topography(arr, physical_sizes, periodic=periodic, unit=unit)
    if height_scale_factor is not None:
        surface = surface.scale(height_scale_factor)
    return surface


MatrixReader = make_wrapped_reader(
    read_matrix,
    class_name="MatrixReader",
    format="matrix",
    mime_types=["text/plain"],
    file_extensions=["txt", "asc", "dat"],
    name="Plain text (matrix)",
)

# Regex for floating-point numbers
_float_regex = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


# Convert to string, but empty strings to None
def to_str(x):
    if x == "":
        return None
    return str(x)


class AscReader(ReaderBase):
    _format = "asc"
    _mime_types = ["text/plain"]
    _file_extensions = ["txt", "asc", "dat"]

    _name = "Plain text"
    _description = """
Imports plain text files. The reader supports parsing file headers for
additional metadata. This allows to specify the physical size of the
topography and the unit. In particular, it supports reading ASCII files
exported from Wyko, SPIP and Gwyddion.

Topography data stored in plain text (ASCII) format needs to be stored in a
matrix format. Each row contains the height information for subsequent
points in x-direction separated by a whitespace. The next row belong to the
following y-coordinate. Note that if the file has three or less columns, it
will be interpreted as a topography stored in a coordinate format (the three
columns contain the x, y and z coordinates of the same points). The smallest
topography that can be provided in this format is therefore 4 x 1.

When writing your own ASCII files, we recommend to prepend the header with a
'#'. The following file is an example that contains 4 x 3 data points:
```
# Channel: Main
# Width: 10 µm
# Height: 10 µm
# Value units: m
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
 9.0 10.0 11.0 12.0
```
"""

    # Regular expressions for parsing the header
    _metadata_regex = [
        # File format flavors
        (re.compile(r"Wyko ASCII Data File Format\s*"), ("wyko",), ("format_flavor",)),
        # Resolution keywords
        (
            re.compile(r"\b(?:x-pixels|h)\b\s*=\s*(?P<nb_grid_pts_y>[0-9]+)"),
            (int,),
            ("nb_grid_pts_y",),
        ),
        (
            re.compile(r"\b(?:y-pixels|w)\b\s*=\s*(?P<nb_grid_pts_x>[0-9]+)"),
            (int,),
            ("nb_grid_pts_x",),
        ),
        (
            re.compile(r"\b(?:X Size|h)\b\s*(?P<nb_grid_pts_y>[0-9]+)"),
            (int,),
            ("nb_grid_pts_y",),
        ),
        (
            re.compile(r"\b(?:Y Size|h)\b\s*(?P<nb_grid_pts_x>[0-9]+)"),
            (int,),
            ("nb_grid_pts_x",),
        ),
        # Size keywords
        (
            re.compile(
                r"\b(?:x-length|Width|Breite)\b\s*(?:=|\:)\s*(?P<physical_size_x>"
                + _float_regex
                + ")(?P<xunit>.*)"
            ),
            (float, to_str),
            ("physical_size_x", "xunit"),
        ),
        (
            re.compile(
                r"\b(?:y-length|Height|Höhe)\b\s*(?:=|\:)\s*(?P<physical_size_y>"
                + _float_regex
                + ")(?P<yunit>.*)"
            ),
            (float, to_str),
            ("physical_size_y", "yunit"),
        ),
        (
            re.compile(
                r"\b(?:Pixel_size|h)\b\s*7\s*[0-9]+\s*(?P<wyko_pixel_size>"
                + _float_regex
                + ")"
            ),
            (float,),
            ("wyko_pixel_size",),
        ),
        (
            re.compile(
                r"\b(?:Aspect|h)\b\s*7\s*[0-9]+\s*(?P<wyko_aspect_ratio>"
                + _float_regex
                + ")"
            ),
            (float,),
            ("wyko_aspect_ratio",),
        ),
        # Unit keywords
        (re.compile(r"\b(?:x-unit)\b\s*(?:=|\:)\s*(\w+)"), (to_str,), ("xunit",)),
        (re.compile(r"\b(?:y-unit)\b\s*(?:=|\:)\s*(\w+)"), (to_str,), ("yunit",)),
        (
            re.compile(r"\b(?:z-unit|Value units)\b\s*(?:=|\:)\s*(?P<zunit>\w+)"),
            (to_str,),
            ("zunit",),
        ),
        # Scale factor keywords
        (
            re.compile(
                r"(?:pixel\s+size)\s*=\s*(?P<xfac>" + _float_regex + ")(?P<xunit>.*)"
            ),
            (float, to_str),
            ("xfac", "xunit"),
        ),
        (
            re.compile(
                (
                    r"(?:height\s+conversion\s+factor\s+\(->\s+(?P<zunit>.*)\))\s*="
                    r"\s*(?P<zfac>" + _float_regex + ")"
                )
            ),
            (
                to_str,
                float,
            ),
            (
                "zunit",
                "zfac",
            ),
        ),
        (
            re.compile(
                r"\b(?:Mult|h)\b\s*7\s*[0-9]+\s*(?P<wyko_mult>" + _float_regex + ")"
            ),
            (float,),
            ("wyko_mult",),
        ),
        (
            re.compile(
                r"\b(?:Wavelength|h)\b\s*7\s*[0-9]+\s*(?P<wyko_wavelength>"
                + _float_regex
                + ")"
            ),
            (float,),
            ("wyko_wavelength",),
        ),
        # Channel name keywords
        (
            re.compile(
                r"\b(?:Channel|Kanal)\b\s*(?:=|\:)\s*(?P<channel_name>[\w|\s]+)"
            ),
            (to_str,),
            ("channel_name",),
        ),
    ]

    _undefined_data_keywords = ["bad", "nan", "inf", "infinite"]

    @classmethod
    def to_float(cls, s):
        if s.lower() in cls._undefined_data_keywords:
            # This is a placeholder for missing data
            return np.nan
        return float(s)

    def parse_data(self, line):
        return [self.to_float(val) for val in line.split()]

    def parse_metadata(self, line):
        for reg, funs, keys in self._metadata_regex:
            match = reg.search(line)
            if match is not None:
                for fun, key in zip(funs, keys):
                    if callable(fun):
                        self._metadata[key] = fun(match.group(key).strip())
                    else:
                        self._metadata[key] = fun

        # Handling of special metadata
        if self._metadata.get("format_flavor") == "wyko":
            s = line.split()
            if len(s) > 0:
                self._metadata["channel_name"] = s[0].strip()
                return

    def __init__(self, file_path):
        # Open file and parse
        self._channel_names = []
        self._metadata = defaultdict(None)
        self._data = defaultdict(list)
        self._dim = 2
        with OpenFromAny(file_path, "r") as fobj:
            for line in fobj:
                try:
                    # Try interpreting the line as data
                    data_in_line = self.parse_data(line)
                except ValueError:
                    # If this fails, we look for metadata keys
                    self.parse_metadata(line)
                else:
                    if data_in_line is not None and data_in_line != []:
                        channel_name = self._metadata.get("channel_name", "Default")
                        if channel_name not in self._channel_names:
                            self._channel_names += [channel_name]
                        self._data[channel_name] += [data_in_line]

            nb_grid_pts_x = self._metadata.get("nb_grid_pts_x")
            nb_grid_pts_y = self._metadata.get("nb_grid_pts_y")
            for channel_name in self._channel_names:
                data = np.array(self._data[channel_name]).T
                if data.shape[0] == 1:
                    self._dim = 1
                    data = np.ravel(data)
                self._data[channel_name] = data
                try:
                    nx, ny = data.shape
                except ValueError:
                    if nb_grid_pts_y is not None:
                        raise Exception(
                            "This file has just a single column and is hence a line "
                            f"scan, but the files metadata specifies {nb_grid_pts_y} "
                            "grid points in y-direction."
                        )
                    (nx,) = data.shape
                    ny = None
                else:
                    if nx == 2 or ny == 2:
                        raise Exception(
                            "This file has just two rows or two columns and is more "
                            "likely a line scan than a map."
                        )
                    if nb_grid_pts_y is not None and nb_grid_pts_y != ny:
                        raise Exception(
                            f"The number of columns (={ny}) of the topography from the "
                            f"file '{fobj}' does not match the number of grid points "
                            f"in the file's metadata (={nb_grid_pts_y})."
                        )
                if nb_grid_pts_x is not None and nb_grid_pts_x != nx:
                    raise Exception(
                        f"The number of rows (={nx}) of the topography from the file "
                        f"'{fobj}' does not match the number of grid points in the "
                        f"file's metadata (={nb_grid_pts_x})."
                    )

            # Set grid points if not in metadata
            if nb_grid_pts_x is None:
                nb_grid_pts_x = nx
            if nb_grid_pts_y is None:
                nb_grid_pts_y = ny

            # Get physical sizes
            physical_size_x = self._metadata.get("physical_size_x")
            physical_size_y = self._metadata.get("physical_size_y")

            # Handle scale factors
            xfac = self._metadata.get("xfac")
            yfac = self._metadata.get("yfac")
            zfac = self._metadata.get("zfac")
            if xfac is not None and yfac is None:
                yfac = xfac
            elif xfac is None and yfac is not None:
                xfac = yfac
            if xfac is not None:
                if physical_size_x is None:
                    if nb_grid_pts_x is not None:
                        physical_size_x = xfac * nb_grid_pts_x
                else:
                    physical_size_x *= xfac
            if yfac is not None:
                if physical_size_y is None:
                    if nb_grid_pts_y is not None:
                        physical_size_y = yfac * nb_grid_pts_y
                else:
                    physical_size_y *= yfac

            # Handle units -> convert to target unit
            xunit = self._metadata.get("xunit")
            yunit = self._metadata.get("yunit")
            zunit = self._metadata.get("zunit")
            if xunit is None and zunit is not None:
                xunit = zunit
            if yunit is None and zunit is not None:
                yunit = zunit

            if self._metadata.get("format_flavor") == "wyko":
                # Wyko files have a special scale factor
                wyko_pixel_size = self._metadata.get("wyko_pixel_size")
                wyko_aspect_ratio = self._metadata.get("wyko_aspect_ratio", 1)
                wyko_mult = self._metadata.get("wyko_mult")
                wyko_wavelength = self._metadata.get("wyko_wavelength")
                if wyko_mult is not None and wyko_wavelength is not None:
                    zfac = wyko_wavelength / wyko_mult

                if wyko_pixel_size is not None:
                    physical_size_x = (
                        wyko_pixel_size * wyko_aspect_ratio * nb_grid_pts_x
                    )
                    physical_size_y = wyko_pixel_size * nb_grid_pts_y

                # Wyko files have special units
                if xunit is None:
                    xunit = "mm"
                else:
                    raise CorruptFile(
                        "This is a Wyko file, but it appears to have unit metadata."
                    )
                if yunit is None:
                    yunit = "mm"
                else:
                    raise CorruptFile(
                        "This is a Wyko file, but it appears to have unit metadata."
                    )
                if zunit is None:
                    zunit = "nm"
                else:
                    raise CorruptFile(
                        "This is a Wyko file, but it appears to have unit metadata."
                    )

            unit = zunit
            if unit is not None:
                if xunit is not None:
                    if physical_size_x is not None:
                        physical_size_x *= length_units[xunit] / length_units[unit]
                if yunit is not None:
                    if physical_size_y is not None:
                        physical_size_y *= length_units[yunit] / length_units[unit]
                if zunit is not None:
                    if zfac is None:
                        if length_units[zunit] != length_units[unit]:
                            zfac = length_units[zunit] / length_units[unit]
                    else:
                        zfac *= length_units[zunit] / length_units[unit]

            # Store processed metadata
            self._nb_grid_pts = None
            if nb_grid_pts_x is not None:
                if nb_grid_pts_y is not None:
                    self._nb_grid_pts = (nb_grid_pts_x, nb_grid_pts_y)
                else:
                    self._nb_grid_pts = (nb_grid_pts_x,)
            self._physical_sizes = None
            if physical_size_x is not None:
                if physical_size_y is not None:
                    self._physical_sizes = (physical_size_x, physical_size_y)
                else:
                    self._physical_sizes = (physical_size_x,)
            self._unit = unit
            self._height_scale_factor = zfac

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                i,  # channel index
                name=name,
                dim=self._dim,
                nb_grid_pts=self._nb_grid_pts,
                physical_sizes=self._physical_sizes,
                uniform=True,
                unit=self._unit,
                height_scale_factor=self._height_scale_factor,
                info={CHANNEL_NAME_INFO_KEY: name, "raw_metadata": self._metadata},
            )
            for i, name in enumerate(self._channel_names)
        ]

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index < 0 or channel_index > len(self._channel_names):
            raise RuntimeError(
                f"There are only {len(self._channel_names)} channels, but channel "
                f"index is {channel_index}."
            )

        physical_sizes = self._check_physical_sizes(
            physical_sizes, self._physical_sizes
        )

        if height_scale_factor is not None and self._height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None and self._unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        _info = info.copy()
        _info["raw_metadata"] = self._metadata

        # handle channel name
        # we use the info dict here to transfer the channel name
        channel_name = self._channel_names[channel_index]
        _info[CHANNEL_NAME_INFO_KEY] = channel_name

        data = self._data[channel_name]
        if np.sum(np.isnan(data)) > 0:
            data = np.ma.masked_invalid(data)
        if self._dim == 1:
            topography = UniformLineScan(
                data,
                physical_sizes,
                unit=unit or self._unit,
                info=_info,
                periodic=periodic,
            )
        else:
            topography = Topography(
                data,
                physical_sizes,
                unit=unit or self._unit,
                info=_info,
                periodic=periodic,
            )
        if height_scale_factor is not None or self._height_scale_factor is not None:
            topography = topography.scale(
                height_scale_factor or self._height_scale_factor
            )
        return topography


def write_matrix(self, fname):
    """
    Saves the topography using `np.savetxt`. Warning: This only saves
    the heights; the physical_sizes is not contained in the file
    """
    np.savetxt(fname, self.heights())


# Register analysis functions from this module
UniformTopographyInterface.register_function("to_matrix", write_matrix)
