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
from ..Support.UnitConversion import length_units, mangle_length_unit_utf8
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
exported from Wyko, SPIP, Attocube, Nova and Gwyddion. Gwyddion exports with
translated headers and exports of multiple images concatenated into a single
file are supported; channels that do not contain height information (e.g.
phase or voltage) are ignored.

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

    # Header labels written by Gwyddion's plain-text matrix export
    # (asciiexport). Gwyddion translates these labels, so we accept the
    # labels of all its user interface languages.
    _gwy_channel_labels = "Channel|Kanál|Kanal|Canal|Canale|チャネル|채널|Канал"
    _gwy_width_labels = (
        "Width|Šířka|Breite|Anchura|Largeur|Larghezza|幅|폭|Largura|Ширина"
    )
    _gwy_height_labels = "Height|Výška|Höhe|Altura|Hauteur|Altezza|高さ|높이|Высота"
    _gwy_value_unit_labels = (
        "Value units|Jednotky hodnot|Einheiten|Unités|unità valore|値の単位|"
        "값 단위|Unidades de valor|Единицы измерения"
    )
    # Key-value separators; Japanese translations use a full-width colon
    _sep = r"\s*(?:=|:|：)\s*"

    # Regular expressions for parsing the header
    _metadata_regex = [
        # File format flavors
        (re.compile(r"Wyko ASCII Data File Format\s*"), ("wyko",), ("format_flavor",)),
        # The Wyko magic line is followed by three flags; the second one
        # indicates whether the data is already stored in real units (nm)
        # rather than in units of the wavelength.
        (
            re.compile(
                r"Wyko ASCII Data File Format\s+[0-9]+\s+(?P<wyko_real_units>[0-9]+)"
            ),
            (int,),
            ("wyko_real_units",),
        ),
        # SPIP ASCII export
        (re.compile(r"^#\s*File Format\s*=\s*ASCII\s*$"), ("spip",), ("format_flavor",)),
        # Resolution keywords. SPIP (and Attocube) `x-pixels` is the number of
        # values per line, i.e. the number of points along x; `y-pixels` is the
        # number of lines.
        (
            re.compile(r"\bx-pixels\b\s*(?:=|:)\s*(?P<nb_grid_pts_x>[0-9]+)"),
            (int,),
            ("nb_grid_pts_x",),
        ),
        (
            re.compile(r"\by-pixels\b\s*(?:=|:)\s*(?P<nb_grid_pts_y>[0-9]+)"),
            (int,),
            ("nb_grid_pts_y",),
        ),
        # `h` (height, number of lines) and `w` (width, values per line)
        (
            re.compile(r"\bh\b\s*=\s*(?P<nb_grid_pts_y>[0-9]+)"),
            (int,),
            ("nb_grid_pts_y",),
        ),
        (
            re.compile(r"\bw\b\s*=\s*(?P<nb_grid_pts_x>[0-9]+)"),
            (int,),
            ("nb_grid_pts_x",),
        ),
        # Nova ASCII export: NX values per line, NY lines
        (
            re.compile(r"^\s*NX\s*=\s*(?P<nb_grid_pts_x>[0-9]+)\s*$"),
            (int,),
            ("nb_grid_pts_x",),
        ),
        (
            re.compile(r"^\s*NY\s*=\s*(?P<nb_grid_pts_y>[0-9]+)\s*$"),
            (int,),
            ("nb_grid_pts_y",),
        ),
        # Wyko ASCII: each line of the file is one `X Size` position
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
                rf"\b(?:x-length|{_gwy_width_labels})\b{_sep}(?P<physical_size_x>"
                + _float_regex
                + ")(?P<xunit>.*)"
            ),
            (float, to_str),
            ("physical_size_x", "xunit"),
        ),
        (
            re.compile(
                rf"\b(?:y-length|{_gwy_height_labels})\b{_sep}(?P<physical_size_y>"
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
        (
            re.compile(r"\b(?:x-unit)\b\s*(?:=|\:)\s*(?P<xunit>\w+)"),
            (to_str,),
            ("xunit",),
        ),
        (
            re.compile(r"\b(?:y-unit)\b\s*(?:=|\:)\s*(?P<yunit>\w+)"),
            (to_str,),
            ("yunit",),
        ),
        (
            re.compile(
                rf"\b(?:z-unit|{_gwy_value_unit_labels})\b{_sep}(?P<zunit>\w+)"
            ),
            (to_str,),
            ("zunit",),
        ),
        # Nova ASCII export: lateral unit (used for x and y) and data unit
        (re.compile(r"^\s*Unit X\s*=\s*(?P<xunit>\w+)"), (to_str,), ("xunit",)),
        (re.compile(r"^\s*Unit Data\s*=\s*(?P<zunit>\w+)"), (to_str,), ("zunit",)),
        # Scale factor keywords
        (
            re.compile(
                r"(?:pixel\s+size)\s*=\s*(?P<xfac>" + _float_regex + ")(?P<xunit>.*)"
            ),
            (float, to_str),
            ("xfac", "xunit"),
        ),
        # Nova ASCII export: pixel sizes
        (
            re.compile(r"^\s*Scale X\s*=\s*(?P<xfac>" + _float_regex + r")\s*$"),
            (float,),
            ("xfac",),
        ),
        (
            re.compile(r"^\s*Scale Y\s*=\s*(?P<yfac>" + _float_regex + r")\s*$"),
            (float,),
            ("yfac",),
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
        # SPIP ASCII: conversion factor from stored values to nanometers
        # (used if there is no `z-unit`)
        (
            re.compile(r"\bBit2nm\b\s*=\s*(?P<spip_bit2nm>" + _float_regex + ")"),
            (float,),
            ("spip_bit2nm",),
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
                rf"\b(?:{_gwy_channel_labels})\b{_sep}(?P<channel_name>[\w|\s]+)"
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
        # Running dictionary of all metadata parsed so far. Each channel
        # receives a snapshot of this dictionary (see below), such that files
        # that contain several blocks with individual headers (e.g. Gwyddion
        # exports of multiple images) are interpreted correctly.
        self._metadata = {}
        channel_metadata = {}
        metadata_changed = False
        self._data = defaultdict(list)
        with OpenFromAny(file_path, "r") as fobj:
            channel_name = None
            for line in fobj:
                try:
                    # Try interpreting the line as data
                    data_in_line = self.parse_data(line)
                except ValueError:
                    # If this fails, we look for metadata keys
                    self.parse_metadata(line)
                    metadata_changed = True
                else:
                    if data_in_line is not None and data_in_line != []:
                        channel_name = self._metadata.get("channel_name", "Default")
                        if channel_name not in self._channel_names:
                            self._channel_names += [channel_name]
                        if metadata_changed or channel_name not in channel_metadata:
                            channel_metadata[channel_name] = self._metadata.copy()
                            metadata_changed = False
                        self._data[channel_name] += [data_in_line]
            if metadata_changed and channel_name is not None:
                # Metadata trailing the data belongs to the last channel
                channel_metadata[channel_name] = self._metadata.copy()

        self._channel_properties = {}
        for channel_name in self._channel_names:
            properties = self._process_channel(
                fobj, self._data[channel_name], channel_metadata[channel_name]
            )
            if properties is None:
                # Not a topography channel, ignore
                del self._data[channel_name]
            else:
                self._data[channel_name] = properties.pop("data")
                self._channel_properties[channel_name] = properties
        self._channel_names = [
            name for name in self._channel_names if name in self._channel_properties
        ]
        if len(self._channel_names) == 0:
            raise CorruptFile("Could not find any topography data in this file.")

    @staticmethod
    def _normalize_unit(unit):
        """Normalize spelling of length units (e.g. GREEK SMALL LETTER MU)"""
        if unit is None or unit in length_units:
            return unit
        return mangle_length_unit_utf8(unit)

    def _process_channel(self, fobj, data, metadata):
        """
        Interpret the raw data and metadata of a single channel. Returns None
        if the channel does not contain topography (height) information.
        """
        data = np.array(data).T
        dim = 2
        if data.shape[0] == 1:
            dim = 1
            data = np.ravel(data)

        nb_grid_pts_x = metadata.get("nb_grid_pts_x")
        nb_grid_pts_y = metadata.get("nb_grid_pts_y")
        try:
            nx, ny = data.shape
        except ValueError:
            if nb_grid_pts_y is not None:
                raise CorruptFile(
                    "This file has just a single column and is hence a line "
                    f"scan, but the files metadata specifies {nb_grid_pts_y} "
                    "grid points in y-direction."
                )
            (nx,) = data.shape
            ny = None
        else:
            if nx == 2 or ny == 2:
                raise CorruptFile(
                    "This file has just two rows or two columns and is more "
                    "likely a line scan than a map."
                )
            if nb_grid_pts_y is not None and nb_grid_pts_y != ny:
                raise CorruptFile(
                    f"The number of rows (={ny}) of the topography from the "
                    f"file '{fobj}' does not match the number of grid points "
                    f"in y-direction in the file's metadata (={nb_grid_pts_y})."
                )
        if nb_grid_pts_x is not None and nb_grid_pts_x != nx:
            raise CorruptFile(
                f"The number of columns (={nx}) of the topography from the file "
                f"'{fobj}' does not match the number of grid points in "
                f"x-direction in the file's metadata (={nb_grid_pts_x})."
            )

        # Set grid points if not in metadata
        if nb_grid_pts_x is None:
            nb_grid_pts_x = nx
        if nb_grid_pts_y is None:
            nb_grid_pts_y = ny

        # Get physical sizes
        physical_size_x = metadata.get("physical_size_x")
        physical_size_y = metadata.get("physical_size_y")

        # Handle scale factors
        xfac = metadata.get("xfac")
        yfac = metadata.get("yfac")
        zfac = metadata.get("zfac")
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
        xunit = self._normalize_unit(metadata.get("xunit"))
        yunit = self._normalize_unit(metadata.get("yunit"))
        zunit = self._normalize_unit(metadata.get("zunit"))

        format_flavor = metadata.get("format_flavor")
        if format_flavor == "spip":
            # SPIP ASCII files store lateral sizes in nanometers
            if xunit is None:
                xunit = "nm"
            if yunit is None:
                yunit = "nm"
            # Without a z-unit, `Bit2nm` converts values to nanometers
            spip_bit2nm = metadata.get("spip_bit2nm")
            if zunit is None and spip_bit2nm is not None:
                zunit = "nm"
                if zfac is None:
                    zfac = spip_bit2nm

        # A single lateral unit applies to both directions
        if yunit is None and xunit is not None:
            yunit = xunit
        if xunit is None and zunit is not None:
            xunit = zunit
        if yunit is None and zunit is not None:
            yunit = zunit

        if format_flavor == "wyko":
            # Wyko files have a special scale factor
            wyko_pixel_size = metadata.get("wyko_pixel_size")
            wyko_aspect_ratio = metadata.get("wyko_aspect_ratio", 1)
            wyko_mult = metadata.get("wyko_mult")
            wyko_wavelength = metadata.get("wyko_wavelength")
            wyko_real_units = metadata.get("wyko_real_units", 0)
            if (
                wyko_mult is not None
                and wyko_wavelength is not None
                and not wyko_real_units
            ):
                # Data is given in units of the wavelength
                zfac = wyko_wavelength / wyko_mult

            if wyko_pixel_size is not None:
                physical_size_x = wyko_pixel_size * wyko_aspect_ratio * nb_grid_pts_x
                physical_size_y = wyko_pixel_size * nb_grid_pts_y

            # Wyko files have special units
            if xunit is None and yunit is None and zunit is None:
                xunit = yunit = "mm"
                zunit = "nm"
            else:
                raise CorruptFile(
                    "This is a Wyko file, but it appears to have unit metadata."
                )

        if zunit is not None and zunit not in length_units:
            # This is not height information (e.g. a phase or voltage channel)
            return None

        unit = zunit
        if unit is not None:
            for u in (xunit, yunit):
                if u is not None and u not in length_units:
                    raise CorruptFile(f"Unknown lateral unit '{u}'.")
            if xunit is not None:
                if physical_size_x is not None:
                    physical_size_x *= length_units[xunit] / length_units[unit]
            if yunit is not None:
                if physical_size_y is not None:
                    physical_size_y *= length_units[yunit] / length_units[unit]

        # Store processed metadata
        nb_grid_pts = None
        if nb_grid_pts_x is not None:
            if nb_grid_pts_y is not None:
                nb_grid_pts = (int(nb_grid_pts_x), int(nb_grid_pts_y))
            else:
                nb_grid_pts = (int(nb_grid_pts_x),)
        physical_sizes = None
        if physical_size_x is not None:
            if physical_size_y is not None:
                physical_sizes = (physical_size_x, physical_size_y)
            else:
                physical_sizes = (physical_size_x,)
        return dict(
            data=data,
            dim=dim,
            nb_grid_pts=nb_grid_pts,
            physical_sizes=physical_sizes,
            unit=unit,
            height_scale_factor=zfac,
            metadata=metadata,
        )

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                i,  # channel index
                name=name,
                dim=self._channel_properties[name]["dim"],
                nb_grid_pts=self._channel_properties[name]["nb_grid_pts"],
                physical_sizes=self._channel_properties[name]["physical_sizes"],
                uniform=True,
                unit=self._channel_properties[name]["unit"],
                height_scale_factor=self._channel_properties[name][
                    "height_scale_factor"
                ],
                info={
                    CHANNEL_NAME_INFO_KEY: name,
                    "raw_metadata": self._channel_properties[name]["metadata"],
                },
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

        if channel_index < 0 or channel_index >= len(self._channel_names):
            raise RuntimeError(
                f"There are only {len(self._channel_names)} channels, but channel "
                f"index is {channel_index}."
            )

        # handle channel name
        # we use the info dict here to transfer the channel name
        channel_name = self._channel_names[channel_index]
        properties = self._channel_properties[channel_name]
        file_unit = properties["unit"]
        file_height_scale_factor = properties["height_scale_factor"]

        physical_sizes = self._check_physical_sizes(
            physical_sizes, properties["physical_sizes"]
        )

        if height_scale_factor is not None and file_height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None and file_unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        _info = info.copy()
        _info["raw_metadata"] = properties["metadata"]
        _info[CHANNEL_NAME_INFO_KEY] = channel_name

        data = self._data[channel_name]
        if np.sum(np.isnan(data)) > 0:
            data = np.ma.masked_invalid(data)
        if properties["dim"] == 1:
            topography = UniformLineScan(
                data,
                physical_sizes,
                unit=unit or file_unit,
                info=_info,
                periodic=periodic,
            )
        else:
            topography = Topography(
                data,
                physical_sizes,
                unit=unit or file_unit,
                info=_info,
                periodic=periodic,
            )
        if height_scale_factor is not None or file_height_scale_factor is not None:
            topography = topography.scale(
                height_scale_factor or file_height_scale_factor
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
