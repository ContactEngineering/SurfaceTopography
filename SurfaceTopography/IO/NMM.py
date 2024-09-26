#
# Copyright 2024 Lars Pastewka
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
import io
import os
import zipfile

import dateutil.parser
import numpy as np
import pandas as pd

from ..Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from ..UniformLineScanAndTopography import Topography
from .Reader import ChannelInfo, ReaderBase

#
# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/nmmxyz.c
#


class NMMReader(ReaderBase):
    _format = "nmm"
    _mime_types = ["application/zip"]
    _file_extensions = ["zip"]

    _name = "Nanomeasuring machine NMM"
    _description = """
Import filter for nanomeasuring machine files. The reader opens ZIP files that contain
a single DAT and a single DSC file. The data is interpreted as a line scan. The reader
rejects ZIP files with more that two files or files with extensions other than DAT and
DSC.
    """  # noqa: E501

    _UNITS = {
        "Lx": "m",
        "Ly": "m",
        "Lz": "m",
        "Ax": "V",
        "Az": "m",
        "Az0": "m",
        "Az1": "m",
        "-Lz+Az": "m",
        "XY vector": "m",
    }

    @staticmethod
    def _assert_line(fobj, line):
        L = fobj.readline().strip()
        if L != line:
            raise CorruptFile(
                f"Expected line '{line}' in DSC file, but got '{L}' instead."
            )

    def _read_scan_procedure(self, dsc_file):
        self._assert_line(dsc_file, "------------------------------------------")
        self._assert_line(dsc_file, "Scan procedure description file")
        metadata = {}
        line = dsc_file.readline().strip()
        while line != "------------------------------------------":
            key, value = line.strip().split(":", maxsplit=1)
            metadata[key.strip()] = value.strip()
            line = dsc_file.readline().strip()
        line = dsc_file.readline()
        while line:
            line = line.strip()
            if len(line) > 0 and line != "------------------------------------------":
                nb, section_name = line.split(" ", maxsplit=1)
                d = {}
                line = dsc_file.readline()
                while (
                    line
                    and line.strip() != "------------------------------------------"
                ):
                    line = line.strip()
                    if len(line) > 0:
                        key, value = line.split(":", maxsplit=1)
                        d[key.strip()] = value.strip()
                    line = dsc_file.readline()
                metadata[section_name.strip()] = d
            else:
                line = dsc_file.readline()
        return metadata

    @staticmethod
    def _parse_value_and_unit(s):
        value, unit = s.split(" ", maxsplit=1)
        return float(value), unit.strip("[").strip("]")

    def __init__(self, fobj, rtol=1e-6):
        """
        Initialize the NMMReader object.

        Parameters
        ----------
        fobj : file-like object or callable
            The file object or a callable that returns a file object to the ZIP file
            containing the NMM data.
        rtol : float, optional
            Relative tolerance for detecting uniform grids. (Default: 1e-6).

        Raises
        ------
        FileFormatMismatch
            If the ZIP file does not contain exactly two files or if the files are not
            a DSC and a DAT file.
        CorruptFile
            If the number of columns in the DAT file does not match the number of
            entries in the DSC file or if the number of data points in the DSC file does
            not match the number of rows in the DAT file.
        """
        self._fobj = fobj
        self._rtol = rtol
        if callable(fobj):
            fobj = fobj()
        with zipfile.ZipFile(fobj, "r") as z:
            filenames = [
                fn
                for fn in z.namelist()
                if not (
                    os.path.split(fn)[1].startswith(".")
                    or os.path.split(fn)[1].startswith("_")
                )
            ]

            self._prefix = None
            for fn in filenames:
                try:
                    p, _ = fn.split(".", maxsplit=1)
                    if self._prefix is None or len(p) < len(self._prefix):
                        self._prefix = p
                except ValueError:
                    pass

            if self._prefix is None:
                raise CorruptFile("Could not identify file prefix.")

            self._metadata = self._read_scan_procedure(
                io.TextIOWrapper(z.open(f"{self._prefix}.dsc", "r"), encoding="utf-8")
            )

            self._nb_scans = int(self._metadata["Scan field"]["Number of scans"])
            self._nb_lines = int(self._metadata["Scan field"]["Number of lines"])

            physical_size_x, xunit1 = self._parse_value_and_unit(
                self._metadata["Scan field"]["Scan line length"]
            )
            physical_size_y, yunit1 = self._parse_value_and_unit(
                self._metadata["Scan field"]["Scan field width"]
            )
            grid_spacing_x, xunit2 = self._parse_value_and_unit(
                self._metadata["Scan field"]["Distance beetween points"]
            )
            grid_spacing_y, yunit2 = self._parse_value_and_unit(
                self._metadata["Scan field"]["Distance beetween lines"]
            )

            self._unit = xunit1

            assert xunit2 == self._unit
            assert yunit1 == self._unit
            assert yunit2 == self._unit

            self._unit = mangle_length_unit_utf8(self._unit)

            self._height_scale_factor = get_unit_conversion_factor("m", self._unit)

            # The NMM files reports a scan field, number of pixels and grid spacing.
            # The scan field is actually (nb_pixels - 1) * grid_spacing, which makes
            # sense if we interpret the points as node positions as in the nonuniform
            # line scans. For the topographic maps, we interpret nodes as pixel centers.
            nb_grid_pts_x = int(physical_size_x / grid_spacing_x)
            assert abs(nb_grid_pts_x * grid_spacing_x / physical_size_x - 1) < rtol
            nb_grid_pts_y = self._nb_lines
            assert (
                abs((nb_grid_pts_y - 1) * grid_spacing_y / physical_size_y - 1) < rtol
            )

            self._physical_sizes = (physical_size_x, physical_size_y)
            self._nb_grid_pts = (nb_grid_pts_x + 1, nb_grid_pts_y)

            self._info = {
                "acquisition_time": dateutil.parser.parse(
                    self._metadata["Creation time"]
                )
            }

    def _read(self, dsc_file, dat_file):
        # Read data file (DAT)
        dat = pd.read_csv(dat_file, sep=r"\s+", header=None)

        # Read index (DSC) file describing the individual data (DAT) file
        dsc = pd.read_csv(
            dsc_file,
            sep=" : ",
            skiprows=1,
            names=["index", "datetime", "name", "nb_grid_pts", "description"],
        )
        dsc = dsc[np.isfinite(dsc["nb_grid_pts"])]

        if len(dsc) != len(dat.columns):
            raise CorruptFile(
                f"Number of columns reported in DSC file (= {len(dsc)} "
                "does not match the number of columns in the DAT file "
                f"(= {len(dat.columns)})"
            )

        if not np.all(len(dat) == dsc["nb_grid_pts"]):
            raise CorruptFile(
                f"Number of data points reported in DSC file (= "
                f"{dsc['nb_grid_pts']} does not match the number of rows in "
                f"the DAT file (= {len(dat)})"
            )

        dat.columns = dsc["name"]

        # Coordinates are stored in the file
        # x = dat["Lx"].values
        # y = dat["Ly"].values
        # r = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
        h = dat["-Lz+Az"].values

        # Check that the grid information is correct
        if len(h) != self._nb_grid_pts[0]:
            raise CorruptFile(
                f"Expected {self._nb_grid_pts[0]} data points, got {len(h)}."
            )

        return h

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                0,  # Channel index
                name=f"Scan {i+1}",  # There is only a single channel
                dim=2,
                nb_grid_pts=self._nb_grid_pts,
                physical_sizes=self._physical_sizes,
                uniform=True,
                unit=self._unit,
                # Height is in meters
                height_scale_factor=self._height_scale_factor,
                info=self._info,
            )
            for i in range(self._nb_scans)
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

        if channel_index >= self._nb_scans:
            raise RuntimeError(
                f"Found {self._nb_scans} scans in this file, but channel index is "
                f"{channel_index}.)"
            )

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        _info = self._info.copy()
        _info.update(info)

        heights = np.empty(self._nb_grid_pts)
        fobj = self._fobj
        if callable(fobj):
            fobj = fobj()
        with zipfile.ZipFile(fobj, "r") as z:
            for i in range(self._nb_lines):
                heights[:, i] = self._read(
                    z.open(f"{self._prefix}_{channel_index+1}_{i+1}.dsc", "r"),
                    z.open(f"{self._prefix}_{channel_index+1}_{i+1}.dat", "r"),
                )

        t = Topography(
            heights,
            self._physical_sizes,
            unit=self._unit,
            info=_info,
            periodic=periodic,
        )
        return t.scale(self._height_scale_factor)


def read_nmm(dsc_file, dat_file, rtol=1e-6):
    """
    Convenience function for reading a Nanomeasuring Machine (NMM) profile.

    Parameters
    ----------
    dsc_file : str or file-like object
        Path to the DSC file or file-like object.
    dat_file : str or file-like object
        Path to the DAT file or file-like object.
    rtol : float, optional
        Relative tolerance for detecting uniform grids. (Default: 1e-6).
    """
    return NMMReader(dsc_file, dat_file, rtol=rtol).topography()
