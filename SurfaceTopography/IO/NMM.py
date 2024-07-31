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
import os.path
import zipfile

import numpy as np
import pandas as pd

from ..Exceptions import (CorruptFile, FileFormatMismatch,
                          MetadataAlreadyFixedByFile)
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import UniformLineScan
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

    def __init__(self, fobj, rtol=1e-6):
        """
        Initialize the NMMReader object.

        Parameters
        ----------
        fobj : file-like object or callable
            The file object or a callable that returns a file object to read the ZIP
            file.
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
        if callable(fobj):
            fobj = fobj()
        with zipfile.ZipFile(fobj, "r") as z:
            filenames = [
                fn
                for fn in z.namelist()
                if not (fn.startswith(".") or fn.startswith("_"))
            ]
            if len(filenames) != 2:
                raise FileFormatMismatch(
                    "Expecting exactly two files in the ZIP container."
                )

            fileexts = [os.path.splitext(filename)[1].lower() for filename in filenames]
            if set(fileexts) != {".dsc", ".dat"}:
                raise FileFormatMismatch(
                    "Expecting a DSC and a DAT file in the ZIP container."
                )

            self._dsc_file = filenames[fileexts.index(".dsc")]
            self._dat_file = filenames[fileexts.index(".dat")]

            # Read data file (DAT)
            with z.open(self._dat_file, "r") as dat_file:
                # Read data file
                self._dat = pd.read_csv(dat_file, sep=r"\s+", header=None)

            # Read index (DSC) file describing the individual data (DAT) file
            with z.open(self._dsc_file, "r") as dsc_file:
                self._dsc = pd.read_csv(
                    dsc_file,
                    sep=" : ",
                    skiprows=1,
                    names=["index", "datetime", "name", "nb_grid_pts", "description"],
                )
            self._dsc = self._dsc[np.isfinite(self._dsc["nb_grid_pts"])]

            if len(self._dsc) != len(self._dat.columns):
                raise CorruptFile(
                    f"Number of columns reported in DSC file (= {len(self._dsc)} "
                    "does not match the number of columns in the DAT file "
                    f"(= {len(self._dat.columns)})"
                )

            if not np.all(len(self._dat) == self._dsc["nb_grid_pts"]):
                raise CorruptFile(
                    f"Number of data points reported in DSC file (= "
                    f"{self._dsc['nb_grid_pts']} does not match the number of rows in "
                    f"the DAT file (= {len(self._dat)})"
                )

            self._dat.columns = self._dsc["name"]
            self._info = {"acquisition_data": self._dsc["datetime"].values[0]}

            self._x = self._dat["Lx"].values
            self._h = self._dat["-Lz+Az"].values
            self._physical_size = self._x[-1] - self._x[0]
            if self._physical_size < 0:
                self._x = self._x[::-1]
                self._h = self._h[::-1]
                self._physical_size = self._x[-1] - self._x[0]

            if np.max(np.abs(np.diff(self._x) / (self._x[1] - self._x[0]))) < 1 + rtol:
                # This is a uniform grid
                self._uniform = True
            else:
                self._uniform = False

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                0,  # channel index
                name="Default",  # There is only a single channel
                dim=1,
                nb_grid_pts=len(self._dat),
                physical_sizes=self._physical_size,
                uniform=self._uniform,
                unit="m",  # Everything is in meters
                height_scale_factor=1,  # Data is in natural heights
                info=self._info,
            )
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

        if channel_index != 0:
            raise RuntimeError(
                "Channel index must be zero. (DATX files only have a single height "
                "channel.)"
            )

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if periodic and not self._uniform:
            raise ValueError(
                "Nonuniform line scans cannot be periodic (while reading NMM file)"
            )

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        _info = self._info.copy()
        _info.update(info)

        if self._uniform:
            t = UniformLineScan(
                self._h,
                self._physical_size,
                unit="m",
                info=_info,
                periodic=periodic,
            )
        else:
            t = NonuniformLineScan(self._x, self._h, unit="m", info=_info)

        return t.scale(1)
