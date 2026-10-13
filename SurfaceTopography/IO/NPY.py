#
# Copyright 2019-2021, 2023-2024 Lars Pastewka
#           2020-2021 Michael Röttger
#           2019 Antoine Sanner
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
In MPI Parallelized programs:

- we have to use `MPI.File.Open` instead of `open` to allow several processors
  to access the same file simultaneously
- make the file reading in 3 steps:
    - read the nb_grid_pts only (Reader.__init__)
    - make the domain decomposition according to the nb_grid_pts
    - load the relevant subdomain on each processor in Reader.topography()
"""

import zipfile

import numpy as np
import NuMPI
import NuMPI.IO
from NuMPI import MPI

from ..Exceptions import FileFormatMismatch, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase


def _check_npy_array(shape, dtype):
    """
    Check that an array stored in an NPY file can be interpreted as a
    topography (two-dimensional) or line scan (one-dimensional) with real
    numbers as heights.
    """
    if len(shape) not in (1, 2):
        raise UnsupportedFormatFeature(
            f"NPY file contains a {len(shape)}-dimensional array; only one- "
            "(line scan) and two-dimensional (topography) arrays are supported."
        )
    if any(n < 1 for n in shape):
        raise UnsupportedFormatFeature("NPY file contains an empty array.")
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)) \
            or np.issubdtype(dtype, np.complexfloating):
        raise UnsupportedFormatFeature(
            f"NPY file contains data of type '{dtype}' which cannot be "
            "interpreted as heights."
        )


class NPYReader(ReaderBase):
    """
    NPY is a file format made specially for numpy arrays. They contain no extra
    metadata so we use directly the implementation from numpy and NuMPI.

    For a description of the file format, see here:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
    """

    _format = "npy"
    _mime_types = ["application/x-npy"]
    _file_extensions = ["npy"]

    _name = "NumPy array (NPY)"
    _description = """
Load topography information stored as a numpy array. The numpy array format is
specified
[here](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).
The reader expects a two-dimensional array and interprets it as a map of
heights; the first index of the array runs along x, the second along y.
One-dimensional arrays are interpreted as (uniform) line scans. Numpy arrays
do not store units or physical sizes. These need to be manually provided by the
user.
    """

    _MAGIC = b'\x93NUMPY'

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC):
            return MagicMatch.YES
        return MagicMatch.NO

    def __init__(self, fobj, communicator=MPI.COMM_WORLD):
        """
        Open file in the NPY format.

        Parameters
        ----------
        fobj : str
            Name of the file
        communicator : mpi4py MPI communicator or NuMPI stub communicator
            MPI communicator object for parallel loads.
        """
        super().__init__()

        if callable(fobj):
            fobj = fobj()
        try:
            self.mpi_file = NuMPI.IO.mpi_open(fobj, communicator, format="npy")
            self.dtype = self.mpi_file.dtype
            self._nb_grid_pts = tuple(int(n) for n in self.mpi_file.array_shape)
        except NuMPI.IO.MPIFileTypeError:
            raise FileFormatMismatch()
        _check_npy_array(self._nb_grid_pts, self.dtype)

        # TODO: maybe implement extras specific to SurfaceTopography, like
        #  loading the units and the physical_sizes

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=len(self._nb_grid_pts),
                uniform=True,
                nb_grid_pts=self._nb_grid_pts,
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

        if channel_index is not None and channel_index != 0:
            raise ValueError("`channel_index` must be None or 0.")

        physical_sizes = self._check_physical_sizes(physical_sizes)
        if len(self._nb_grid_pts) == 1:
            # This is a line scan
            if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
                raise ValueError(
                    "Parallel reading only works for topographies, not line scans."
                )
            topography = UniformLineScan(
                self.mpi_file.read(),
                physical_sizes,
                periodic=periodic,
                unit=unit,
                info=info,
            )
        elif subdomain_locations is None and nb_subdomain_grid_pts is None:
            if self.mpi_file.comm.size > 1:
                raise ValueError(
                    "This is a parallel run, you should provide "
                    "subdomain location and number of grid "
                    "points"
                )
            topography = Topography(
                heights=self.mpi_file.read(
                    subdomain_locations=subdomain_locations,
                    nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                ),
                physical_sizes=physical_sizes,
                periodic=periodic,
                unit=unit,
                info=info,
            )
        else:
            topography = Topography(
                heights=self.mpi_file.read(
                    subdomain_locations=subdomain_locations,
                    nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                ),
                decomposition="subdomain",
                subdomain_locations=subdomain_locations,
                nb_grid_pts=self._nb_grid_pts,
                communicator=self.mpi_file.comm,
                physical_sizes=physical_sizes,
                periodic=periodic,
                unit=unit,
                info=info,
            )

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


class NPZReader(ReaderBase):
    """
    NPZ is a ZIP archive of NPY files, as written by `numpy.savez` and
    `numpy.savez_compressed`. Each two-dimensional array is presented as a
    separate channel, named after the array. (Gwyddion's `npyfile.c` was used
    as a reference for which arrays are considered.)
    """

    _format = "npz"
    _mime_types = ["application/x-npz"]
    _file_extensions = ["npz"]

    _name = "NumPy array archive (NPZ)"
    _description = """
Load topography information stored as multiple numpy arrays in a single NPZ
archive (as written by `numpy.savez` or `numpy.savez_compressed`). Each
two-dimensional array in the archive is interpreted as a separate map of
heights (channel); the first index of the array runs along x, the second along
y. Numpy arrays do not store units or physical sizes. These need to be manually
provided by the user.
    """

    _MAGIC = b"PK\x03\x04"

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC):
            # Generic ZIP container; other readers use ZIP containers as well
            return MagicMatch.MAYBE
        return MagicMatch.NO

    def __init__(self, fobj):
        self._fobj = fobj
        self._channels = []
        self._array_names = []
        with OpenFromAny(fobj, "rb") as f:
            try:
                archive = zipfile.ZipFile(f)
            except zipfile.BadZipFile:
                raise FileFormatMismatch("This is not a ZIP archive.")
            with archive:
                for member in archive.namelist():
                    if not member.endswith(".npy"):
                        continue
                    with archive.open(member) as npy:
                        try:
                            version = np.lib.format.read_magic(npy)
                            if version == (1, 0):
                                read_header = np.lib.format.read_array_header_1_0
                            else:
                                read_header = np.lib.format.read_array_header_2_0
                            shape, _, dtype = read_header(npy)
                            _check_npy_array(shape, dtype)
                        except (ValueError, UnsupportedFormatFeature):
                            continue
                    if len(shape) != 2:
                        continue
                    self._array_names += [member]
                    self._channels += [
                        ChannelInfo(
                            self,
                            len(self._channels),
                            name=member[: -len(".npy")],
                            dim=2,
                            nb_grid_pts=tuple(int(n) for n in shape),
                            uniform=True,
                        )
                    ]
        if len(self._channels) == 0:
            raise FileFormatMismatch(
                "ZIP archive does not contain any two-dimensional NPY arrays."
            )

    @property
    def channels(self):
        return self._channels

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
        member = self._array_names[channel_index]

        physical_sizes = self._check_physical_sizes(physical_sizes)
        with OpenFromAny(self._fobj, "rb") as f:
            with zipfile.ZipFile(f) as archive:
                with archive.open(member) as npy:
                    heights = np.lib.format.read_array(npy, allow_pickle=False)

        topography = Topography(
            heights, physical_sizes, periodic=periodic, unit=unit, info=info
        )
        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)
        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def save_npy(fn, topography):
    NuMPI.IO.save_npy(
        fn=fn,
        data=topography.heights(),
        subdomain_locations=topography.subdomain_locations,
        nb_grid_pts=topography.nb_subdomain_grid_pts,
        comm=topography.communicator,
    )
