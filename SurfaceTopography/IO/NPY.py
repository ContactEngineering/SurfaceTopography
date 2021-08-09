#
# Copyright 2019-2021 Lars Pastewka
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

from ..UniformLineScanAndTopography import Topography
from .Reader import ReaderBase, FileFormatMismatch, ChannelInfo

from NuMPI import MPI
import NuMPI
import NuMPI.IO


class NPYReader(ReaderBase):
    """
    NPY is a file format made specially for numpy arrays. They contain no extra
    metadata so we use directly the implementation from numpy and NuMPI.

    For a description of the file format, see here:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
    """

    _format = 'npy'
    _name = 'Numpy arrays (NPY)'
    _description = '''
Load topography information stored as a numpy array. The numpy array format is
specified
[here](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).
The reader expects a two-dimensional array and interprets it as a map of
heights. Numpy arrays do not store units or physical sizes. These need to be
manually provided by the user.
    '''

    def __init__(self, fn, communicator=MPI.COMM_WORLD):
        """
        Open file in the NPY format.

        Parameters
        ----------
        fn : str
            Name of the file
        communicator : mpi4py MPI communicator or NuMPI stub communicator
            MPI communicator object for parallel loads.
        """
        super().__init__()

        try:
            self.mpi_file = NuMPI.IO.make_mpi_file_view(fn, communicator,
                                                        format="npy")
            self.dtype = self.mpi_file.dtype
            self._nb_grid_pts = self.mpi_file.nb_grid_pts
        except NuMPI.IO.MPIFileTypeError:
            raise FileFormatMismatch()

        # TODO: maybe implement extras specific to SurfaceTopography, like
        #  loading the units and the physical_sizes

    @property
    def channels(self):
        return [ChannelInfo(self, 0,
                            name='Default',
                            dim=len(self._nb_grid_pts),
                            nb_grid_pts=self._nb_grid_pts)]

    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None, nb_subdomain_grid_pts=None):

        if channel_index is not None and channel_index != 0:
            raise ValueError('`channel_index` must be None or 0.')

        physical_sizes = self._check_physical_sizes(physical_sizes)
        if subdomain_locations is None and nb_subdomain_grid_pts is None:
            if self.mpi_file.comm.size > 1:
                raise ValueError("This is a parallel run, you should provide "
                                 "subdomain location and number of grid "
                                 "points")
            topography = Topography(
                heights=self.mpi_file.read(
                    subdomain_locations=subdomain_locations,
                    nb_subdomain_grid_pts=nb_subdomain_grid_pts),
                physical_sizes=physical_sizes,
                periodic=periodic,
                unit=unit,
                info=info
            )
        else:
            topography = Topography(
                heights=self.mpi_file.read(
                    subdomain_locations=subdomain_locations,
                    nb_subdomain_grid_pts=nb_subdomain_grid_pts),
                decomposition="subdomain",
                subdomain_locations=subdomain_locations,
                nb_grid_pts=self._nb_grid_pts,
                communicator=self.mpi_file.comm,
                physical_sizes=physical_sizes,
                periodic=periodic,
                unit=unit,
                info=info)

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def save_npy(fn, topography):
    NuMPI.IO.save_npy(fn=fn, data=topography.heights(),
                      subdomain_locations=topography.subdomain_locations,
                      nb_grid_pts=topography.nb_subdomain_grid_pts,
                      comm=topography.communicator)
