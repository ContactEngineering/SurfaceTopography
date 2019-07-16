#
# Copyright 2019 Antoine Sanner
#           2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

- we have to use `MPI.File.Open` instead of `open` to allow several processors to access the same file simultaneously
- make the file reading in 3 steps:
    - read the nb_grid_pts only (Reader.__init__)
    - make the domain decomposition according to the nb_grid_pts
    - load the relevant subdomain on each processor in Reader.topography()

"""





from numpy.lib.format import read_magic, _read_array_header, _check_version
import numpy as np

from PyCo.Topography import Topography
from PyCo.Topography.IO.Reader import ReaderBase, FileFormatMismatch

from NuMPI import MPI
import NuMPI
import NuMPI.IO

class NPYReader(ReaderBase):
    """
    npy is a fileformat made specially for numpy arrays. They contain no extra
    metadata so we use directly the implementation from numpy and NuMPI
    """

    def __init__(self, fn, communicator=MPI.COMM_WORLD):
        """

        Parameters
        ----------
        fn: filename
        comm: MPI communicator
        """
        super().__init__()

        # if comm is None:
        #    raise ValueError("you should provide comm when running with MPI")
        try:
            self.mpi_file = NuMPI.IO.make_mpi_file_view(fn, communicator, format="npy")
            self.dtype = self.mpi_file.dtype
            self._nb_grid_pts = self.mpi_file.nb_grid_pts
        except NuMPI.IO.MPIFileTypeError:
            raise FileFormatMismatch()

        # TODO: maybe implement extras specific to Topography , like loading the units and the physical_sizes

    def topography(self, substrate=None, physical_sizes=None, channel=None, info={}):
        """
        Returns the `Topography` object containing the data attributed to the
        processors. `substrate` prescribes the domain decomposition.


        Parameters
        ----------
        substrate: Free- or PeriodicFFTElasticHalfspace instance
        has attributes topography_subdomain_locations, topography_nb_subdomain_grid_pts and nb_grid_pts
        size: (float, float)
        physical_sizes of the topography
        channel: int or None
        ignored here because only one channel is availble here
        info: dict
        updates for the info dictionary

        Returns
        -------
        Topography
        """
        info = self._process_info(info)

        # TODO: Are sometimes the units Stored?
        if self.mpi_file.comm.size > 1:
            if (substrate is None):
                raise ValueError("you should provide substrate to specify the domain decomposition")
            subdomain_locations=substrate.topography_subdomain_locations
            nb_subdomain_grid_pts = substrate.topography_nb_subdomain_grid_pts
        else:
            subdomain_locations=(0,0)
            nb_subdomain_grid_pts=self.nb_grid_pts
        if substrate is not None and hasattr(substrate, "physical_sizes"):
            if physical_sizes is not None:
                raise ValueError("physical_sizes is already provided by substrate")
            physical_sizes=substrate.physical_sizes

        return Topography(
            heights=self.mpi_file.read(subdomain_locations=subdomain_locations,
                                       nb_subdomain_grid_pts=nb_subdomain_grid_pts),
            subdomain_locations=subdomain_locations,
            nb_grid_pts=self.nb_grid_pts,
            communicator=self.mpi_file.comm,
            physical_sizes=physical_sizes,
            info=info)

def save_npy(fn, topography):
    NuMPI.IO.save_npy(fn=fn, data=topography.heights(), subdomain_locations=topography.subdomain_locations,
                      nb_grid_pts=topography.nb_subdomain_grid_pts, comm=topography.communicator)
