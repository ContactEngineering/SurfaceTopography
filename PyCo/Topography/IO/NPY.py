#
# Copyright 2019 Antoine Sanner
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
    - read the resolution only (Reader.__init__)
    - make the domain decomposition according to the resolution
    - load the relevant subdomain on each processor in Reader.topography()

"""


try: #TODO: Code should look like the same with and without mpi4py
    from mpi4py import MPI
    _with_mpi = True
except:
    _with_mpi = False
if _with_mpi:
    import NuMPI.IO #TODO: NuMPI should provide the same interface with and without mpi4py

from PyCo.Topography import Topography
from PyCo.Topography.IO.Reader import ReaderBase, FileFormatMismatch

import abc
from numpy.lib.format import read_magic, _read_array_header, _check_version
import numpy as np
import warnings

class NPYReader(ReaderBase):
    """
    npy is a fileformat made specially for numpy arrays. They contain no extra
    metadata so we use straightforwardly the implementation from numpy and MPITools
    """
    def __init__(self, fn, comm=None):
        """

        Parameters
        ----------
        fn: filename
        comm: MPI communicator
        """

        if comm is not None: # TODO: not ok code should look the same for MPI and nonmpi: have to write stub for MPI.File
            #if comm is None:
            #    raise ValueError("you should provide comm when running with MPI")
            self._with_mpi = True
            try:
                self.mpi_file = NuMPI.IO.make_mpi_file_view(fn, comm, format="npy")
                self.dtype = self.mpi_file.dtype
                self._resolution = self.mpi_file.resolution
            except NuMPI.IO.MPIFileTypeError:
                raise FileFormatMismatch()
        else: # just use the functions from numpy
            self._with_mpi = False
            self.file=open(fn, "rb")
            try:
                version = read_magic(self.file)
                _check_version(version)
                self._resolution, fortran_order, self.dtype = _read_array_header(self.file, version)
            except ValueError:
                raise FileFormatMismatch()
        super().__init__()
        # TODO: maybe implement extras specific to Topography , like loading the units and the size

    def topography(self, substrate=None, size = None, channel=None, info={}):
        """
        Returns the `Topography` object containing the data attributed to the
        processors. `substrate` prescribes the domain decomposition.


        Parameters
        ----------
        substrate: Free- or PeriodicFFTElasticHalfspace instance
        has attributes topography_subdomain_location, topography_subdomain_resolution and resolution
        size: (float, float)
        size of the topography
        channel: int or None
        ignored here because only one channel is availble here
        info: dict
        updates for the info dictionary

        Returns
        -------
        Topography
        """
        info = self._process_info(info)
        # TODO: Are sometimes the Units Stored?
        if self._with_mpi:
            if ( substrate is None ):
                raise ValueError("you should provide substrate to specify the domain decomposition")
            if size is not None:
                raise ValueError("size is already provided by substrate")

            return Topography(
                heights=self.mpi_file.read(subdomain_location=substrate.topography_subdomain_location,
                                           subdomain_resolution=substrate.topography_subdomain_resolution),
                subdomain_location=substrate.topography_subdomain_location,
                resolution=substrate.resolution,
                pnp=substrate.pnp,
                size=substrate.size,
                info =info)

        else:
            size = self._process_size(size)
            array = np.fromfile(self.file, dtype=self.dtype,
                    count=np.multiply.reduce(self.resolution, dtype=np.int64))
            array.shape = self.resolution
            self.file.close() # TODO: Or make this in the destructor ?
            return Topography(heights=array, size=size, info=info)

def save_npy(fn, topography):
    NuMPI.IO.save_npy(fn=fn, data=topography.heights(), subdomain_location=topography.subdomain_location,
                       resolution=topography.subdomain_resolution, comm=topography.comm)
