#
# Copyright 2019 Lars Pastewka
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

import numpy as np

from .. import Topography
from ..HeightContainer import UniformTopographyInterface

from .Reader import ReaderBase


class NCReader(ReaderBase):
    def __init__(self, fobj):
        from netCDF4 import Dataset
        self._nc = Dataset(fobj, 'r')
        self._x_var = self._nc.variables['x']
        self._y_var = self._nc.variables['y']
        self._heights_var = self._nc.variables['heights']
        self.physical_sizes = (self._x_var.length, self._y_var.length)

    def topography(self, size=None, info={}):
        size = self._process_size(size)
        return Topography(self._heights_var[...], size, info=self._process_info(info))


def write_nc(topography, filename, format='NETCDF3_64BIT_DATA'):
    """
    Write topography into a NetCDF file.

    Parameters
    ----------
    topography : :obj:`Topography`
        The topography to write to disk.
    filename : str
        Name of the NetCDF file
    format : str
        NetCDF file format. Default is 'NETCDF3_64BIT_DATA'.
    """
    from netCDF4 import Dataset
    nc = Dataset(filename, 'w', format=format)
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes
    nc.createDimension('x', nx)
    nc.createDimension('y', ny)
    x_var = nc.createVariable('x', 'f8', ('x',))
    x_var.length = sx
    x_var.length_unit = topography.info['unit']
    x_var[...] = (np.arange(nx) + 0.5) * sx / nx
    y_var = nc.createVariable('y', 'f8', ('y',))
    y_var.length = sy
    y_var.length_unit = topography.info['unit']
    y_var[...] = (np.arange(ny) + 0.5) * sy / ny
    heights_var = nc.createVariable('heights', 'f8', ('x', 'y',))
    heights_var[...] = topography.heights()
    nc.close()


### Register analysis functions from this module

UniformTopographyInterface.register_function('to_netcdf', write_nc)
