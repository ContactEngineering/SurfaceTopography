#
# Copyright 2019-2020 Lars Pastewka
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

import numpy as np

from .. import Topography
from ..HeightContainer import UniformTopographyInterface

from .Reader import ReaderBase, ChannelInfo


class NCReader(ReaderBase):
    _format = 'nc'
    _name = 'Network Common Data Format (NetCDF)'

    _description = '''
This reader reads topography data contained in a
[NetCDF](https://www.unidata.ucar.edu/software/netcdf/) container. The
reader looks for a variable named `heights` containing a two-dimensional
array that is interpreted as height information. The respective dimensions are
named `x` and `y`.

The reader additionally looks for two (optional) variables `x` and `y` that
contain the x- and y-coordinates of the first and second index of the height
arrays. The attribute `length` of `x` and `y` must contain the physical size
in the respective direction. The optional attribute `length_unit` of these
variables describes the physical unit. The optional additional attribute
`periodic` indicates whether the direction contains periodic data. If
`periodic` is missing, the reader interprets the data as non-periodic.

An example file layout (output of `ncdump -h`) containing a topography map
with 128 x 128 pixels looks like this:
```
netcdf test_nc_file {
dimensions:
    x = 128 ;
    y = 128 ;
variables:
    double x(x) ;
        x:length = 3LL ;
        x:periodic = 1LL ;
        x:length_unit = "μm" ;
    double y(y) ;
        y:length = 3LL ;
        y:periodic = 1LL ;
        y:length_unit = "μm" ;
    double heights(x, y) ;
}
```
The following code snippets reads the file and displays the topography data as
a two-dimensional color map in Python:
```python
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

with Dataset('parallel_save_test.nc') as nc:
    heights = np.array(nc.variables['heights'])
    length_x = nc.variables['x'].length
    length_y = nc.variables['y'].length
    unit = nc.variables['x'].length_unit

plt.figure()
plt.subplot(aspect=1)

nx, ny = heights.shape
x = (np.arange(nx)+0.5)*length_x/nx
y = (np.arange(ny)+0.5)*length_y/ny
plt.pcolormesh(x, y, heights)

plt.show()
```
'''

    def __init__(self, fobj, communicator=None):
        self._nc = None
        from netCDF4 import Dataset
        if communicator is not None and communicator.Get_size() > 1:
            self._nc = Dataset(fobj, 'r', parallel=True, comm=communicator)
        else:
            self._nc = Dataset(fobj, 'r')
        self._communicator = communicator
        self._x_dim = self._nc.dimensions['x']
        self._y_dim = self._nc.dimensions['y']
        self._x_var = self._nc.variables[
            'x'] if 'x' in self._nc.variables else None
        self._y_var = self._nc.variables[
            'y'] if 'y' in self._nc.variables else None
        self._heights_var = self._nc.variables['heights']

        # The following information may be missing from the NetCDF file
        self._physical_sizes = None
        self._periodic = False
        self._info = {}
        if self._x_var is not None:
            if self._y_var is not None:
                self._physical_sizes = (self._x_var.length, self._y_var.length)
            try:
                self._periodic = self._x_var.periodic != 0
            except AttributeError:
                pass
            try:
                self._info['unit'] = self._x_var.length_unit
            except AttributeError:
                pass

    def __del__(self):
        self.close()

    def close(self):
        if self._nc is not None:
            self._nc.close()
            self._nc = None

    @property
    def channels(self):
        return [ChannelInfo(self, 0,
                            name='Default',
                            dim=2,
                            nb_grid_pts=(len(self._x_dim), len(self._y_dim)),
                            physical_sizes=self._physical_sizes,
                            periodic=self._periodic,
                            info=self._info)]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, info={},
                   periodic=None,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        physical_sizes = self._check_physical_sizes(physical_sizes,
                                                    self._physical_sizes)
        _info = self._info.copy()
        _info.update(info)
        if subdomain_locations is None and nb_subdomain_grid_pts is None:
            return Topography(self._heights_var, physical_sizes,
                              periodic=self._periodic
                              if periodic is None else periodic,
                              info=_info)
        else:
            return Topography(self._heights_var, physical_sizes,
                              periodic=self._periodic
                              if periodic is None else periodic,
                              decomposition='domain',
                              subdomain_locations=subdomain_locations,
                              nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                              communicator=self.communicator,
                              info=_info)

    @property
    def communicator(self):
        return self._communicator

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def write_nc(topography, filename, format='NETCDF3_64BIT_OFFSET'):
    """
    Write topography into a NetCDF file.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        The topography to write to disk.
    filename : str
        Name of the NetCDF file
    format : str
        NetCDF file format. Default is 'NETCDF3_64BIT_OFFSET'.
    """
    from netCDF4 import Dataset
    if not topography.is_domain_decomposed and \
            topography.communicator.rank > 1:
        return
    with Dataset(filename, 'w', format=format,
                 parallel=topography.is_domain_decomposed,
                 comm=topography.communicator) as nc:
        nx, ny = topography.nb_grid_pts
        sx, sy = topography.physical_sizes

        nc.createDimension('x', nx)
        nc.createDimension('y', ny)

        x_var = nc.createVariable('x', 'f8', ('x',))
        y_var = nc.createVariable('y', 'f8', ('y',))
        heights_var = nc.createVariable('heights', 'f8', ('x', 'y',))

        x_var.length = sx
        x_var.periodic = 1 if topography.is_periodic else 0
        if 'unit' in topography.info:
            x_var.length_unit = topography.info['unit']
        x_var[...] = (np.arange(nx) + 0.5) * sx / nx
        y_var.length = sy
        y_var.periodic = 1 if topography.is_periodic else 0
        if 'unit' in topography.info:
            y_var.length_unit = topography.info['unit']
        y_var[...] = (np.arange(ny) + 0.5) * sy / ny

        if topography.is_domain_decomposed:
            heights_var.set_collective(True)
        heights_var[topography.subdomain_slices] = topography.heights()


# Register analysis functions from this module
UniformTopographyInterface.register_function('to_netcdf', write_nc)
