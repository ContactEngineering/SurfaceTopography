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

import json

import numpy as np

from numpyencoder import NumpyEncoder

from ..HeightContainer import UniformTopographyInterface, NonuniformLineScanInterface
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..UnitConversion import mangle_length_unit_utf8, mangle_length_unit_ascii

from .Reader import ReaderBase, ChannelInfo, MetadataAlreadyFixedByFile

# We run serial I/O through scipy. This has several advantages:
# 1) lightweight, 2) can handle streams
from scipy.io.netcdf import netcdf_file

format_to_scipy_version = {
    'NETCDF3_CLASSIC': 1,
    'NETCDF3_64BIT_OFFSET': 2
}


# we subclass the netcdf_file class such it
# is not closed by garbage collector in case of file streams
class _SpecialNetCDFFile(netcdf_file):
    def close(self):
        if self.filename != 'None':
            super().close()
        else:
            self.flush()
        # we want to disable calling closing the file
        # in case of file streams
    __del__ = close


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
        x:length = 3 ;
        x:periodic = 1 ;
        x:unit = "um" ;
    double y(y) ;
        y:length = 3 ;
        y:periodic = 1 ;
        y:unit = "um" ;
    double heights(x, y) ;
        heights:unit = "um" ;
}
```
The following code snippets reads the file and displays the topography data as
a two-dimensional color map in Python:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.netcdf import netcdf_file

nc = netcdf_file('test.nc')
heights = np.array(nc.variables['heights'][...])
x = np.array(nc.variables['x'][...])
y = np.array(nc.variables['y'][...])
unit = nc.variables['x'].unit

plt.figure()
plt.subplot(aspect=1)
plt.pcolormesh(x, y, heights.T)

plt.show()
```
'''

    def __init__(self, fobj, communicator=None):
        self._nc = None
        self._var_kwargs = {}  # Keywords arguments to attach to variable creation
        if communicator is not None and communicator.size > 1:
            # For parallel I/O we need netCDF4
            from netCDF4 import Dataset
            self._nc = Dataset(fobj, 'r', parallel=True, comm=communicator, format='NETCDF3_64BIT_OFFSET')
        else:
            # We need to check magic ourselves if this is a stream because
            # netcdf_file may close the stream in case of error, probably
            # because of garbage collection
            if hasattr(fobj, 'seek'):
                magic = fobj.read(3)
                if not magic == b'CDF':
                    raise TypeError('File or stream is not a valid NetCDF 3 file')
                # we must rewind the file
                fobj.seek(0)

            self._nc = _SpecialNetCDFFile(fobj, 'r', maskandscale=True)

        self._communicator = communicator
        self._n_dim = self._nc.dimensions['n'] if 'n' in self._nc.dimensions else None
        self._x_dim = self._nc.dimensions['x'] if 'x' in self._nc.dimensions else None
        self._y_dim = self._nc.dimensions['y'] if 'y' in self._nc.dimensions else None
        self._x_var = self._nc.variables['x'] if 'x' in self._nc.variables else None
        self._y_var = self._nc.variables['y'] if 'y' in self._nc.variables else None
        self._heights_var = self._nc.variables['heights']

        # The following information may be missing from the NetCDF file
        self._physical_sizes = None
        self._periodic = False
        self._unit = None
        self._info = {}

        # Deserialize info dictionary, if present
        try:
            self._info = json.loads(self._nc.json)
        except AttributeError:
            pass

        # Determine physical sizes of topography
        if self._n_dim is not None:
            # This is a nonuniform topography
            pass
        else:
            # This is a uniform topography...
            if self._y_var is not None:
                # ...and it is 2D
                try:
                    self._physical_sizes = (self._x_var.length, self._y_var.length)
                except AttributeError:
                    pass
            else:
                # ...and it is 1D (a line scan)
                try:
                    self._physical_sizes = (self._x_var.length,)
                except AttributeError:
                    pass

        # Determine if the topography is periodic
        try:
            self._periodic = self._x_var.periodic != 0
        except AttributeError:
            pass

        # Determine unit of topography
        try:
            self._unit = mangle_length_unit_utf8(self._x_var.unit)
        except AttributeError:
            pass

    def __del__(self):
        self.close()

    def close(self):
        if self._nc is not None:
            if self._x_var is not None:
                del self._x_var
            if self._y_var is not None:
                del self._y_var
            if self._heights_var is not None:
                del self._heights_var

            del self._nc
            self._nc = None

    @property
    def channels(self):
        if self._x_dim is not None:
            # This is a uniform topography
            try:
                # netCDF4
                nx = self._x_dim.size
            except AttributeError:
                # scipy.io.netcdf_file
                nx = self._x_dim
            nb_grid_pts = (nx,)
            if self._y_dim is not None:
                # This is a 2D topography
                try:
                    # netCDF4
                    ny = self._y_dim.size
                except AttributeError:
                    # scipy.io.netcdf_file
                    ny = self._y_dim
                nb_grid_pts = (nx, ny)
        else:
            # This is a nonuniform line scan
            try:
                # netCDF4
                nx = self._x_dim.size
            except AttributeError:
                # scipy.io.netcdf_file
                nx = self._x_dim
            nb_grid_pts = (nx,)
        return [ChannelInfo(self, 0,
                            name='Default',
                            dim=len(nb_grid_pts),
                            nb_grid_pts=nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            periodic=self._periodic,
                            unit=self._unit,
                            info=self._info)]

    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None, nb_subdomain_grid_pts=None):
        if channel_index is not None and channel_index != 0:
            raise ValueError('`channel_index` must be None or 0.')

        if unit is not None:
            if self._unit is not None:
                raise MetadataAlreadyFixedByFile('unit')
        else:
            unit = self._unit

        _info = self._info.copy()
        _info.update(info)

        heights = self._heights_var[...]
        if not np.ma.is_masked(heights):
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)
        if not np.ma.is_masked(heights):
            # If is it not masked, make sure it becomes a proper numpy array
            heights = np.array(heights)
        heights = heights.copy()

        if subdomain_locations is None and nb_subdomain_grid_pts is None:
            if self._x_dim is not None:
                physical_sizes = self._check_physical_sizes(physical_sizes,
                                                            self._physical_sizes)
                # This is a uniform topography...
                if self._y_dim is not None:
                    # ...and it is 2D
                    topo = Topography(heights, physical_sizes,
                                      periodic=self._periodic if periodic is None else periodic,
                                      unit=unit, info=_info)
                else:
                    # .. and it is 1D
                    topo = UniformLineScan(heights, physical_sizes,
                                           periodic=self._periodic if periodic is None else periodic,
                                           unit=unit, info=_info)
            else:
                if physical_sizes is not None:
                    raise ValueError('You cannot specify physical sizes for a nonuniform topography.')

                if self._periodic:
                    raise ValueError('NetCDF file indicates periodicity, but this is a nonuniform line scan which '
                                     'cannot be periodic.')

                # This is a nonuniform line scan
                topo = NonuniformLineScan(self._x_var[...].copy(), heights,
                                          unit=unit, info=_info)
        else:
            if self._y_dim is None:
                raise ValueError('Parallel reading only works for topographies, not line scans.')

            physical_sizes = self._check_physical_sizes(physical_sizes,
                                                        self._physical_sizes)

            topo = Topography(heights, physical_sizes,
                              periodic=self._periodic if periodic is None else periodic,
                              decomposition='domain',
                              subdomain_locations=subdomain_locations,
                              nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                              communicator=self.communicator,
                              unit=unit,
                              info=_info)

        if height_scale_factor is not None:
            if 'unit' in _info:
                raise MetadataAlreadyFixedByFile('height_scale_factor',
                                                 'The height_scale_factor cannot be given for NC files '
                                                 'if units are present in the file.')
                # See https://github.com/ContactEngineering/SurfaceTopography/pull/99#discussion_r648364687
            topo = topo.scale(height_scale_factor)

        return topo

    @property
    def communicator(self):
        return self._communicator

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def write_nc_uniform(topography, fobj, format='NETCDF3_64BIT_OFFSET'):
    """
    Write topography into a NetCDF file.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        The topography to write to disk.
    fobj : str or stream
        Name of the NetCDF file or file stream
    format : str
        NetCDF file format. Default is 'NETCDF3_64BIT_OFFSET'.
    """
    if topography.communicator is not None and topography.communicator.size > 1:
        from netCDF4 import Dataset, default_fillvals
        kwargs = dict(format=format,
                      parallel=topography.is_domain_decomposed,
                      comm=topography.communicator)
        var_kwargs = {'fill_value': default_fillvals['f8']}
    else:
        Dataset = _SpecialNetCDFFile
        kwargs = dict(version=format_to_scipy_version[format],
                      maskandscale=True)
        var_kwargs = {}
    if not topography.is_domain_decomposed and \
            topography.communicator.rank > 1:
        return
    with Dataset(fobj, 'w', **kwargs) as nc:
        # Serialize info dictionary as JSON and write to NetCDF file
        info = topography.info.copy()
        try:
            # We store the unit information separately
            del info['unit']
        except KeyError:
            pass
        if info != {}:
            nc.json = json.dumps(info, ensure_ascii=True, cls=NumpyEncoder)

        # Create dimensions and heights variable
        try:
            nx, ny = topography.nb_grid_pts
        except ValueError:
            nx, = topography.nb_grid_pts

        nc.createDimension('x', nx)
        if topography.dim > 1:
            nc.createDimension('y', ny)
            heights_var = nc.createVariable('heights', 'f8', ('x', 'y'), **var_kwargs)
        else:
            heights_var = nc.createVariable('heights', 'f8', ('x',), **var_kwargs)

        # Create variables for x- and y-positions, but only if physical_sizes
        # exist. (physical_sizes should always exist, but who knows...)
        if topography.physical_sizes is not None:
            x = topography.positions()
            try:
                sx, sy = topography.physical_sizes
                x, y = x
            except ValueError:
                sx, = topography.physical_sizes

            x_var = nc.createVariable('x', 'f8', ('x',), **var_kwargs)
            x_var.length = sx

            x_var.periodic = 1 if topography.is_periodic else 0
            if topography.unit is not None:
                # scipy.io.netcdf_file does not support UTF-8
                x_var.unit = mangle_length_unit_ascii(topography.unit)

            if topography.dim > 1:
                x_var[...] = np.arange(nx) / nx * sx

                y_var = nc.createVariable('y', 'f8', ('y',), **var_kwargs)

                y_var.length = sy
                y_var.periodic = 1 if topography.is_periodic else 0
                if topography.unit is not None:
                    # scipy.io.netcdf_file does not support UTF-8
                    y_var.unit = mangle_length_unit_ascii(topography.unit)
                y_var[...] = np.arange(ny) / ny * sy

        if topography.is_domain_decomposed:
            heights_var.set_collective(True)
            heights_var[topography.subdomain_slices] = topography.heights()
        else:
            heights_var[...] = topography.heights()
        if topography.unit is not None:
            heights_var.unit = mangle_length_unit_ascii(topography.unit)


def write_nc_nonuniform(line_scan, fobj, format='NETCDF3_64BIT_OFFSET'):
    """
    Write nonuniform line scan into a NetCDF file.

    Parameters
    ----------
    line_scan : :obj:`SurfaceTopography`
        The topography to write to disk.
    fobj : str or stream
        Name of the NetCDF file or file stream
    format : str
        NetCDF file format. Default is 'NETCDF3_64BIT_OFFSET'.
    """
    if line_scan.communicator is not None and line_scan.communicator.size > 1:
        raise RuntimeError('Parallel writing is not supported for nonuniform line scans')

    with _SpecialNetCDFFile(fobj, 'w', version=format_to_scipy_version[format], maskandscale=True) as nc:
        # Serialize info dictionary as JSON and write to NetCDF file
        info = line_scan.info.copy()
        try:
            # We store the unit information separately
            del info['unit']
        except KeyError:
            pass
        if info != {}:
            nc.json = json.dumps(info)

        nx, = line_scan.nb_grid_pts

        nc.createDimension('n', nx)

        x_var = nc.createVariable('x', 'f8', ('n',))
        heights_var = nc.createVariable('heights', 'f8', ('n',))

        x_var[...] = line_scan.positions()
        if line_scan.unit is not None:
            # scipy.io.netcdf_file does not support UTF-8
            x_var.unit = mangle_length_unit_ascii(line_scan.unit)
        heights_var[...] = line_scan.heights()


# Register analysis functions from this module
UniformTopographyInterface.register_function('to_netcdf', write_nc_uniform)
NonuniformLineScanInterface.register_function('to_netcdf', write_nc_nonuniform)
