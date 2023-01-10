#
# Copyright 2022 Lars Pastewka
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

#
# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/surffile.c
#

import datetime

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor


class SURReader(ReaderBase):
    _format = 'sur'
    _name = 'Digital Surf'
    _description = '''
This reader imports Digital Surf SUR data files.
'''

    _MAGIC = b'DIGITAL SURF'

    _header_structure = [
        ('format', 'H'),
        ('nb_objects', 'H'),
        ('version', 'H'),
        ('type', 'H'),
        ('object_name', '30s'),
        ('instrument_name', '30s'),
        ('material_code', 'H'),
        ('acquisition', 'H'),
        ('range', 'H'),
        ('special_points', 'H'),
        ('absolute', 'H'),
        ('_', 'I'),
        ('_', 'I'),
        ('itemsize', 'H'),
        ('zmin', 'i'),
        ('zmax', 'i'),
        ('nb_grid_pts_x', 'i'),
        ('nb_grid_pts_y', 'i'),
        ('nb_points', 'I'),
        ('grid_spacing_x', 'f'),
        ('grid_spacing_y', 'f'),
        ('height_scale_factor', 'f'),
        ('name_x', '16s'),
        ('name_y', '16s'),
        ('data_name', '16s'),
        ('unit_delta_x', '16s'),
        ('unit_delta_y', '16s'),
        ('delta_data_unit', '16s'),
        ('unit_x', '16s'),
        ('unit_y', '16s'),
        ('data_unit', '16s'),
        ('unit_ratio_x', 'f'),
        ('unit_ratio_y', 'f'),
        ('data_unit_ratio', 'f'),
        ('imprint', 'H'),
        ('inversion', 'H'),
        ('leveling', 'H'),
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'I'),
        ('second', 'H'),
        ('minute', 'H'),
        ('hour', 'H'),
        ('day', 'H'),
        ('month', 'H'),
        ('year', 'H'),
        ('dayof', 'H'),
        ('measurement_duration', 'f'),
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'H'),
        ('comment_size', 'H'),
        ('private_size', 'H'),
        ('client_zone', '128s'),
        ('x_offset', 'f'),
        ('y_offset', 'f'),
        ('data_offset', 'f'),
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Detect file magic
            if f.read(len(self._MAGIC)) != self._MAGIC:
                raise FileFormatMismatch('File magic does not match. This is not a Digital Surf file.')

            self._header = decode(f, self._header_structure, '<')

        nx = self._header['nb_grid_pts_x']
        ny = self._header['nb_grid_pts_y']
        if nx * ny != self._header['nb_points']:
            raise CorruptFile(
                'The file reported a grid of {} x {} data points and a total number of {} data points, which is '
                'inconsistent'.format(self._nb_grid_pts[0], self._nb_grid_pts[1], self._header['nb_points']))

        if self._header['itemsize'] not in [16, 32]:
            raise CorruptFile('The file reported an item size of {} bits, which I cannot read.'
                              .format(self._header['itemsize']))

        # Check that units and delta units are the same. Not sure if they differ and why there are two different sets
        # of units provided.
        unit_x = self._header['unit_x']
        unit_y = self._header['unit_y']
        self._unit = self._header['data_unit']  # We use the data unit as the primary unit for topography objects

        if self._header['unit_delta_x'] != unit_x or self._header['unit_delta_y'] != unit_y or \
                self._header['delta_data_unit'] != self._unit:
            raise CorruptFile('Units and delta units differ. Not sure how to handle this.')

        # Get the conversion factors for converting x,y to the main system of units
        fac_x = get_unit_conversion_factor(unit_x, self._unit)
        fac_y = get_unit_conversion_factor(unit_y, self._unit)

        # All good, now initialize some convenience variables

        self._nb_grid_pts = (nx, ny)
        self._physical_sizes = \
            (fac_x * self._header['grid_spacing_x'] * nx, fac_y * self._header['grid_spacing_y'] * ny)

        self._info = {
            'acquisition_time':
                str(datetime.datetime(self._header['year'], self._header['month'], self._header['day'],
                                      self._header['hour'], self._header['minute'], self._header['second'])),
            'instrument': {'name': self._header['instrument_name']},
            'raw_metadata': self._header
        }

    def read_height_data(self, f):
        if self._header['itemsize'] == 16:
            dtype = np.dtype('<i2')
        elif self._header['itemsize'] == 32:
            dtype = np.dtype('<i4')
        else:
            raise RuntimeError('Unknown itemsize')  # Should not happen because we check this in the constructor

        f.seek(512)

        nx, ny = self._nb_grid_pts
        buffer = f.read(nx * ny * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype=dtype).reshape(self._nb_grid_pts)

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=float(self._header['height_scale_factor']),
                            uniform=True,
                            unit=self._unit,
                            info=self._info)]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        with OpenFromAny(self.file_path, 'rb') as f:
            height_data = self.read_height_data(f)

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(float(self._header['height_scale_factor']))
