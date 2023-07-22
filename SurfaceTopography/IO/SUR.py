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

from .binary import BinaryStructure, Convert, Validate
from .common import OpenFromAny
from .Reader import DeclarativeReaderBase, ChannelInfo, FileLayout
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor


class SURReaderBase(DeclarativeReaderBase):
    _format = 'sur'
    _mime_types = ['application/x-surf-spm']
    _file_extensions = ['sur']

    _name = 'Digital Surf'
    _description = '''
This reader imports Digital Surf SUR data files.
'''

    _header = BinaryStructure('header', [
        ('magic', '12s', Validate(lambda magic, data: magic == 'DIGITAL SURF', FileFormatMismatch)),
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
        (None, 'I'),
        (None, 'I'),
        ('itemsize', 'H', Validate(lambda x, data: x in [16, 32], UnsupportedFormatFeature)),
        ('zmin', 'i'),
        ('zmax', 'i'),
        ('nb_grid_pts_x', 'i', Validate(lambda x, data: x > 0, CorruptFile)),
        ('nb_grid_pts_y', 'i', Validate(lambda x, data: x > 0, CorruptFile)),
        ('nb_points', 'I', Validate(lambda x, data: x == data['nb_grid_pts_x'] * data['nb_grid_pts_y'], CorruptFile)),
        ('grid_spacing_x', 'f', Validate(lambda x, data: x > 0, CorruptFile)),
        ('grid_spacing_y', 'f', Validate(lambda x, data: x > 0, CorruptFile)),
        ('height_scale_factor', 'f', Convert(float)),
        ('name_x', '16s'),
        ('name_y', '16s'),
        ('data_name', '16s'),
        ('unit_delta_x', '16s'),
        ('unit_delta_y', '16s', Validate(lambda x, data: x == data['unit_delta_x'], UnsupportedFormatFeature)),
        ('delta_data_unit', '16s', Validate(lambda x, data: x == data['unit_delta_x'], UnsupportedFormatFeature)),
        ('unit_x', '16s', Validate(lambda x, data: x == data['unit_delta_x'], UnsupportedFormatFeature)),
        ('unit_y', '16s', Validate(lambda x, data: x == data['unit_delta_x'], UnsupportedFormatFeature)),
        ('data_unit', '16s', Validate(lambda x, data: x == data['unit_delta_x'], UnsupportedFormatFeature)),
        ('unit_ratio_x', 'f'),
        ('unit_ratio_y', 'f'),
        ('data_unit_ratio', 'f'),
        ('imprint', 'H'),
        ('inversion', 'H'),
        ('leveling', 'H'),
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        ('second', 'H'),
        ('minute', 'H'),
        ('hour', 'H'),
        ('day', 'H'),
        ('month', 'H'),
        ('year', 'H'),
        ('dayof', 'H'),
        ('measurement_duration', 'f'),
        (None, 'I'),
        (None, 'I'),
        (None, 'H'),
        ('comment_size', 'H'),
        ('private_size', 'H'),
        ('client_zone', '128s'),
        ('x_offset', 'f'),
        ('y_offset', 'f'),
        ('data_offset', 'f'),
    ])

    _file_layout = FileLayout([
        _header
    ])

    def read_height_data(self, f):
        header = self._metadata.header
        if header.itemsize == 16:
            dtype = np.dtype('<i2')
        elif header.itemsize == 32:
            dtype = np.dtype('<i4')
        else:
            raise RuntimeError('Unknown itemsize')  # Should not happen because we check this in the constructor

        f.seek(512)

        buffer = f.read(header.nb_points * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype=dtype).reshape((header.nb_grid_pts_x, header.nb_grid_pts_y))

    @property
    def channels(self):
        header = self._metadata.header

        info = {
            'instrument': {'name': header.instrument_name},
            'raw_metadata': header
        }

        try:
            info['acquisition_time'] = \
                str(datetime.datetime(header.year, header.month, header.day,
                                      header.hour, header.minute, header.second))
        except ValueError:
            # This can fail if the date is not valid, e.g. if there are just zeros
            pass

        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
                            physical_sizes=(header.grid_spacing_x * header.nb_grid_pts_x,
                                            header.grid_spacing_y * header.nb_grid_pts_y),
                            height_scale_factor=header.height_scale_factor,
                            uniform=True,
                            unit=header.unit_x,
                            info=info)]

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

        channel = self.channels[0]

        with OpenFromAny(self.file_path, 'rb') as f:
            height_data = self.read_height_data(f)

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          channel.physical_sizes,
                          unit=channel.unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(self._metadata.header.height_scale_factor)
