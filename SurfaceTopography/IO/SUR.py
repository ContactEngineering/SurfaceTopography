#
# Copyright 2022-2023 Lars Pastewka
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

from ..Exceptions import (CorruptFile, FileFormatMismatch,
                          UnsupportedFormatFeature)
from ..Support.UnitConversion import get_unit_conversion_factor
from .binary import BinaryArray, BinaryStructure, Convert, Validate
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase


class SURReader(DeclarativeReaderBase):
    _format = 'sur'
    _mime_types = ['application/x-surf-spm']
    _file_extensions = ['sur']

    _name = 'Digital Surf'
    _description = '''
This reader imports Digital Surf SUR data files.
'''

    _file_layout = CompoundLayout([
        BinaryStructure([
            ('magic', '12s', Validate('DIGITAL SURF', FileFormatMismatch)),
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
            ('itemsize', 'H', Validate(lambda x, context: x in [16, 32], UnsupportedFormatFeature)),
            ('zmin', 'i'),
            ('zmax', 'i'),
            ('nb_grid_pts_x', 'i', Validate(lambda x, context: x > 0, CorruptFile)),
            ('nb_grid_pts_y', 'i', Validate(lambda x, context: x > 0, CorruptFile)),
            ('nb_points', 'I',
             Validate(lambda x, context: x == context['nb_grid_pts_x'] * context['nb_grid_pts_y'], CorruptFile)),
            ('grid_spacing_x', 'f', Validate(lambda x, context: x > 0, CorruptFile)),
            ('grid_spacing_y', 'f', Validate(lambda x, context: x > 0, CorruptFile)),
            ('height_scale_factor', 'f', Convert(float)),
            ('name_x', '16s'),
            ('name_y', '16s'),
            ('data_name', '16s'),
            ('delta_x_unit', '16s'),
            ('delta_y_unit', '16s'),
            ('delta_data_unit', '16s'),
            ('x_unit', '16s'),
            ('y_unit', '16s'),
            ('data_unit', '16s'),
            ('x_unit_ratio', 'f',
             Validate(lambda x, context: get_unit_conversion_factor(context.x_unit, context.delta_x_unit))),
            ('y_unit_ratio', 'f',
             Validate(lambda x, context: get_unit_conversion_factor(context.y_unit, context.delta_y_unit))),
            ('data_unit_ratio', 'f',
             Validate(lambda x, context: get_unit_conversion_factor(context.data_unit, context.delta_data_unit))),
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
            (None, '34b')
        ], name='header'),
        BinaryArray(
            'data',
            lambda context: (context.header.nb_grid_pts_y, context.header.nb_grid_pts_x),
            lambda context: np.dtype('<i2') if context.header.itemsize == 16 else np.dtype('<i4'),
            conversion_fun=lambda arr: arr.T
        )
    ])

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

        unit = header.delta_data_unit
        fac_x = get_unit_conversion_factor(header.delta_x_unit, unit)
        fac_y = get_unit_conversion_factor(header.delta_y_unit, unit)

        return [ChannelInfo(
            self,
            0,  # channel index
            name='Default',
            dim=2,
            nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
            physical_sizes=(fac_x * header.grid_spacing_x * header.nb_grid_pts_x,
                            fac_y * header.grid_spacing_y * header.nb_grid_pts_y),
            height_scale_factor=header.height_scale_factor,
            uniform=True,
            unit=unit,
            info=info,
            tags={'reader': self._metadata.data}
        )]
