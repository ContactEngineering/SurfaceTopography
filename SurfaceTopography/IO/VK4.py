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
import datetime
#
# Reference information and implementations:
# https://www.osti.gov/biblio/1419732
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/keyence.c
# https://github.com/torkian/vk4-python-driver

from struct import unpack

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class VK4Reader(ReaderBase):
    _format = 'vk4'
    _name = 'Keyence VK4'
    _description = '''
VK4 file format of the Keyence laser conformal microscope.
'''

    _MAGIC4 = b'VK4_'
    _MAGIC0 = b'\x00\x00\x00\x00'

    _offset_table_structure = [
        ('setting', 'I'),
        ('color_peak', 'I'),
        ('color_light', 'I'),
        ('light1', 'I'),
        ('light2', 'I'),
        ('light3', 'I'),
        ('height1', 'I'),
        ('height2', 'I'),
        ('height3', 'I'),
        ('color_peak_thumbnail', 'I'),
        ('color_thumbnail', 'I'),
        ('light_thumbnail', 'I'),
        ('height_thumbnail', 'I'),
        ('assemble', 'I'),
        ('line_measure', 'I'),
        ('line_thickness', 'I'),
        ('string_data', 'I'),
        ('reserved', 'I'),
    ]

    _header_structure = [
        ('size', 'I'),
        ('year', 'I'),
        ('month', 'I'),
        ('day', 'I'),
        ('hour', 'I'),
        ('minute', 'I'),
        ('second', 'I'),
        ('diff_utc_by_minutes', 'i'),
        ('image_attributes', 'I'),
        ('user_interface_mode', 'I'),
        ('color_composite_mode', 'I'),
        ('num_layer', 'I'),
        ('run_mode', 'I'),
        ('peak_mode', 'I'),
        ('sharpening_level', 'I'),
        ('speed', 'I'),
        ('distance', 'I'),
        ('pitch', 'I'),
        ('optical_zoom', 'I'),
        ('num_line', 'I'),
        ('line0_pos', 'I'),
        ('reserved1', 'I'),
        ('reserved2', 'I'),
        ('reserved3', 'I'),
        ('lens_mag', 'I'),
        ('pmt_gain_mode', 'I'),
        ('pmt_gain', 'I'),
        ('pmt_offset', 'I'),
        ('nd_filter', 'I'),
        ('reserved2', 'I'),
        ('persist_count', 'I'),
        ('shutter_speed_mode', 'I'),
        ('shutter_speed', 'I'),
        ('white_balance_mode', 'I'),
        ('white_balance_red', 'I'),
        ('white_balance_blue', 'I'),
        ('camera_gain', 'I'),
        ('plane_compensation', 'I'),
        ('length_unit', 'I'),
        ('height_unit', 'I'),
        ('xy_decimal_place', 'I'),
        ('height_decimal_place', 'I'),
        ('x_length_per_pixel', 'I'),
        ('y_length_per_pixel', 'I'),
        ('height_scale_factor', 'I'),
        ('reserved3[5]', 'I'),
        ('light_filter_type', 'I'),
        ('reserved4', 'I'),
        ('gamma_reverse', 'I'),
        ('gamma', 'I'),
        ('gamma_offset', 'I'),
        ('ccd_bw_offset', 'I'),
        ('numerical_aperture', 'I'),
        ('head_type', 'I'),
        ('pmt_gain2', 'I'),
        ('omit_color_image', 'I'),
        ('lens_id', 'I'),
        ('light_lut_mode', 'I'),
        ('light_lut_in0', 'I'),
        ('light_lut_out0', 'I'),
        ('light_lut_in1', 'I'),
        ('light_lut_out1', 'I'),
        ('light_lut_in2', 'I'),
        ('light_lut_out2', 'I'),
        ('light_lut_in3', 'I'),
        ('light_lut_out3', 'I'),
        ('light_lut_in4', 'I'),
        ('light_lut_out4', 'I'),
        ('upper_position', 'I'),
        ('lower_position', 'I'),
        ('light_effective_bit_depth', 'I'),
        ('height_effective_bit_depth', 'I'),
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Check VK4 file magic
            if f.read(len(self._MAGIC4)) != self._MAGIC4:  # 'VK4_'
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')
            f.read(4)  # dll version
            if f.read(len(self._MAGIC0)) != self._MAGIC0:  # All zeros
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')

            # Offset table
            self._offset_table = decode(f, self._offset_table_structure, '<')

            # Measurement conditions
            self._header = decode(f, self._header_structure, '<')

            # Right now, we are assuming that there is only a single (height)
            # channel per VK4 file. Not sure if this is correct.

            # Height data
            f.seek(self._offset_table['height1'])
            self._width, self._height, self._itemsize, compression, byte_size = unpack('<IIIII', f.read(20))

            if byte_size != self._width * self._height * self._itemsize // 8:
                raise CorruptFile('Reported size does not match image dimensions.')

            if self._itemsize not in [8, 16, 32]:
                raise CorruptFile(f'Reported item size is {self._itemsize} bits, expected 8, 16 or 32 bits.')

            self._physical_sizes = ((self._width - 1) * self._header['x_length_per_pixel'],
                                    (self._height - 1) * self._header['y_length_per_pixel'])
            self._unit = 'pm'

            self._info = {
                'acquisition_time':
                    str(datetime.datetime(
                        self._header['year'], self._header['month'], self._header['day'],
                        self._header['hour'], self._header['minute'], self._header['second'],
                        tzinfo=datetime.timezone(datetime.timedelta(minutes=self._header['diff_utc_by_minutes']))
                    )),
                'raw_metadata': self._header
            }

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=(self._width, self._height),
                            physical_sizes=self._physical_sizes,
                            uniform=True,
                            unit=self._unit)]

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
            f.seek(self._offset_table['height1']
                   + 7 * 8  # width, height, bit depth, compression, byte size, palette min, palette max
                   + 768  # palette
                   )
            dtype = np.uint8 if self._itemsize == 8 else np.uint16 if self._itemsize == 16 else np.uint32
            height_data = np.fromfile(f, dtype=dtype, count=self._width * self._height) \
                .reshape((self._height, self._width)).T

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(float(self._header['height_scale_factor']))
