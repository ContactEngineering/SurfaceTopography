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
# https://www.osti.gov/biblio/1419732
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/keyence.c
# https://github.com/torkian/vk4-python-driver
#

import datetime
from zipfile import ZipFile

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class VKReader(ReaderBase):
    _format = 'vk'
    _mime_types = ['application/x-keyence-vk3',
                   'application/x-keyence-vk4',
                   'application/x-keyence-vk5',
                   'application/x-keyence-vk6']
    _file_extensions = ['vk3', 'vk4', 'vk6', 'vk7']

    _name = 'Keyence VK'
    _description = '''
VK3, VK4, VK6 and VK7 file formats of the Keyence laser confocal microscope.
'''

    _MAGIC3 = b'VK3_'
    _MAGIC4 = b'VK4_'
    _MAGIC0 = b'\x00\x00\x00\x00'

    _MAGIC6 = b'VK6'
    _MAGIC7 = b'VK7'

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
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'I'),
        ('lens_mag', 'I'),
        ('pmt_gain_mode', 'I'),
        ('pmt_gain', 'I'),
        ('pmt_offset', 'I'),
        ('nd_filter', 'I'),
        ('_', 'I'),
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
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'I'),
        ('_', 'I'),
        ('light_filter_type', 'I'),
        ('_', 'I'),
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
        ('light_effective_itemsize', 'I'),
        ('height_effective_itemsize', 'I'),
    ]

    _image_structure = [
        ('width', 'I'),
        ('height', 'I'),
        ('itemsize', 'I'),
        ('compression', 'I'),
        ('byte_size', 'I'),
        ('palette_range_min', 'I'),
        ('palette_range_max', 'I'),
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        self._file_version = None
        self.read_vk3467_header(file_path)

    def read_vk3467_header(self, file_path):
        with OpenFromAny(file_path, 'rb') as f:
            # Detect file version
            magic = f.read(4)

            # Check VK* file magic
            if magic.startswith(self._MAGIC3):  # 'VK3_'
                file_version = 3
            elif magic.startswith(self._MAGIC4):  # 'VK4_'
                file_version = 4
            elif magic.startswith(self._MAGIC6):  # 'VK6'
                file_version = 6
            elif magic.startswith(self._MAGIC7):  # 'VK7'
                file_version = 7
            else:
                raise FileFormatMismatch('File magic does not match. This is not a Keyence VK file.')

            if self._file_version is None:
                self._file_version = file_version

            if file_version in [3, 4]:
                f.read(4)  # skip dll version
                if f.read(len(self._MAGIC0)) != self._MAGIC0:  # All zeros
                    raise FileFormatMismatch('File magic does not match. I thought this was a Keyence VK3 or VK4 file, '
                                             'but it seems this is not the case.')
                self.read_vk34_header(f)
            else:
                # VK6/7 contains a .zip file that has VK4 file name 'Vk4File'.
                # ZipFile gracefully skips VK6/7 header information before the
                # zip actually starts.
                with ZipFile(f, 'r') as z:
                    self.read_vk3467_header(z.open('Vk4File'))

    def read_vk34_header(self, f):
        # Offset table
        self._offset_table = decode(f, self._offset_table_structure, '<')

        # File version 4 (and 6, 7, which is 4) have an additional entry here
        if self._file_version in [4, 6, 7]:
            f.read(4)

        # Measurement conditions
        self._header = decode(f, self._header_structure, '<')

        # Right now, we are assuming that there is only a single (height)
        # channel per VK4 file. Not sure if this is correct.

        # Height data
        f.seek(self._offset_table['height1'])
        self._data = decode(f, self._image_structure, '<')

        if self._data['byte_size'] != self._data['width'] * self._data['height'] * self._data['itemsize'] // 8:
            raise CorruptFile('Reported size does not match image dimensions.')

        if self._data['itemsize'] not in [8, 16, 32]:
            raise CorruptFile('Reported item size is {} bits, expected 8, 16 or 32 bits.'
                              .format(self._data['itemsize']))

        self._physical_sizes = (float(self._data['width'] - 1) * self._header['x_length_per_pixel'],
                                float(self._data['height'] - 1) * self._header['y_length_per_pixel'])
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

    def read_height_data(self, f):
        f.seek(self._offset_table['height1'])
        image_header = decode(f, self._image_structure, '<')
        f.seek(768, 1)  # Skip palette
        assert image_header['width'] == self._data['width']
        assert image_header['height'] == self._data['height']
        dtype = np.dtype('<u1') if self._data['itemsize'] == 8 \
            else np.dtype('<u2') if self._data['itemsize'] == 16 \
            else np.dtype('<u4')
        buffer = f.read(self._data['width'] * self._data['height'] * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype=dtype).reshape((self._data['height'], self._data['width'])).T

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=(self._data['width'], self._data['height']),
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=self._header['height_scale_factor'],
                            uniform=True,
                            info=self._info,
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
            if self._file_version in [3, 4]:  # VK3 or VK4
                height_data = self.read_height_data(f)
            else:  # VK6 or VK7
                # ZipFile gracefully skips VK6 header information before the
                # zip actually starts.
                with ZipFile(f) as z:
                    height_data = self.read_height_data(z.open('Vk4File'))

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(float(self._header['height_scale_factor']))
