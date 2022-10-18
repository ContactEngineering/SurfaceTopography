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

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography

MAGIC = b'VK4_'
MAGIC0 = b'\x00\x00\x00\x00'

KEYENCE4_HEADER_SIZE = 12
KEYENCE4_OFFSET_TABLE_SIZE = 72
KEYENCE4_MEASUREMENT_CONDITIONS_MIN_SIZE = 304
KEYENCE4_ASSEMBLY_INFO_SIZE = 16
KEYENCE4_ASSEMBLY_CONDITIONS_SIZE = 8
KEYENCE4_ASSEMBLY_HEADERS_SIZE = KEYENCE4_ASSEMBLY_INFO_SIZE + KEYENCE4_ASSEMBLY_CONDITIONS_SIZE
KEYENCE4_ASSEMBLY_FILE_SIZE = 532
KEYENCE4_TRUE_COLOR_IMAGE_MIN_SIZE = 20
KEYENCE4_FALSE_COLOR_IMAGE_MIN_SIZE = 796
KEYENCE4_LINE_MEASUREMENT_LEN = 1024
KEYENCE4_LINE_MEASUREMENT_SIZE = 18440

KEYENCE4_NORMAL_FILE = 0
KEYENCE4_ASSEMBLY_FILE = 1
KEYENCE4_ASSEMBLY_FILE_UNICODE = 2


class VK4Reader(ReaderBase):
    _format = 'vk4'
    _name = 'Keyence VK4'
    _description = '''
File format of the Keyence laser conformal microscope.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Check VK4 file magic
            if f.read(len(MAGIC)) != MAGIC:  # 'VK4_'
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')
            f.read(4)  # dll version
            if f.read(len(MAGIC0)) != MAGIC0:  # All zeros
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')

            # Offset table
            f.seek(6 * 4, 1)
            self.height_offsets = unpack('<III', f.read(12))
            f.seek(9 * 4, 1)

            # Measurement conditions
            size, year, month, day, hour, minute, second, diff_utc_by_minutes = unpack('<IIIIIIIi', f.read(8 * 4))
            f.seek(30 * 4, 1)  # Skip 30 uint32s
            xy_length_unit, z_length_unit, xy_decimal_place, z_decimal_place, x_length_per_pixel, y_length_per_pixel, \
                self._height_scale_factor = unpack('<IIIIIII', f.read(7 * 4))

            # Right now, we are assuming that there is only a single (height)
            # channel per VK4 file. Not sure if this is correct.

            # Height data
            f.seek(self.height_offsets[0])
            self._width, self._height, self._itemsize = unpack('<III', f.read(12))

            compression, byte_size = unpack('<II', f.read(8))

            if byte_size != self._width * self._height * self._itemsize // 8:
                raise CorruptFile('Reported size does not match image dimensions.')

            if self._itemsize not in [8, 16, 32]:
                raise CorruptFile(f'Reported item size is {self._itemsize} bits, expected 8, 16 or 32 bits.')

            self._physical_sizes = ((self._width - 1) * x_length_per_pixel,
                                    (self._height - 1) * y_length_per_pixel)
            self._unit = 'pm'

            self._info = {
                'acquisition_time':
                    datetime.datetime(
                        year, month, day, hour, minute, second,
                        tzinfo=datetime.timezone(datetime.timedelta(minutes=diff_utc_by_minutes))
                    )
            }

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
                   periodic=False, subdomain_locations=None,
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
            f.seek(self.height_offsets[0]
                   + 7 * 8  # width, height, bit depth, compression, byte size, palette min, palette max
                   + 768  # palette
                   )
            dtype = np.uint8 if self._itemsize == 8 else np.uint16 if self._itemsize == 16 else np.uint32
            height_data = np.fromfile(f, dtype=dtype, count=self._width * self._height) \
                .reshape((self._height, self._width)).T

        info = self._info.copy()
        info.update(info)

        topo = Topography(height_data.astype(np.float), self._physical_sizes, unit=self._unit, periodic=True, info=info)
        return topo.scale(self._height_scale_factor)
