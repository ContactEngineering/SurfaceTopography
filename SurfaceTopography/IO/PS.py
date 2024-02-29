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

# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/psia.c

import io

import numpy as np

from tiffile import TiffFile, TiffFileError

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor


class PSReader(ReaderBase):
    _format = 'ps'
    _mime_types = ['image/tiff']
    _file_extensions = ['tiff', 'tif']

    _name = 'Park Systems'
    _description = '''
TIFF-based file format of Park Systems instruments.
'''

    _MAGIC = 0x0E031301
    _VERSION1 = 0x1000001
    _VERSION2 = 0x1000002

    _TAG_MAGIC = 50432
    _TAG_VERSION = 50433
    _TAG_DATA = 50434
    _TAG_HEADER = 50435

    _IMAGE_TYPE_2D_MAPPED = 0
    _IMAGE_TYPE_LINE_PROFILE = 1
    _IMAGE_TYPE_SPECTROSCOPY = 2

    _DATA_TYPE_INT16 = 0
    _DATA_TYPE_INT32 = 1
    _DATA_TYPE_FLOAT = 2

    _header_structure = [
        ('image_type', 'I'),
        ('source_name', '32U'),
        ('image_mode', '8U'),
        ('lpf_strength', 'd'),
        ('auto_flatten', 'I'),
        ('ac_track', 'I'),
        ('nb_grid_pts_x', 'I'),
        ('nb_grid_pts_y', 'I'),
        ('angle', 'd'),
        ('sine_scan', 'I'),
        ('overscan_rate', 'd'),
        ('forward', 'I'),
        ('scan_up', 'I'),
        ('swap_xy', 'I'),
        ('physical_size_x', 'd'),
        ('physical_size_y', 'd'),
        ('xoff', 'd'),
        ('yoff', 'd'),
        ('scan_rate', 'd'),
        ('set_point', 'd'),
        ('set_point_unit', '8U'),
        ('tip_bias', 'd'),
        ('sample_bias', 'd'),
        ('data_gain', 'd'),
        ('data_scale_factor', 'd'),
        ('data_offset', 'd'),
        ('data_unit', '8U'),
        ('data_min', 'i'),
        ('data_max', 'i'),
        ('data_avg', 'i'),
        ('compression', 'I'),
        ('logscale', 'I'),
        ('square', 'I'),
    ]

    _version2_header_structure = [
        ('data_servo_gain', 'd'),
        ('data_scanner_range', 'd'),
        ('xy_voltage_mode', '8U'),
        ('data_voltage_mode', '8U'),
        ('xy_servo_mode', '8U'),
        ('data_type', 'I'),
        ('reserved1', 'I'),
        ('reserved2', 'I'),
        ('ncm_amplitude', 'd'),
        ('ncm_frequency', 'd'),
        ('cantilever_name', '16U'),
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self._file_path = file_path
        with OpenFromAny(self._file_path, 'rb') as f:
            try:
                with TiffFile(f) as t:
                    if len(t.pages) != 1:
                        raise FileFormatMismatch('More than one image in TIFF. This is not a Park Systems TIFF.')
                    p = t.pages[0]

                    # Check file magic and version information
                    if p.tags[self._TAG_MAGIC].value != self._MAGIC:
                        raise FileFormatMismatch('This is not a Park Systems TIFF.')
                    self._version = p.tags[self._TAG_VERSION].value
                    if self._version not in [self._VERSION1, self._VERSION2]:
                        raise CorruptFile('Only version 1 and 2 of Park Systems TIFFs are supported.')

                    # Parse header
                    header_data = p.tags[self._TAG_HEADER].value
                    if (self._version == self._VERSION1 and len(header_data) < 356) or (
                            self._version == self._VERSION2 and len(header_data) < 580):
                        raise CorruptFile('Header too short.')

                    header_stream = io.BytesIO(header_data)
                    self._header = decode(header_stream, self._header_structure, '<')

                    if self._version == self._VERSION2:
                        self._header.update(decode(header_stream, self._version2_header_structure, '<'))
                    else:
                        self._header['data_type'] = self._DATA_TYPE_INT16

                    if self._header['data_type'] not in [self._DATA_TYPE_INT16, self._DATA_TYPE_INT32,
                                                         self._DATA_TYPE_FLOAT]:
                        raise CorruptFile('Cannot handle data type {}'.format(self._header['data_type']))

                    self._nb_grid_pts = (self._header['nb_grid_pts_x'], self._header['nb_grid_pts_y'])
                    self._physical_sizes = (self._header['physical_size_x'], self._header['physical_size_y'])
                    self._unit = 'µm'

                    unit_conversion_factor = get_unit_conversion_factor(self._header['data_unit'], 'µm')
                    self._height_scale_factor = \
                        self._header['data_scale_factor'] * self._header['data_gain'] * unit_conversion_factor

                    self._info = {'raw_metadata': self._header}
            except TiffFileError:
                raise FileFormatMismatch('This is not a TIFF file, so it cannot be a Park Systems TIFF.')

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name=self._header['source_name'],
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=self._height_scale_factor,
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
            raise RuntimeError(
                f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        with OpenFromAny(self._file_path, 'rb') as f:
            with TiffFile(f) as t:
                # The data itself is not the image, but inside a tag
                raw_data = t.pages[0].tags[self._TAG_DATA].value
                t = self._header['data_type']
                dtype = np.int16 if t == self._DATA_TYPE_INT16 else \
                    np.int32 if t == self._DATA_TYPE_INT32 else np.single
                height_data = \
                    np.frombuffer(raw_data, dtype=dtype, count=np.prod(self._nb_grid_pts)).reshape(self._nb_grid_pts).T

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(self._height_scale_factor)
