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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/alicona.c
#

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class AL3DReader(ReaderBase):
    _format = 'al3d'
    _mime_types = ['application/x-alicona-imaging-al3d']
    _file_extensions = ['al3d']

    _name = 'Alicona Imaging AL3D'
    _description = '''
AL3D format of Alicona Imaging.
'''

    _MAGIC = b'AliconaImaging\x00\r\n'

    # Relative tolerance for catching invalid pixels
    _INVALID_RELTOL = 1.5e-7

    _tag_structure = [
        ('key', '20s'),
        ('value', '30s'),
        ('crlf', '2s'),
    ]

    _header_structure = [
        ('nb_tags', 'I'),
        ('nb_grid_pts_x', 'I'),
        ('nb_grid_pts_y', 'I'),
        ('nb_planes', 'I'),
        ('grid_spacing_x', 'd'),
        ('grid_spacing_y', 'd'),
        ('icon_offset', 'I'),
        ('texture_offset', 'I'),
        ('depth_offset', 'I'),
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Detect file version
            magic = f.read(len(self._MAGIC))

            # Check AL3D file magic
            if magic != self._MAGIC:
                # This is not an AL3D file
                raise FileFormatMismatch

            version = int(self._read_tag(f, 'Version'))
            nb_tags = int(self._read_tag(f, 'TagCount'))

            self._header = {'Version': version, 'TagCount': nb_tags}
            for i in range(nb_tags):
                tag = decode(f, self._tag_structure, '<')
                if tag['crlf'] != '\r\n':
                    raise CorruptFile('CRLF tag terminator missing.')
                self._header[tag['key']] = tag['value']

            nx, ny = int(self._header['Cols']), int(self._header['Rows'])
            self._nb_grid_pts = (nx, ny)
            self._physical_sizes = (nx * float(self._header['PixelSizeXMeter']),
                                    ny * float(self._header['PixelSizeYMeter']))

        self._unit = 'm'

    def _read_tag(self, f, key):
        tag = decode(f, self._tag_structure, '<')
        if tag['key'] != key:
            raise CorruptFile('Expected tag key {}, found {}.'.format(key, tag['key']))
        if tag['crlf'] != '\r\n':
            raise CorruptFile('CRLF tag terminator missing.')
        return tag['value']

    def read_height_data(self, f):
        f.seek(int(self._header['DepthImageOffset']))
        invalid_pixel_value = float(self._header['InvalidPixelValue'])
        dtype = np.single
        buffer = f.read(np.prod(self._nb_grid_pts) * np.dtype(dtype).itemsize)
        nx, ny = self._nb_grid_pts
        data = np.frombuffer(buffer, dtype=dtype).reshape((ny, nx))
        mask = np.isnan(data)
        if not np.isnan(invalid_pixel_value):
            mask = np.logical_or(mask, np.abs(data - invalid_pixel_value) < self._INVALID_RELTOL * invalid_pixel_value)
        return np.ma.masked_array(data.T, mask=mask.T)

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            uniform=True,
                            unit=self._unit,
                            height_scale_factor=1,
                            info={'raw_metadata': self._header})]

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

        _info = info.copy()
        _info['raw_metadata'] = self._header

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(1)
