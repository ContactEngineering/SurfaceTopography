#
# Copyright 2023 Lars Pastewka
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
import os
#
# The DI file format is described in detail here:
# http://www.physics.arizona.edu/~smanne/DI/software/fileformats.html
#

import dateutil.parser

import numpy as np

from .common import OpenFromAny
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8

from .Reader import ReaderBase, ChannelInfo


###

class WSXMReader(ReaderBase):
    _format = 'wsxm'
    _mime_types = ['application/x-wsxm-spm']
    _file_extensions = ['cur', 'stp', 'tom', 'top']

    _name = 'WSxM SPM data'
    _description = '''
This reader imports WSxM data files. WSxM is a standalone software package for
scanning probe microscopy available at http://www.wsxm.eu/.
'''

    # Binary data types
    _DTYPE_MAP = {
        'short': np.dtype('<i2'),
        'integer': np.dtype('<i2'),
        'float': np.dtype('<f4'),
        'double': np.dtype('<f8')
    }

    def __init__(self, file_path):
        """
        Load WSxM data files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path
        with OpenFromAny(self._file_path, 'rb') as fobj:
            L = fobj.readline().decode('latin-1').strip()
            if not L.startswith('WSxM file copyright'):
                raise FileFormatMismatch("This is not a WSxM file, because the first line does not start with "
                                         "'WSxM file copyright'.")
            L = fobj.readline().decode('latin-1').strip()
            if L != 'SxM Image file':
                raise FileFormatMismatch("This is not a WSxM file, because the second line does not equal "
                                         "'SxM Image file'.")
            L = fobj.readline().decode('latin-1').strip()
            _IMAGE_HEADER_SIZE = 'Image header size:'
            if not L.startswith(_IMAGE_HEADER_SIZE):
                raise CorruptFile("This WSxM file appear corrupted, because the third line does not start with "
                                  f"'{_IMAGE_HEADER_SIZE}'.")

            # image_header_size = int(L[len(_IMAGE_HEADER_SIZE):])

            metadata = {}
            section_name = None
            L = fobj.readline()
            while L is not None and L.decode('latin-1').strip() != '[Header end]':
                L = L.decode('latin-1').strip()
                if len(L) > 0:
                    if L.startswith('[') and L.endswith(']'):
                        # This is a section header
                        section_name = L[1:-1]
                        metadata[section_name] = {}
                    else:
                        if section_name is None:
                            raise CorruptFile('Found metadata entry, but did not yet encounter a section header.')
                        key, value = L.split(':', maxsplit=1)
                        metadata[section_name].update({key.strip(): value.strip()})
                L = fobj.readline()
            if section_name is None:
                raise IOError('No sections found in header.')

            # Topography size and data type
            nx = int(metadata['General Info']['Number of columns'])
            ny = int(metadata['General Info']['Number of rows'])
            dtype = self._DTYPE_MAP[metadata['General Info']['Image Data Type']]

            # Check file size
            file_offset = fobj.tell()
            fobj.seek(nx * ny * dtype.itemsize, os.SEEK_CUR)
            if fobj.tell() - file_offset != nx * ny * dtype.itemsize:
                raise CorruptFile('File has wrong size for data buffer.')

            # Physical sizes and units
            physical_size_x, x_unit = [x.strip() for x in metadata['Control']['X Amplitude'].split(' ', maxsplit=2)]
            physical_size_y, y_unit = [x.strip() for x in metadata['Control']['Y Amplitude'].split(' ', maxsplit=2)]
            z_amplitude, z_unit = [x.strip() for x in metadata['General Info']['Z Amplitude'].split(' ', maxsplit=2)]

            unit = mangle_length_unit_utf8(z_unit)
            physical_size_x = float(physical_size_x) * get_unit_conversion_factor(x_unit, unit)
            physical_size_y = float(physical_size_y) * get_unit_conversion_factor(y_unit, unit)

            # The height scale is defined in terms of the amplitude. For this we need to know the actual data range.
            # The data range is stored in the metadata, but only up to fixed precision. The code below may lead to
            # imprecise data conversion, but we should not open and scan through the whole file here.
            z_amplitude = float(z_amplitude)
            min_value = float(metadata['Miscellaneous']['Minimum'])
            max_value = float(metadata['Miscellaneous']['Maximum'])
            height_scale_factor = z_amplitude / (max_value - min_value)

            self._channels = [ChannelInfo(
                self,
                0,  # channel index
                name='Default',
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(physical_size_x, physical_size_y),
                uniform=True,
                unit=unit,
                height_scale_factor=height_scale_factor,
                info={
                    'acquisition_time': str(dateutil.parser.parse(metadata['General Info']['Acquisition time'])),
                    'raw_metadata': metadata
                },
                tags={
                    'file_offset': file_offset,
                    'z_amplitude': z_amplitude,
                    'dtype': dtype
                }
            )]

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
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

        channel = self.channels[channel_index]

        with OpenFromAny(self._file_path, 'rb') as f:
            nx, ny = channel.nb_grid_pts
            f.seek(channel.tags['file_offset'])
            dtype = channel.tags['dtype']
            height_data = np.frombuffer(f.read(nx * ny * dtype.itemsize), dtype=dtype).reshape((ny, nx)).T

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          channel.physical_sizes,
                          unit=channel.unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        print(height_data.min(), height_data.max())
        return topo.scale(channel.height_scale_factor)

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
