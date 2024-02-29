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

#
# Reference information and implementations:
# http://www.imagemet.com/WebHelp6/Default.htm#Reference_Guide/BCR_STM_File_Format.htm
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/bcrfile.c
#

import logging

import numpy as np

from .common import OpenFromAny
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor, is_length_unit

from .Reader import ReaderBase, ChannelInfo

_log = logging.getLogger(__file__)


class BCRReader(ReaderBase):
    _format = 'bcr'
    _mime_types = ['application/x-bcr-spm', 'application/x-bcrf-spm']
    _file_extensions = ['bcr', 'bcrf']

    _name = 'BCR-STM file format'
    _description = '''
BCR-STM and BCRF file formats
'''

    _MAGIC_BCRSTM = 'fileformat = bcrstm'
    _MAGIC_BCRF = 'fileformat = bcrf'
    _MAGIC_BCRSTM_UNICODE = 'fileformat = bcrstm_unicode'
    _MAGIC_BCRF_UNICODE = 'fileformat = bcrf_unicode'  # we currently only have an example for this type

    _MIN_HEADER_SIZE = 2048

    def __init__(self, file_path):
        """
        Load NanoSurf easyScan data files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path

        # The start of the file is textual with metadata; we need to parse it
        with OpenFromAny(self._file_path, 'rb') as fobj:
            buffer = fobj.read(self._MIN_HEADER_SIZE)  # Header is either 2048 or 4096 bytes

            # Check what type of file we are dealing with
            if buffer.decode('latin-1').startswith(self._MAGIC_BCRSTM):
                self._file_type = 'bcrstm'
                self._encoding = 'latin-1'
                data_type = 'i2'  # int16
            elif buffer.decode('latin-1').startswith(self._MAGIC_BCRF):
                self._file_type = 'bcrf'
                self._encoding = 'latin-1'
                data_type = 'f4'  # single precision flaot
            elif buffer.decode('utf-16').startswith(self._MAGIC_BCRSTM):
                self._file_type = 'bcrstm'
                self._encoding = 'utf-16'
                data_type = 'i2'  # int16
            elif buffer.decode('utf-16').startswith(self._MAGIC_BCRF):
                self._file_type = 'bcrf'
                self._encoding = 'utf-16'
                data_type = 'f4'  # single precision flaot
            else:
                raise FileFormatMismatch('This is not a BCR-STM/BCRF data file')

            # Find out how long the header is
            buffer_str = buffer.decode(self._encoding)
            line, buffer_str = buffer_str.split('\n', 1)
            line = line.strip()
            self._headersize = None
            eof = False
            while not eof:
                if not line.startswith('%'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'headersize':
                        self._headersize = int(value)
                        break
                try:
                    line, buffer_str = buffer_str.split('\n', 1)
                    line = line.strip()
                except ValueError:
                    eof = True
            if self._headersize is None:
                raise CorruptFile("Could not find 'headersize' key in file metadata")

            # Header size is given in number of characters
            if self._encoding == 'utf-16':
                self._headersize *= 2

            # We can now read and parse the full header
            if self._headersize > self._MIN_HEADER_SIZE:
                # Read rest of header
                buffer += fobj.read(self._headersize - self._MIN_HEADER_SIZE)
            buffer_str = buffer.decode(self._encoding)
            line, buffer_str = buffer_str.split('\n', 1)
            line = line.strip()
            self._metadata = {}
            eof = False
            while not eof:
                if not line.startswith('%') and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        self._metadata[key] = value
                    except ValueError:
                        _log.warning(f"Skipping line '{line}' because it does not appear to be a key/value pair.")
                try:
                    line, buffer_str = buffer_str.split('\n', 1)
                    line = line.strip()
                except ValueError:
                    eof = True

        # Data layout
        little_endian = int(self._metadata['intelmode'])
        endian_str = '<' if little_endian else '>'
        self._dtype = np.dtype(f'{endian_str}{data_type}')

        xunit = self._metadata['xunit']
        assert xunit == self._metadata['yunit']
        zunit = self._metadata['zunit']

        if not is_length_unit(zunit):
            raise UnsupportedFormatFeature(f"This BCR/BCRF file reports data in units of '{zunit}'. This is not "
                                           f"a height unit as expected for topography data.")

        self._channels = [
            ChannelInfo(self,
                        0,  # channel index
                        name='Default',
                        dim=2,
                        nb_grid_pts=(int(self._metadata['xpixels']), int(self._metadata['ypixels'])),
                        physical_sizes=(float(self._metadata['xlength']), float(self._metadata['ylength'])),
                        uniform=True,
                        unit=xunit,
                        height_scale_factor=float(self._metadata['bit2nm']) * get_unit_conversion_factor(zunit, xunit),
                        info={
                            'raw_metadata': self._metadata
                        })
        ]

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):

        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self._channels[channel_index]
        with OpenFromAny(self._file_path, 'rb') as fobj:
            sx, sy = self._check_physical_sizes(physical_sizes,
                                                channel.physical_sizes)

            nx, ny = channel.nb_grid_pts

            fobj.seek(self._headersize)
            data = np.frombuffer(fobj.read(nx * ny * self._dtype.itemsize), dtype=self._dtype).reshape(ny, nx).T

        # internal information from file
        _info = channel.info.copy()
        _info.update(info)

        # it is not allowed to provide extra `physical_sizes` here:
        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        # the orientation of the heights is modified in order to match
        # the image of gwyddion when plotted with imshow(t.heights().T)
        # or pcolormesh(t.heights().T) for origin in lower left and
        # with inverted y axis (cartesian coordinate system)

        invalid_pixel_value = float(self._metadata['voidpixels'])
        topography = Topography(
            np.ma.masked_array(data, mask=data == invalid_pixel_value),
            physical_sizes=(sx, sy),
            unit=channel.unit,
            info=_info,
            periodic=periodic)
        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
