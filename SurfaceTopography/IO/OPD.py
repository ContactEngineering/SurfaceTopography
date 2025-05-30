#
# Copyright 2021-2023 Lars Pastewka
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

from struct import unpack

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase


def mask_undefined(data, maxval=1e32):
    """
    If data contains undefined points, then return a masked array with all
    undefined points masked.

    The following heuristics is applied to identify undefined points:
    - Remove points that are +/-inf or nan
    - Remove points that are >+maxval or <-maxval
    """
    # First, we mask all points that are infinite or nan
    mask = np.logical_and(np.isfinite(data), np.abs(data) < maxval)
    if mask.sum() < len(mask):
        return np.ma.masked_array(data, mask=np.logical_not(mask))
    else:
        return data


class OPDReader(ReaderBase):
    _format = 'opd'
    _mime_types = ['application/x-wyko-opd']
    _file_extensions = ['opd']

    _name = 'Wyko OPD'
    _description = '''
Files generated by the Vision software of the Bruker Wyko white-light
interferometer.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, fobj):
        self._fobj = fobj

        channel_index = 0
        self._channels = []

        with OpenFromAny(fobj, 'rb') as f:
            BLOCK_SIZE = 24

            def read_block(f):
                blkname = f.read(16).split(b'\0', 1)[0].decode('latin-1')
                blktype, blklen, blkattr = unpack('<hlH', f.read(8))
                return blkname, blktype, blklen, blkattr

            # Skip header
            f.read(2)

            # Read directory block
            dirname, dirtype, dirlen, dirattr = read_block(f)
            if dirname != 'Directory':
                raise IOError("Error reading directory block. "
                              "Header is '{}', expected 'Directory'".format(dirname))
            num_blocks = dirlen // BLOCK_SIZE
            if num_blocks * BLOCK_SIZE != dirlen:
                raise IOError('Directory length is not a multiple of the block physical_sizes.')

            blocks = []
            for i in range(num_blocks - 1):
                blocks += [read_block(f)]

            pixel_size = 1.0
            aspect = 1.0
            mult = 1.0
            for n, t, L, a in blocks:
                if L <= 0:
                    continue
                if n == 'RAW DATA' or n == 'RAW_DATA' or n == 'OPD' or n == 'Raw':
                    nx, ny, elsize = unpack('<HHH', f.read(6))
                    if elsize == 1:
                        dtype = np.dtype('c')
                    elif elsize == 2:
                        dtype = np.dtype('<i2')
                    elif elsize == 4:
                        dtype = np.dtype('f4')
                    else:
                        raise IOError("Don't know how to handle element of size {}."
                                      .format(elsize))

                    self._channels += [ChannelInfo(self, channel_index,
                                                   name=n,
                                                   dim=2,
                                                   nb_grid_pts=(nx, ny),
                                                   uniform=True,
                                                   unit='mm',
                                                   tags=dict(file_offset=f.tell(), dtype=dtype))]
                    channel_index += 1

                    # Skip over data buffer
                    f.seek(nx * ny * dtype.itemsize, 1)
                elif n == 'Wavelength':
                    wavelength, = unpack('<f', f.read(4))
                elif n == 'Mult':
                    mult, = unpack('<H', f.read(2))
                elif n == 'Aspect':
                    aspect, = unpack('<f', f.read(4))
                elif n == 'Pixel_size':
                    pixel_size, = unpack('<f', f.read(4))
                else:
                    f.read(L)

        # Loop over all channels and adjust physical sizes and height scale factor
        for channel in self._channels:
            nx, ny = channel.nb_grid_pts
            channel.physical_sizes = (nx * pixel_size, ny * pixel_size * aspect)
            channel.height_scale_factor = wavelength / mult * 1e-6

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

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        channel = self._channels[channel_index]

        nb_pixels = np.prod(channel.nb_grid_pts)
        dtype = channel.tags['dtype']
        with OpenFromAny(self._fobj, 'rb') as f:
            f.seek(channel.tags['file_offset'])
            rawdata = f.read(nb_pixels * dtype.itemsize)
        data = np.frombuffer(rawdata, count=nb_pixels, dtype=dtype)
        data = mask_undefined(data)
        data.shape = channel.nb_grid_pts

        # Height are in nm, width in mm
        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')
        topography = Topography(np.fliplr(data), channel.physical_sizes, unit='mm', info=info, periodic=periodic)
        return topography.scale(channel.height_scale_factor)

    @property
    def channels(self):
        return self._channels

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
