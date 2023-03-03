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

import numpy as np
import olefile
import xml.etree.ElementTree as ElementTree

from .common import OpenFromAny
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor

from .Reader import ReaderBase, ChannelInfo



def xml_to_dict(xml):
    ElementTree.fromstring(xml)


class MNTReader(ReaderBase):
    _format = 'mnt'
    _name = 'Mountains file format'
    _description = '''
File format of the Mountains software
'''

    def __init__(self, file_path):
        """
        Load Mountains data files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path

        f = olefile.OleFileIO(self._file_path)
        s = f.openstream('XmlHeader')
        s.read(6)  # Skip the first 6 bytes, not sure what is in there
        xml_header = s.read().decode('utf-16')
        header_metadata = {}
        xmlroot = ElementTree.fromstring(xml_header)
        for child in xmlroot:
            header_metadata[child.tag] = child.text
        print(header_metadata)

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
            data = np.fromfile(fobj, count=nx * ny, dtype=self._dtype).reshape(ny, nx).T

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
