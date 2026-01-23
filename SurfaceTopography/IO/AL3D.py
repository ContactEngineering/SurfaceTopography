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
from numpy.lib.stride_tricks import as_strided

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryStructure, Convert, Validate
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, For, MagicMatch


class AL3DReader(DeclarativeReaderBase):
    _format = 'al3d'
    _mime_types = ['application/x-alicona-imaging-al3d']
    _file_extensions = ['al3d']

    _name = 'Alicona Imaging AL3D'
    _description = '''
AL3D format of Alicona Imaging.
'''

    _MAGIC = b'AliconaImaging\x00\r\n'

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC):
            return MagicMatch.YES
        return MagicMatch.NO

    _tag_structure = BinaryStructure([
        ('key', '20s'),
        ('value', '30s'),
        ('crlf', '2s', Validate('\r\n', CorruptFile)),
    ], byte_order='<')

    _file_layout = CompoundLayout([
        BinaryStructure([
            ('magic', f'{len(_MAGIC)}s',
             Validate(_MAGIC.decode('latin1').strip('\x00').strip(' '),
                      FileFormatMismatch)),
        ], byte_order='<', name='header'),
        BinaryStructure([
            ('key', '20s', Validate('Version', CorruptFile)),
            ('value', '30s', Convert(lambda x: int(x))),
            ('crlf', '2s', Validate('\r\n', CorruptFile)),
        ], byte_order='<', name='version_tag'),
        BinaryStructure([
            ('key', '20s', Validate('TagCount', CorruptFile)),
            ('value', '30s', Convert(lambda x: int(x))),
            ('crlf', '2s', Validate('\r\n', CorruptFile)),
        ], byte_order='<', name='tag_count_tag'),
        For(lambda ctx: ctx.tag_count_tag.value, _tag_structure, name='tags')
    ])

    # Relative tolerance for catching invalid pixels
    _INVALID_RELTOL = 1.5e-7

    def _validate_metadata(self):
        self._header = {
            'Version': self._metadata.version_tag.value,
            'TagCount': self._metadata.tag_count_tag.value
        }
        for tag in self._metadata.tags:
            self._header[tag.key] = tag.value

    def read_height_data(self, f):
        f.seek(int(self._header['DepthImageOffset']))
        invalid_pixel_value = float(self._header['InvalidPixelValue'])
        dtype = np.single
        nx, ny = int(self._header['Cols']), int(self._header['Rows'])
        rowstride = (nx * np.dtype(dtype).itemsize + 7) // 8 * 8
        buffer = f.read(rowstride * ny * np.dtype(dtype).itemsize)
        data = as_strided(np.frombuffer(buffer, dtype=dtype), shape=(ny, nx),
                          strides=(rowstride, np.dtype(dtype).itemsize))
        mask = np.isnan(data)
        if not np.isnan(invalid_pixel_value):
            mask = np.logical_or(mask, np.abs(data - invalid_pixel_value) < self._INVALID_RELTOL * invalid_pixel_value)
        return np.ma.masked_array(data.T, mask=mask.T)

    @property
    def channels(self):
        nx, ny = int(self._header['Cols']), int(self._header['Rows'])
        instrument_info = {"vendor": "Alicona Imaging"}
        if "CreatingApplication" in self._header:
            instrument_info["software"] = self._header["CreatingApplication"]

        return [
            ChannelInfo(
                self,
                0,  # channel index
                name="Default",
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(
                    nx * float(self._header["PixelSizeXMeter"]),
                    ny * float(self._header["PixelSizeYMeter"]),
                ),
                uniform=True,
                unit="m",
                height_scale_factor=1,
                info={"instrument": instrument_info, "raw_metadata": self._header},
                tags={"reader": self.read_height_data},
            )
        ]
