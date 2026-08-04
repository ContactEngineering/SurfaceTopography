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

from ..Exceptions import CorruptFile, FileFormatMismatch
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, Cond, DictExpr, F, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase, For, Seek

_MAGIC = b'AliconaImaging\x00\r\n'

# The AL3D header is a tag list; this maps it to a key/value dictionary
_tags = F.to_map(C.tags, "key", "value")

_nx = F.int(_tags.Cols)
_ny = F.int(_tags.Rows)

# Rows of the depth image are padded to multiples of 8 bytes
_row_elements = (_nx * 4 + 7) // 8 * 8 // 4

# Invalid pixels are marked by NaN or by a marker value from the header.
# The tolerance must be scaled with the magnitude of the marker; without
# abs() a negative marker would make the mask unconditionally false.
_INVALID_RELTOL = 1.5e-7
_invalid = F.float(_tags.InvalidPixelValue)
_invalid_mask = F.isnan(V) | Cond(
    F.isnan(_invalid),
    False,
    F.abs(V - _invalid) < _INVALID_RELTOL * F.abs(_invalid),
)


class AL3DReader(DeclarativeReaderBase):
    _format = 'al3d'
    _mime_types = ['application/x-alicona-imaging-al3d']
    _file_extensions = ['al3d']

    _name = 'Alicona Imaging AL3D'
    _description = '''
AL3D format of Alicona Imaging.
'''

    _magic = [(0, _MAGIC)]

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
            ('value', '30s', F.int(V)),
            ('crlf', '2s', Validate('\r\n', CorruptFile)),
        ], byte_order='<', name='version_tag'),
        BinaryStructure([
            ('key', '20s', Validate('TagCount', CorruptFile)),
            ('value', '30s', F.int(V)),
            ('crlf', '2s', Validate('\r\n', CorruptFile)),
        ], byte_order='<', name='tag_count_tag'),
        For(C.tag_count_tag.value, _tag_structure, name='tags'),
        # The depth image sits at an absolute offset given in the tag list
        Seek(F.int(_tags.DepthImageOffset), comment='depth image'),
        BinaryArray(
            'data',
            Tup(_ny, _row_elements),
            F.dtype('<f4'),
            conversion_fun=F.transpose(V[:, :_nx]),  # Crop row padding
            mask_fun=_invalid_mask,
        ),
    ])

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(_nx, _ny),
            "physical_sizes": Tup(
                _nx * F.float(_tags.PixelSizeXMeter),
                _ny * F.float(_tags.PixelSizeYMeter),
            ),
            "uniform": True,
            "unit": "m",
            "height_scale_factor": 1,
            "info": {
                "instrument": {
                    "vendor": "Alicona Imaging",
                    "software": F.get(_tags, "CreatingApplication", None),
                },
                "raw_metadata": F.merge(
                    DictExpr(
                        {
                            "Version": C.version_tag.value,
                            "TagCount": C.tag_count_tag.value,
                        }
                    ),
                    _tags,
                ),
            },
            "data": C.data,
        }
    ]
