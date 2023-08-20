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

#
# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/oirfile.c
# https://github.com/ome/bioformats/blob/develop/components/formats-gpl/src/loci/formats/in/OIRReader.java
#

import datetime

import numpy as np
import xmltodict

from .binary import BinaryArray, BinaryStructure, Validate, DebugOutput, Convert
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, If, For, While, Skip, SizedChunk
from ..Exceptions import CorruptFile, FileFormatMismatch

OIR_CHUNK_XML0 = 0
OIR_CHUNK_XML = 1
OIR_CHUNK_BMP = 2
OIR_CHUNK_WTF = 3

oir_header = BinaryStructure([
    ('magic', '16s', Validate('OLYMPUSRAWFORMAT', FileFormatMismatch)),
    ('unknown1', 'I', Validate(12, CorruptFile)),
    ('unknown2', 'I', Validate(0, CorruptFile)),
    ('unknown3', 'I', Validate(1, CorruptFile)),
    ('unknown4', 'I', Validate(2, CorruptFile)),
    ('file_size', 'I'),
    ('unknown5', 'I'),  # , Validate(2, CorruptFile)),
    ('some_size', 'I'),
    ('unknown6', 'I', Validate(0, CorruptFile)),
    ('unknown7', 'I', Validate(17, CorruptFile)),
    ('unknown8', 'I', Validate(0, CorruptFile)),
    ('unknown9', 'I', Validate(1, CorruptFile)),
    ('unknown10', 'I', Validate(0, CorruptFile)),
    ('unknown11', 'I'),
    ('unknown12', 'I', Validate(0, CorruptFile)),
    ('unknown_str', '8s', Validate('UNKNOWN', FileFormatMismatch)),
    ('unknown13', 'I', Validate(1, CorruptFile)),
    ('unknown14', 'I', Validate(1, CorruptFile)),
    ('unknown15', 'I', Validate(0xFFFFFFFF, CorruptFile)),
    ('unknown16', 'I', Validate(0xFFFFFFFF, CorruptFile)),
], name='header')


def make_oir_metadata_header(name=None, valid_block_types=[2, 7, 8, 11]):
    return CompoundLayout([
        BinaryStructure([
            ('block_type', 'I', DebugOutput(), Validate(lambda x, context: x in valid_block_types, CorruptFile)),
            ('aux_type', 'I', DebugOutput(), Validate(lambda x, context: x in [1, 2], CorruptFile)),
            ('second_block_type', 'I', DebugOutput(), Validate(lambda x, context: x in [1, 2, 4, 7], CorruptFile)),
            # Validate(lambda x, context: x == 1 or (context.block_type in [2, 7, 8, 11] and x == 4), CorruptFile)),
            ('xml_dxx', 'I', Validate(lambda x, context: x in [0x0001, 0x0d48, 0x0dc6], CorruptFile), DebugOutput()),
            ('unknown16', 'I', Validate(1, CorruptFile)),
            ('unknown17', 'I', Validate(1, CorruptFile)),
            ('unknown18', 'I', Validate(1, CorruptFile)),
            ('nb_entries', 'I', DebugOutput())
        ]),
        If(
            lambda context: context.aux_type == 2,
            BinaryStructure([
                ('aux1', 'I', DebugOutput()),
                ('aux2', 'I', DebugOutput()),
            ])
        )
    ], name=name)


oir_block_7_8 = For(
    lambda context: context.nb_entries,
    BinaryStructure([
        ('uuid', 'T', DebugOutput()),
        ('unknown12_1', 'I', DebugOutput())
    ]),
    name='entries'
)

oir_block_11 = For(
    lambda context: context.nb_entries,
    BinaryStructure([
        ('uuid', 'T', DebugOutput()),
        ('xml', 'T'),
    ]),
    name='entries'
)

oir_block_2_subitems = CompoundLayout([
    BinaryStructure([
        ('nb_subitems1', 'I', DebugOutput()),
    ]),
    make_oir_metadata_header(name='header1', valid_block_types=[7, 8, 11]),
    For(
        lambda context: context.nb_subitems1,
        If(
            lambda context: context.block_type == 7 or context.block_type == 8,
            oir_block_7_8,
            lambda context: context.block_type == 11,  # Never seen this
            oir_block_11,
            context_mapper=lambda context: context.header1
        ), name='subitems1'
    ),
    BinaryStructure([
        ('unknown', 'I', Validate(8, CorruptFile)),  # This may be an id
    ]),
    make_oir_metadata_header('header2'),
    BinaryStructure([
        ('nb_subitems2', 'I', DebugOutput()),
    ]),
    For(
        lambda context: context.nb_subitems2,
        BinaryStructure([
            ('uuid', 'T', DebugOutput()),
            ('unknown12_2', 'I', DebugOutput())
        ]),
        name='subitems2'
    )
])

oir_block_2 = If(
    lambda context: context.xml_dxx in [0x0d48, 0x0dc6],
    BinaryStructure([
        ('data', 'T', Convert(xmltodict.parse), DebugOutput())
    ]),
    CompoundLayout([
        BinaryStructure([
            ('nb_items', 'I', DebugOutput()),
        ]),
        If(
            lambda context: context.nb_items == 1,
            CompoundLayout([
                BinaryStructure([
                    ('name', 'T', DebugOutput()),  # This thing has a name, followed by nb_items
                ]),
                If(
                    lambda context: context.name == 'CAMERA',  # CAMERA appears to be special...
                    BinaryStructure([
                        ('nb_subitems1', 'I', Validate(1, CorruptFile)),
                        ('data1', 'T', Convert(xmltodict.parse), DebugOutput()),
                        ('data2', 'T', Convert(xmltodict.parse), DebugOutput())
                    ]),
                    lambda context: context.name == '',  # No name seems to be special,
                    BinaryStructure([
                        ('name', 'T', DebugOutput()),
                        ('name2', 'T', DebugOutput())
                    ]),
                    oir_block_2_subitems,
                )
            ]),
            For(
                lambda context: context.nb_items,
                BinaryStructure([
                    ('uuid', 'T', DebugOutput()),
                    ('data', 'T', Convert(xmltodict.parse))
                ]),
                name='items'
            )
        )
    ])
)

oir_metadata_block = CompoundLayout([
    make_oir_metadata_header(),
    If(
        lambda context: context.block_type == 2,
        oir_block_2,
        lambda context: context.block_type == 7 or context.block_type == 8,
        oir_block_7_8,
        lambda context: context.block_type == 11,
        oir_block_11,
    )
])

oir_chunk_xml0 = CompoundLayout([
    While(
        BinaryStructure([
            ('id', 'I', DebugOutput()),
        ]),
        If(
            # The id seems to be the only distinguishing feature of this block
            lambda context: context.id == 13,
            CompoundLayout([
                # This block reports type 2 but has a weird structure similar to type 7/8
                make_oir_metadata_header(valid_block_types=[2]),
                # We override prior nb_entries, which is 1
                BinaryStructure([
                    ('nb_entries', 'I', DebugOutput()),
                ]),
                oir_block_7_8
            ]),
            oir_metadata_block,
        ),
        lambda context: context.id < 12,  # This appears to be a terminator
        name='blocks'
    ),
    BinaryStructure([
        ('block_type', 'I', DebugOutput(), Validate(14, CorruptFile)),
        ('aux_type', 'I', DebugOutput(), Validate(2, CorruptFile)),
        ('second_block_type', 'I', DebugOutput(), Validate(1, CorruptFile)),
        ('xml_dxx', 'I', Validate(1, CorruptFile), DebugOutput()),
        ('unknown16', 'I', Validate(2943, CorruptFile)),
        ('unknown17', 'I', Validate(1, CorruptFile)),
        ('unknown18', 'I', Validate(1, CorruptFile)),
        ('nb_entries', 'I', DebugOutput()),
        ('aux1', 'I', DebugOutput()),
        ('meta', 'T', Convert(xmltodict.parse), DebugOutput())
    ]),
])

oir_chunk_xml = CompoundLayout([
    BinaryStructure([
        ('id', 'I', DebugOutput())
    ]),
    oir_metadata_block
])

oir_chunk_bmp = CompoundLayout([
    BinaryStructure([
        ('unknown1', 'I', DebugOutput()),
        ('unknown2', 'I', DebugOutput()),
        ('unknown3', 'I', DebugOutput()),
        ('unknown4', 'I', DebugOutput()),
        ('unknown5', 'I', DebugOutput()),
        ('unknown6', 'I', DebugOutput()),
        ('unknown7', 'I', DebugOutput()),
        ('xml', 'T', Convert(xmltodict.parse), DebugOutput()),
        ('unknown8', 'I', DebugOutput()),
        ('unknown9', 'I', DebugOutput()),
        ('image_size', 'I', DebugOutput()),  # Image size in bytes
        ('unknown11', 'I', DebugOutput()),
        ('image_type', '3s', Validate('BMP', CorruptFile)),
        (None, 'b')
    ]),
    Skip(lambda context: context.image_size),
    BinaryStructure([
        ('unknown12', 'I', Validate(0, CorruptFile)),
        ('unknown13', 'I', Validate(1, CorruptFile)),
        ('unknown14', 'I', Validate(2, CorruptFile)),
        ('unknown15', 'I', Validate(1, CorruptFile)),
        ('unknown16', 'I', Validate(1, CorruptFile)),
        ('unknown17', 'I'),  # Validate(lambda x, context: x == context.unknown3)),
        ('unknown18', 'I', Validate(1, CorruptFile)),
        ('unknown19', 'I', Validate(1, CorruptFile)),
        ('unknown20', 'I', Validate(1, CorruptFile)),
        ('unknown21', 'I', Validate(1, CorruptFile)),
        ('meta', 'T', Convert(xmltodict.parse), DebugOutput()),
    ]),
    oir_chunk_xml0,
])

oir_chunk_wtf = CompoundLayout([
    SizedChunk(
        lambda context: context.__parent__.chunk_size,
        BinaryStructure([
            ('unknown2', 'I', Validate(0, CorruptFile)),
            ('image_size', 'I', DebugOutput()),  # Validate(935 * 1024, CorruptFile)),
            ('uuid', 'T', DebugOutput())
        ], name='header'),
    ),
    BinaryStructure([
        ('image_size', 'I', Validate(lambda x, context: x == context.header.image_size, CorruptFile)),
        ('unknown3', 'I', Validate(4, CorruptFile))
    ]),
    BinaryArray('data', lambda context: context.image_size, np.dtype('b'))
])


class OIRReader(DeclarativeReaderBase):
    _format = 'oir'
    _mime_types = ['application/x-olympus-oir']
    _file_extensions = ['oir']

    _name = 'Olympus OIR'
    _description = '''
This reader imports Olympus OIR data files.
'''
    _debug = True

    _file_layout = CompoundLayout([
        oir_header,
        While(
            BinaryStructure([
                ('chunk_size', 'I', DebugOutput()),
                ('chunk_type', 'I', DebugOutput())
            ]),
            # Continue as long as we understand the chunk type
            lambda context: context.chunk_size > 0 and
                            context.chunk_type in [OIR_CHUNK_XML0, OIR_CHUNK_XML, OIR_CHUNK_BMP, OIR_CHUNK_WTF],
            If(
                lambda context: context.chunk_type == OIR_CHUNK_XML0,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_xml0
                ),
                lambda context: context.chunk_type == OIR_CHUNK_XML,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_xml
                ),
                lambda context: context.chunk_type == OIR_CHUNK_BMP,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_bmp
                ),
                lambda context: context.chunk_type == OIR_CHUNK_WTF,
                oir_chunk_wtf
            ),
            name='chunks'
        ),
    ])

    @property
    def channels(self):
        import json
        # print(json.dumps(self._metadata.header, indent=4))

        # for chunk in self._metadata.chunks:
        #    print(chunk)

        header = self._metadata.header

        return [ChannelInfo(
            self,
            0,  # channel index
            name='Default',
            dim=2,
            nb_grid_pts=(header.nb_grid_pts_x, header.nb_grid_pts_y),
            physical_sizes=(header.grid_spacing_x * header.nb_grid_pts_x,
                            header.grid_spacing_y * header.nb_grid_pts_y),
            height_scale_factor=header.height_scale_factor,
            uniform=True,
            unit=header.unit_x,
            info=info,
            tags={'reader': self._metadata.data}
        )]
