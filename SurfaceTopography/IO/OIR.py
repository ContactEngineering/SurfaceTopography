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

from enum import IntEnum

import dateutil.parser
import numpy as np
import xmltodict

from .binary import BinaryStructure, Convert, RawBuffer, Validate
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, If, For, While, Skip, SizedChunk
from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from ..Support.UnitConversion import mangle_length_unit_utf8, get_unit_conversion_factor


class OirChunkType(IntEnum):
    XML0 = 0
    XML = 1
    BMP = 2
    WTF = 3
    EOF = 5  # Is this really the end of file?


# For all fields with name `None` below, we do not know the meaning.

oir_header = BinaryStructure([
    ('magic', '16s', Validate('OLYMPUSRAWFORMAT', FileFormatMismatch)),
    (None, 'I', Validate(12, CorruptFile)),
    (None, 'I', Validate(0, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(2, CorruptFile)),
    ('file_size', 'I'),
    (None, 'I'),  # , Validate(2, CorruptFile)),
    ('some_size', 'I'),
    (None, 'I', Validate(0, CorruptFile)),
    (None, 'I', Validate(17, CorruptFile)),
    (None, 'I', Validate(0, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(0, CorruptFile)),
    (None, 'I'),
    (None, 'I', Validate(0, CorruptFile)),
    (None, '8s', Validate('UNKNOWN', FileFormatMismatch)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(0xFFFFFFFF, CorruptFile)),
    (None, 'I', Validate(0xFFFFFFFF, CorruptFile)),
], name='header')


def make_oir_metadata_header(name=None, valid_block_types=[2, 7, 8, 11]):
    return CompoundLayout([
        BinaryStructure([
            ('block_type', 'I', Validate(lambda x, context: x in valid_block_types, CorruptFile)),
            ('aux_type', 'I', Validate(lambda x, context: x in [1, 2], CorruptFile)),
            ('second_block_type', 'I', Validate(lambda x, context: x in [1, 2, 4, 7], CorruptFile)),
            # Validate(lambda x, context: x == 1 or (context.block_type in [2, 7, 8, 11] and x == 4), CorruptFile)),
            ('xml_dxx', 'I', Validate(lambda x, context: x in [0x0001, 0x0d48, 0x0dc6], CorruptFile)),
            (None, 'I', Validate(1, CorruptFile)),
            (None, 'I', Validate(1, CorruptFile)),
            (None, 'I', Validate(1, CorruptFile)),
            ('nb_entries', 'I')
        ]),
        If(
            lambda context: context.aux_type == 2,
            BinaryStructure([
                ('aux1', 'I'),
                ('aux2', 'I'),
            ])
        )
    ], name=name)


oir_block_7_8 = For(
    lambda context: context.nb_entries,
    BinaryStructure([
        ('uuid', 'T'),
        (None, 'I')
    ]),
    name='entries'
)

oir_block_11 = For(
    lambda context: context.nb_entries,
    BinaryStructure([
        ('uuid', 'T'),
        ('xml', 'T'),
    ]),
    name='entries'
)

oir_block_2_subitems = CompoundLayout([
    BinaryStructure([
        ('nb_subitems1', 'I'),
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
        (None, 'I', Validate(8, CorruptFile)),  # This may be an id
    ]),
    make_oir_metadata_header('header2'),
    BinaryStructure([
        ('nb_subitems2', 'I'),
    ]),
    For(
        lambda context: context.nb_subitems2,
        BinaryStructure([
            ('uuid', 'T'),
            (None, 'I')
        ]),
        name='subitems2'
    )
])

oir_block_2 = If(
    lambda context: context.xml_dxx in [0x0d48, 0x0dc6],
    BinaryStructure([
        ('data', 'T', Convert(xmltodict.parse))
    ]),
    CompoundLayout([
        BinaryStructure([
            ('nb_items', 'I'),
        ]),
        If(
            lambda context: context.nb_items == 1,
            CompoundLayout([
                BinaryStructure([
                    ('name', 'T'),  # This thing has a name, followed by nb_items
                ]),
                If(
                    lambda context: context.name == 'CAMERA',  # CAMERA appears to be special...
                    BinaryStructure([
                        ('nb_subitems1', 'I', Validate(1, CorruptFile)),
                        ('data1', 'T', Convert(xmltodict.parse)),
                        ('data2', 'T', Convert(xmltodict.parse))
                    ]),
                    lambda context: context.name == '',  # No name seems to be special...
                    BinaryStructure([
                        ('name1', 'T'),
                        ('name2', 'T')
                    ]),
                    oir_block_2_subitems,
                )
            ]),
            For(
                lambda context: context.nb_items,
                BinaryStructure([
                    ('uuid', 'T'),
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

oir_chunk_type_xml0 = CompoundLayout([
    While(
        BinaryStructure([
            ('id', 'I'),
        ]),
        If(
            # The id seems to be the only distinguishing feature of this block
            lambda context: context.id == 13,
            CompoundLayout([
                # This block reports type 2 but has a weird structure similar to type 7/8
                make_oir_metadata_header(valid_block_types=[2]),
                # We override prior nb_entries, which is 1
                BinaryStructure([
                    ('nb_entries', 'I'),
                ]),
                oir_block_7_8
            ]),
            oir_metadata_block,
        ),
        lambda context: context.id < 12,  # This appears to be a terminator
        name='blocks'
    ),
    BinaryStructure([
        ('block_type', 'I', Validate(14, CorruptFile)),
        ('aux_type', 'I', Validate(2, CorruptFile)),
        ('second_block_type', 'I', Validate(1, CorruptFile)),
        ('xml_dxx', 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(2943, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        ('nb_entries', 'I'),
        (None, 'I'),
        ('meta', 'T', Convert(xmltodict.parse))
    ]),
])

oir_chunk_type_xml = CompoundLayout([
    BinaryStructure([
        ('id', 'I')
    ]),
    oir_metadata_block
])

oir_chunk_type_bmp = CompoundLayout([
    BinaryStructure([
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        (None, 'I'),
        ('meta1', 'T', Convert(xmltodict.parse)),
        (None, 'I'),
        (None, 'I'),
        ('image_size', 'I'),  # Image size in bytes
        (None, 'I'),
        (None, '3s', Validate('BMP', CorruptFile)),
        (None, 'b')
    ]),
    Skip(lambda context: context.image_size),
    BinaryStructure([
        (None, 'I', Validate(0, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(2, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I'),  # Validate(lambda x, context: x == context.unknown3)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        (None, 'I', Validate(1, CorruptFile)),
        ('meta2', 'T', Convert(xmltodict.parse)),
    ]),
    oir_chunk_type_xml0,
])

oir_chunk_type_wtf = CompoundLayout([
    SizedChunk(
        lambda context: context.__parent__.chunk_size,
        BinaryStructure([
            (None, 'I', Validate(0, CorruptFile)),
            ('image_size', 'I'),  # Validate(935 * 1024, CorruptFile)),
            ('uuid', 'T')
        ], name='header'),
    ),
    BinaryStructure([
        ('image_size', 'I', Validate(lambda x, context: x == context.header.image_size, CorruptFile)),
        (None, 'I', Validate(4, CorruptFile))
    ]),
    RawBuffer('data', lambda context: context.image_size)
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
                ('chunk_size', 'I'),
                ('chunk_type', 'I', lambda name, data, context: OirChunkType(data))
            ]),
            # Continue as long as there is data in the chunk and we understand the chunk type
            lambda context: context.chunk_size > 0 and context.chunk_type in [OirChunkType.XML0, OirChunkType.XML,
                                                                              OirChunkType.BMP, OirChunkType.WTF],
            If(
                lambda context: context.chunk_type == OirChunkType.XML0,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_type_xml0
                ),
                lambda context: context.chunk_type == OirChunkType.XML,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_type_xml
                ),
                lambda context: context.chunk_type == OirChunkType.BMP,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_type_bmp
                ),
                lambda context: context.chunk_type == OirChunkType.WTF,
                oir_chunk_type_wtf
            ),
            name='chunks'
        ),
    ])

    def _validate_metadata(self):
        # We simplify the raw metadata extracted from the file
        metadata = {}
        data = {}
        for chunk in self._metadata.chunks:
            if chunk.chunk_type == OirChunkType.WTF:
                data[chunk.header.uuid] = chunk.data
            elif chunk.chunk_type == OirChunkType.XML:
                metadata.update(chunk.data)
            elif chunk.chunk_type == OirChunkType.XML0:
                for block in chunk.blocks:
                    if 'data' in block:
                        metadata.update(block.data)
                    elif 'items' in block:
                        uuids = {}
                        for item in block.items:
                            if item.uuid not in uuids:
                                uuids[item.uuid] = item.data
                            else:
                                uuids[item.uuid].update(item.data)
                        metadata.update(uuids)

        # Override metadata information
        self._data = data
        self._metadata = metadata

        # Further parsing of unit information
        image_properties = self.metadata['lsmimage:imageProperties']
        self._info = {
            'acquisition_time': str(dateutil.parser.parse(
                image_properties['commonimage:general']['base:creationDateTime'])),
            'instrument': {
                'name': image_properties['commonimage:microscope']['base:name']
            }
        }

        # Extract channel information
        channels = []
        image_info = image_properties['commonimage:imageInfo']
        raw_channels = image_info['commonimage:phase']['commonphase:group']['commonphase:channel']
        for raw_channel in raw_channels:
            image_definition = raw_channel['commonphase:imageDefinition']
            if image_definition['commonphase:imageType'] == 'HEIGHT':
                lengths = raw_channel['commonphase:length']
                units = raw_channel['commonphase:pixelUnit']

                xunit = mangle_length_unit_utf8(units['commonphase:x'])
                yunit = mangle_length_unit_utf8(units['commonphase:y'])
                zunit = mangle_length_unit_utf8(units['commonphase:z'])

                xfac = get_unit_conversion_factor(xunit, zunit)
                yfac = get_unit_conversion_factor(yunit, zunit)

                info = self._info.copy()
                info.update({'raw_metadata': raw_channel})

                uuid = raw_channel['@id']

                nb_grid_pts = (int(image_info['commonimage:width']), int(image_info['commonimage:height']))
                bit_counts = int(image_definition['commonphase:bitCounts'])
                dtype = np.dtype('<u2')
                if bit_counts != 16:
                    raise UnsupportedFormatFeature(f'Cannot read height data with {bit_counts} bits per pixel.')

                channels += [ChannelInfo(
                    self,
                    len(channels),  # channel index
                    name=uuid,
                    dim=2,
                    nb_grid_pts=nb_grid_pts,
                    physical_sizes=(xfac * float(lengths['commonparam:x']), yfac * float(lengths['commonparam:y'])),
                    uniform=True,
                    unit=zunit,
                    height_scale_factor=float(lengths['commonparam:z']),
                    info=info,
                    tags={
                        # I have no idea what prefix and suffix mean and whether they are fixed.
                        'reader': lambda stream_obj: np.frombuffer(
                            data[f't001_0_1_{uuid}_0'](stream_obj), dtype).reshape(nb_grid_pts)
                    }
                )]

        # Store channel information
        self._channels = channels

    @property
    def channels(self):
        return self._channels
