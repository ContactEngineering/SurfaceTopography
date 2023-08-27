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

from .binary import BinaryStructure, Convert, RawBuffer, Validate, DebugOutput
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, If, For, While, Skip, SizedChunk
from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from ..Support.UnitConversion import mangle_length_unit_utf8, get_unit_conversion_factor


XML = Convert(lambda x: None if x == '' else xmltodict.parse(x))

class OirChunkType(IntEnum):
    METADATA_LIST = 0  # Multiple metadata entries
    METADATA = 1  # Single metadata entry
    IMAGE = 2  # BMP image
    DATA_BLOCK = 3  # Data block
    # What is 4?
    SPACER = 5  # This chunk type has size 0 and contains no data


class OirMetadataBlockType(IntEnum):
    VERSION = 1
    PROPERTIES = 2  # Measurement and image properties; this is the main metadata block
    ANNOTATIONS = 3  # Annotation, have not yet seen a file where this is not empty
    OVERLAYS = 4  # Overlays, have not yet seen a file where this is not empty
    LOOKUP_TABLES = 5  # Tables for conversion of raw data to color
    PREFIX = 6  # Data prefix, typically "t001_0_1"
    DATASETS = 7  # List of datasets, each dataset has a UUID and is prefixed by above prefix
    UUIDS = 8  # List of UUIDs
    PREFIX2 = 9  # Prefix again?!?
    CAMERA = 10  # Camera information
    LOOKUP_TABLES2 = 11  # Lookup tables again?!?
    REF_CAMERA = 12  # Have not seen a file where this is not empty
    UUIDS2 = 13  # Another set of UUIDs
    EVENTS = 14  # Events, have not seen a file where this is not empty


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

# Common header in front of all metadata blocks
oir_metadata_header = BinaryStructure([
    ('aux_type1', 'I', Validate(lambda x, context: x in [1, 2], CorruptFile)),
    ('aux_type2', 'I'),
    ('aux_type3', 'I'),
    ('aux_type4', 'I'),  # Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
])

oir_block_1 = oir_block_2 = oir_block_3 = oir_block_4 = BinaryStructure([
    ('data', 'T', XML)
])

oir_block_5 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('text', 'T'),
            ('data', 'T', XML)
        ]),
        name='entries'
    )
])

oir_block_6 = oir_block_7 = oir_block_8 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('text', 'T'),
            (None, 'I')
        ]),
        name='entries'
    )
])

oir_block_9 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            (None, 'I'),
            ('text', 'T'),
            ('text', 'T'),
        ]),
        name='entries'
    )
])

oir_block_10 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('uuid', 'T'),
            (None, 'I'),
            ('xml1', 'T', XML),
            ('xml2', 'T', XML),
        ]),
        name='entries'
    )
])

oir_block_11 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('uuid', 'T'),
            ('data', 'T', XML),
        ]),
        name='entries'
    )
])

oir_block_12 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('uuid', 'T'),
            ('xml', 'T', XML),
        ]),
        name='entries'
    )
])

oir_block_13 = CompoundLayout([
    BinaryStructure([
        ('nb_entries', 'I')
    ]),
    For(
        lambda context: context.nb_entries,
        BinaryStructure([
            ('uuid', 'T'),
            (None, 'I'),
        ]),
        name='entries'
    )
])

oir_block_14 = CompoundLayout([
    BinaryStructure([
        ('xml', 'T', XML),
    ]),
])

oir_metadata_block = CompoundLayout([
    BinaryStructure([
        ('block_type', 'I', Convert(OirMetadataBlockType, CorruptFile)),
    ]),
    If(
        lambda context: context.block_type != 0,
        CompoundLayout([
            oir_metadata_header,
            If(
                lambda context: context.__parent__.block_type == 1,
                oir_block_1,
                lambda context: context.__parent__.block_type == 2,
                oir_block_2,
                lambda context: context.__parent__.block_type == 3,
                oir_block_3,
                lambda context: context.__parent__.block_type == 4,
                oir_block_4,
                lambda context: context.__parent__.block_type == 5,
                oir_block_5,
                lambda context: context.__parent__.block_type == 6,
                oir_block_6,
                lambda context: context.__parent__.block_type == 7,
                oir_block_7,
                lambda context: context.__parent__.block_type == 8,
                oir_block_8,
                lambda context: context.__parent__.block_type == 9,
                oir_block_9,
                lambda context: context.__parent__.block_type == 10,
                oir_block_10,
                lambda context: context.__parent__.block_type == 11,
                oir_block_11,
                lambda context: context.__parent__.block_type == 12,
                oir_block_12,
                lambda context: context.__parent__.block_type == 13,
                oir_block_13,
                lambda context: context.__parent__.block_type == 14,
                oir_block_14,
            )
        ])
    )
])

oir_chunk_type_bmp = CompoundLayout([
    BinaryStructure([
        (None, '3s', Validate('BMP', CorruptFile)),
        (None, 'b'),
        (None, '2s', Validate('BM', CorruptFile)),
        ('image_size', 'I')
    ]),
    Skip(lambda context: context.image_size - 6)
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

    _file_layout = CompoundLayout([
        oir_header,
        While(
            BinaryStructure([
                ('chunk_size', 'i'),
            ]),
            # Continue as long chunk size is not -1
            lambda context: context.chunk_size != -1,
            BinaryStructure([
                ('chunk_type', 'i', lambda data, context, **kwargs: OirChunkType(data))
            ]),
            # Continue as long as there is data in the chunk and we understand the chunk type
            lambda context: context.chunk_type in [OirChunkType.METADATA_LIST, OirChunkType.METADATA,
                                                   OirChunkType.IMAGE, OirChunkType.DATA_BLOCK, OirChunkType.SPACER],
            If(
                lambda context: context.chunk_type == OirChunkType.METADATA_LIST,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_metadata_block,
                    mode='loop',
                    name='items'
                ),
                lambda context: context.chunk_type == OirChunkType.METADATA,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_metadata_block
                ),
                lambda context: context.chunk_type == OirChunkType.IMAGE,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_chunk_type_bmp
                ),
                lambda context: context.chunk_type == OirChunkType.DATA_BLOCK,
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
            if chunk.chunk_size == -1:
                break
            if chunk.chunk_type == OirChunkType.DATA_BLOCK:
                data[chunk.header.uuid] = chunk.data
            elif chunk.chunk_type == OirChunkType.METADATA:
                metadata.update(chunk.data)
            elif chunk.chunk_type == OirChunkType.METADATA_LIST:
                for block in chunk.items:
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
