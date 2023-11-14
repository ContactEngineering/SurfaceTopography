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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/wsxmfile.c
#

from enum import IntEnum
from zipfile import ZipFile

import dateutil.parser
import numpy as np
import xmltodict

from .binary import BinaryStructure, Convert, RawBuffer, Validate
from .common import OpenFromAny
from .Reader import ChannelInfo, CompoundLayout, ReaderBase, DeclarativeReaderBase, If, For, While, Skip, SizedChunk
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..Support.UnitConversion import mangle_length_unit_utf8, get_unit_conversion_factor
from ..UniformLineScanAndTopography import Topography

XML = Convert(xmltodict.parse)


class OirChunkType(IntEnum):
    METADATA_LIST = 0  # Multiple metadata entries
    METADATA = 1  # Single metadata entry
    IMAGE = 2  # BMP image
    DATA_BLOCK = 3  # Data block
    # What is 4?
    SPACER = 5  # This chunk type has size 0 and contains no data


class OirMetadataBlockType(IntEnum):
    VERSION = 1  # Version information
    PROPERTIES = 2  # Measurement and image properties; this is the main metadata block
    ANNOTATIONS = 3  # Annotation, have not yet seen a file where this is not empty
    OVERLAYS = 4  # Overlays, have not yet seen a file where this is not empty
    LOOKUP_TABLES = 5  # Tables for conversion of raw data to color
    TOPOGRAPHY_PREFIX = 6  # Topography prefix, typically "t001_0_1"
    DATASETS = 7  # List of datasets, each dataset has a UUID and is prefixed by above prefix
    TOPOGRAPHY_UUIDS = 8  # List of UUIDs containing topography data
    TOPOGRAPHY_PREFIX_AGAIN = 9  # Prefix again, but twice?!?
    CAMERA = 10  # Camera information
    LOOKUP_TABLES2 = 11  # Lookup tables again?!?
    CAMERA_PREFIX = 12  # Camera prefix, typically "REF_CAMERA0"
    CAMERA_UUIDS = 13  # List of UUIDs containing camera data
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

# Common header in front of all metadata blocks - I have no idea what this information means
oir_metadata_header = BinaryStructure([
    (None, 'I', Validate(lambda x, context: x in [1, 2], CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I'),
    (None, 'I'),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
    (None, 'I', Validate(1, CorruptFile)),
])

oir_simple_xml_block = BinaryStructure([
    ('data', 'T', XML)
])

oir_metadata_blocks = {
    OirMetadataBlockType.VERSION: oir_simple_xml_block,  # Block type 1
    OirMetadataBlockType.PROPERTIES: oir_simple_xml_block,  # Block type 2
    OirMetadataBlockType.ANNOTATIONS: oir_simple_xml_block,  # Block type 3
    OirMetadataBlockType.OVERLAYS: oir_simple_xml_block,  # Block type 4
    OirMetadataBlockType.LOOKUP_TABLES: CompoundLayout([  # Block type 5
        BinaryStructure([
            ('nb_entries', 'I')
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('text', 'T'),
                ('data', 'T', XML)
            ]),
            name='data'
        )
    ]),
    OirMetadataBlockType.TOPOGRAPHY_PREFIX: CompoundLayout([  # Block type 6
        BinaryStructure([
            ('nb_entries', 'I', Validate(1, CorruptFile))
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('prefix', 'T'),
                (None, 'I')
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': context.entries[0].prefix}),
    OirMetadataBlockType.DATASETS: CompoundLayout([  # Block type 7
        BinaryStructure([
            ('nb_entries', 'I')
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('prefix', 'T'),
                (None, 'I')
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': [entry.prefix for entry in context.entries]}),
    OirMetadataBlockType.TOPOGRAPHY_UUIDS: CompoundLayout([  # Block type 8
        BinaryStructure([
            ('nb_entries', 'I')
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('prefix', 'T'),
                (None, 'I')
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': [entry.prefix for entry in context.entries]}),
    OirMetadataBlockType.TOPOGRAPHY_PREFIX_AGAIN: CompoundLayout([  # Block type 9
        BinaryStructure([
            ('nb_entries', 'I')
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                (None, 'I'),
                ('prefix1', 'T'),
                ('prefix2', 'T'),
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': context.entries}),
    OirMetadataBlockType.CAMERA: CompoundLayout([  # Block type 10
        BinaryStructure([
            ('nb_entries', 'I')
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('uuid', 'T'),
                (None, 'I'),
                ('camera_data', 'T', XML),
                ('image_data', 'T', XML),
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': {entry.uuid: entry for entry in context.entries}}),
    OirMetadataBlockType.LOOKUP_TABLES2: CompoundLayout([  # Block type 11
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
    ], context_mapper=lambda context: {'data': {entry.uuid: entry.data for entry in context.entries}}),
    OirMetadataBlockType.CAMERA_PREFIX: CompoundLayout([  # Block type 12
        BinaryStructure([
            ('nb_entries', 'I', Validate(1, CorruptFile))
        ]),
        For(
            lambda context: context.nb_entries,
            BinaryStructure([
                ('prefix', 'T', Validate("REF_CAMERA0", CorruptFile)),
                (None, 'I'),
            ]),
            name='entries'
        )
    ], context_mapper=lambda context: {'data': context.entries[0].prefix}),
    OirMetadataBlockType.CAMERA_UUIDS: CompoundLayout([  # Block type 13
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
    ], context_mapper=lambda context: {'data': [entry.uuid for entry in context.entries]}),
    OirMetadataBlockType.EVENTS: CompoundLayout([  # Block type 14
        BinaryStructure([
            ('data', 'T', XML),
        ]),
    ])
}

oir_metadata_block = CompoundLayout([
    BinaryStructure([
        ('block_type', 'I', Convert(OirMetadataBlockType, CorruptFile)),
    ]),
    CompoundLayout([
        oir_metadata_header,
        lambda context: oir_metadata_blocks[context.__parent__.block_type]
    ])
], context_mapper=lambda context: {context.block_type: context.data})

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
                    context_mapper=lambda context: {list(item.keys())[0]: list(item.values())[0] for item in context}
                ),
                lambda context: context.chunk_type == OirChunkType.METADATA,
                SizedChunk(
                    lambda context: context.chunk_size,
                    oir_metadata_block,
                    context_mapper=lambda context: list(context.values())[0]
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
        for i, chunk in enumerate(self._metadata.chunks):
            if chunk.chunk_size == -1:
                break
            if chunk.chunk_type == OirChunkType.DATA_BLOCK:
                data[chunk.header.uuid] = chunk.data
            elif chunk.chunk_type == OirChunkType.METADATA:
                pass  # Ignore this, those are per-frame properties. I have not yet seen files with multiple frames.
            elif chunk.chunk_type == OirChunkType.METADATA_LIST:
                # This entry occurs multiple times in the files that I have seen, but typically with identical data.
                metadata.update(chunk)

        # Override metadata information
        self._data = data
        self._metadata = metadata

        prefix = self.metadata[OirMetadataBlockType.TOPOGRAPHY_PREFIX]

        # Further parsing of unit information
        properties = self.metadata[OirMetadataBlockType.PROPERTIES]
        if 'lsmimage:imageProperties' in properties:
            # This is a laser scanning image
            image_properties = properties['lsmimage:imageProperties']
        else:
            # There is a 'cameraimage' entry in some files, but that appears to be raw camera data
            raise UnsupportedFormatFeature("OIR file does not contain neither 'lsmimage' entry. I do not know how to "
                                           "extract height information.")

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
                uuid = raw_channel['@id']

                if uuid not in self._metadata[OirMetadataBlockType.TOPOGRAPHY_UUIDS]:
                    raise CorruptFile(f'UUID {uuid} not in topography UUID block, which contains: '
                                      f'{self._metadata[OirMetadataBlockType.TOPOGRAPHY_UUIDS]}')

                lengths = raw_channel['commonphase:length']
                units = raw_channel['commonphase:pixelUnit']

                xunit = mangle_length_unit_utf8(units['commonphase:x'])
                yunit = mangle_length_unit_utf8(units['commonphase:y'])
                zunit = mangle_length_unit_utf8(units['commonphase:z'])

                xfac = get_unit_conversion_factor(xunit, zunit)
                yfac = get_unit_conversion_factor(yunit, zunit)

                info = self._info.copy()
                info.update({'raw_metadata': raw_channel})

                nx = int(image_info['commonimage:width'])
                ny = int(image_info['commonimage:height'])
                bit_counts = int(image_definition['commonphase:bitCounts'])
                dtype = np.dtype('<u2')
                if bit_counts != 16:
                    raise UnsupportedFormatFeature(f'Cannot read height data with {bit_counts} bits per pixel.')

                channels += [ChannelInfo(
                    self,
                    len(channels),  # channel index
                    name=uuid,
                    dim=2,
                    nb_grid_pts=(nx, ny),
                    physical_sizes=(xfac * float(lengths['commonparam:x']), yfac * float(lengths['commonparam:y'])),
                    uniform=True,
                    unit=zunit,
                    height_scale_factor=float(lengths['commonparam:z']),
                    info=info,
                    tags={
                        # The suffix _0 is probably the frame number, but I have never seen files with multiple frames.
                        'reader': lambda stream_obj: np.frombuffer(
                            data[f'{prefix}_{uuid}_0'](stream_obj), dtype).reshape((ny, nx)).T
                    }
                )]

        # Store channel information
        self._channels = channels

    @property
    def channels(self):
        return self._channels


class POIRReader(ReaderBase):
    _format = 'poir'
    _mime_types = ['application/zip']
    _file_extensions = ['poir']

    _name = 'Olympus packed OIR'
    _description = '''
This reader imports Olympus packed OIR data files. These files are ZIP
containers that contain a number of OIR files.
'''

    def __init__(self, fobj):
        self._fobj = fobj
        self._readers = []
        with OpenFromAny(fobj, 'rb') as f:
            with ZipFile(f, 'r') as z:
                for fn in z.namelist():
                    try:
                        self._readers += [(fn, OIRReader(z.open(fn, 'r')))]
                    except UnsupportedFormatFeature:
                        # We ignore files with unsupported features. Those typically do not contain height information.
                        pass

    @property
    def channels(self):
        channels = []
        for fn, r in self._readers:
            for c in r.channels:
                c.reader = self
                c.tags['fn'] = fn
                channels += [c]
        return channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        channels = self.channels
        if channel_index < 0 or channel_index >= len(channels):
            raise RuntimeError(f'Channel index is {channel_index} but must be between 0 and {len(channels) - 1}.')

        # Get channel information
        channel = channels[channel_index]

        if physical_sizes is None:
            physical_sizes = channel.physical_sizes
        elif channel.physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is None:
            unit = channel.unit
        elif channel.unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        with OpenFromAny(self._fobj, 'rb') as f:
            with ZipFile(f, 'r') as z:
                reader = channel.tags['reader']
                fn = channel.tags['fn']
                height_data = reader(z.open(fn, 'r'))

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          physical_sizes,
                          unit=unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(height_scale_factor)
