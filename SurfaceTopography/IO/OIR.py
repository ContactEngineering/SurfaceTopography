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
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, If, For, While, Skip
from ..Exceptions import CorruptFile, FileFormatMismatch

OIR_CHUNK_XML0 = 0
OIR_CHUNK_XML = 1
OIR_CHUNK_BMP = 2
OIR_CHUNK_WTF = 3
OIR_CHUNK_TERMINATOR = 96


def _metadata_block(name):
    return BinaryStructure(name, [
        ('content_size', 'I'),
        ('nb_fragments', 'I'),
        For('fragments',
            lambda context: context[name].nb_fragments,
            BinaryStructure('xml', [
                ('xml', 'T'),
                ('root_name', 'T'),
                ('md5', '16s'),
            ]))
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
        BinaryStructure([
            ('magic', '16s', Validate('OLYMPUSRAWFORMAT', FileFormatMismatch)),
            ('unknown1', 'I', Validate(12, CorruptFile)),
            ('unknown2', 'I', Validate(0, CorruptFile)),
            ('unknown3', 'I', Validate(1, CorruptFile)),
            ('unknown4', 'I', Validate(2, CorruptFile)),
            ('file_size', 'I', DebugOutput()),
            ('unknown5', 'I', DebugOutput()),  # , Validate(2, CorruptFile)),
            ('some_size', 'I', DebugOutput()),
            ('unknown6', 'I', Validate(0, CorruptFile)),
            ('unknown7', 'I', Validate(17, CorruptFile)),
            ('unknown8', 'I', Validate(0, CorruptFile)),
            ('unknown9', 'I', Validate(1, CorruptFile)),
            ('unknown10', 'I', Validate(0, CorruptFile)),
            ('unknown11', 'I', DebugOutput()),
            ('unknown12', 'I', Validate(0, CorruptFile)),
            ('unknown_str', '8s', Validate('UNKNOWN', FileFormatMismatch)),
            ('unknown13', 'I', Validate(1, CorruptFile)),
            ('unknown14', 'I', Validate(1, CorruptFile)),
            ('unknown15', 'I', Validate(0xFFFFFFFF, CorruptFile)),
            ('unknown16', 'I', Validate(0xFFFFFFFF, CorruptFile)),
        ], name='header'),
        While(
            BinaryStructure([('chunk_size', 'I', DebugOutput()),
                             ('chunk_type', 'I', DebugOutput())]),
            lambda context: context.chunk_type in [OIR_CHUNK_XML0, OIR_CHUNK_XML, OIR_CHUNK_BMP, OIR_CHUNK_WTF],
            If(lambda context: context.chunk_type == OIR_CHUNK_XML0,
               While(
                   BinaryStructure([
                       ('id', 'I', DebugOutput()),
                       ('unknown3', 'I', Validate(2, CorruptFile)),
                       ('unknown4', 'I', Validate(1, CorruptFile)),
                       ('unknown5', 'I', DebugOutput()),
                       ('xml_dxx', 'I', DebugOutput()),
                       ('unknown7', 'I', Validate(1, CorruptFile)),
                       ('unknown8', 'I', Validate(1, CorruptFile)),
                       ('unknown9', 'I', Validate(1, CorruptFile)),
                       ('unknown10', 'I', Validate(1, CorruptFile)),
                       ('xml', 'T', Convert(xmltodict.parse), DebugOutput())]),
                   lambda context: context.id < 4,
                   name='fragments'
               ),
               lambda context: context.chunk_type == OIR_CHUNK_XML,
               BinaryStructure([
                   ('id', 'I', DebugOutput()),
                   ('unknown3', 'I', Validate(2, CorruptFile)),
                   ('unknown4', 'I', Validate(1, CorruptFile)),
                   ('unknown5', 'I', Validate(4, CorruptFile)),
                   ('xml_dxx', 'I', DebugOutput()),
                   ('unknown7', 'I', Validate(1, CorruptFile)),
                   ('unknown8', 'I', Validate(1, CorruptFile)),
                   ('unknown9', 'I', Validate(1, CorruptFile)),
                   ('unknown10', 'I', Validate(1, CorruptFile)),
                   ('xml', 'T', Convert(xmltodict.parse), DebugOutput())]),
               lambda context: context.chunk_type == OIR_CHUNK_BMP,
               CompoundLayout([
                   BinaryStructure([
                       ('unknown1', 'I', DebugOutput()),
                       ('unknown2', 'I', DebugOutput()),
                       ('unknown3', 'I', DebugOutput()),
                       ('unknown4', 'I', DebugOutput()),
                       ('unknown5', 'I', DebugOutput()),
                       ('unknown6', 'T', DebugOutput()),
                   ]),
                   Skip(lambda context: context.chunk_size)
               ]),
               lambda context: context.chunk_type == OIR_CHUNK_WTF,
               CompoundLayout([
                   BinaryStructure([
                       ('unknown2', 'I', DebugOutput()),  # Validate(0, CorruptFile)),
                       ('image_size', 'I', DebugOutput()),  # Validate(935 * 1024, CorruptFile)),
                       ('uuid', 'T', DebugOutput()),
                       ('image_size_again', 'I', DebugOutput(),
                        Validate(lambda x, context: x == context.image_size, CorruptFile)),
                       ('unknown3', 'I', Validate(4, CorruptFile))],
                       name='header'),
                   BinaryArray('data', lambda context: context.header.image_size, np.dtype('b'))
               ])),
            name='chunks'
        ),
    ])

    @property
    def channels(self):
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
