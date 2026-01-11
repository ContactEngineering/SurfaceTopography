#
# Copyright 2023-2025 Lars Pastewka
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

"""
Reader for Digital Surf Mountains MNT files.

The MNT format is a Microsoft Compound Document File (OLE) containing:
- ImagePreview: JPEG preview image
- ScopedContents: Binary data with TLV-encoded metadata and zlib-compressed
  height data
- ScopedResults: Parameter data
- XmlHeader: UTF-16 encoded XML metadata

The ScopedContents stream uses a hierarchical TLV (Tag-Length-Value) structure:
- First 8 bytes: uint64 LE size field (= stream size - 8)
- Remaining bytes: TLV entries with tag (uint16 LE) + size (uint64 LE) + data

Height data is stored as int16 pairs: (height, secondary). The height is
in the even indices. The secondary channel may be a validity flag (0/-1)
or other small values.
"""

import struct
import zlib

import numpy as np
import olefile

from ..Exceptions import CorruptFile, FileFormatMismatch
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase


class MNTReader(ReaderBase):
    _format = 'mnt'
    _mime_types = ['application/x-digitalsurf-mnt']
    _file_extensions = ['mnt']

    _name = 'Digital Surf Mountains'
    _description = '''
File format of the Digital Surf Mountains software. This format is a
Microsoft Compound Document (OLE) file containing TLV-encoded metadata
and compressed height data.
'''

    # TLV tag definitions for ScopedContents
    # Structure: tag_id -> (name, format_or_children)
    # Format codes: 'B'=uint8, 'H'=uint16, 'I'=uint32, 'Q'=uint64,
    #               'd'=double, 'utf16'=UTF-16 string with 0x04 prefix
    # Children: dict of nested tag definitions for container tags

    # Top-level tags (before main container)
    _top_level_tags = {
        0x0003: ('format_version', 'B'),
        0x0002: ('format_flags', 'I'),
        0x0004: ('unknown_1', 'Q'),
        0x0005: ('unknown_2', 'Q'),
        0x0001: ('main_container', 'container'),
    }

    # Tags inside tag 0x00c8 (file metadata)
    _tag_c8_children = {
        0x0001: ('factor_a', 'I'),  # Rows factor A for dimension calculation
        0x0002: ('factor_b', 'I'),  # Rows factor B for dimension calculation
        0x0003: ('unknown_c8_1', 'I'),
        0x0004: ('build_number', 'I'),  # Software build number
        0x0005: ('unknown_c8_2', 'I'),
        0x0006: ('unknown_c8_3', 'I'),
        0x0007: ('unknown_c8_4', 'I'),
        0x0008: ('unknown_c8_5', 'container'),  # Variable-size block
        0x0009: ('unknown_c8_6', 'H'),  # Value 67 = 'C' in ASCII
        0x000a: ('unknown_c8_7', 'B'),
    }

    # Tags inside tag 0x012d -> 0x0009 (measurement settings)
    _tag_012d_009_children = {
        0x0001: ('unknown_009_1', 'I'),
        0x0002: ('unknown_009_2', 'I'),
        0x0003: ('unknown_009_3', 'I'),
        0x0004: ('unknown_009_4', 'I'),
        0x0005: ('unknown_009_5', 'I'),
        0x0006: ('unknown_009_6', 'I'),
        0x0007: ('unknown_009_7', 'I'),
        0x0008: ('unknown_009_8', 'I'),
        0x0009: ('unknown_009_9', 'I'),
        0x000a: ('unknown_009_10', 'I'),
        0x000b: ('unknown_009_11', 'I'),
        0x000c: ('unknown_009_12', 'I'),
        0x000d: ('unknown_009_13', 'I'),
        0x000e: ('unknown_009_14', 'I'),
        0x000f: ('unknown_009_15', 'I'),
        0x0010: ('scale_x', 'd'),  # Scale factor X (10.0)
        0x0011: ('scale_y', 'd'),  # Scale factor Y (10.0)
        0x0012: ('scale_z', 'd'),  # Scale factor Z (10.0)
        0x0013: ('unit_x', 'utf16'),  # Unit string X ('mm')
        0x0014: ('unit_y', 'utf16'),  # Unit string Y ('mm')
        0x0015: ('unit_z', 'utf16'),  # Unit string Z ('mm')
        0x0016: ('unknown_009_16', 'I'),
        0x0017: ('unknown_009_17', 'I'),
        0x0018: ('unknown_009_18', 'I'),
        0x0019: ('unknown_009_19', 'd'),  # Value 0.2
        0x001a: ('unknown_009_20', 'd'),  # Value 0.2
        0x001b: ('unknown_009_21', 'I'),
        0x001c: ('unknown_009_22', 'I'),
        0x001d: ('unknown_009_23', 'I'),
        0x001e: ('unknown_009_24', 'I'),
        0x001f: ('unit_display', 'utf16'),  # Display unit ('mm')
        0x0020: ('unit_other', 'utf16'),  # Other unit ('eV')
        0x0021: ('unknown_009_25', 'I'),
    }

    # Tags inside tag 0x012d -> 0x000a (height settings)
    _tag_012d_00a_children = {
        0x0001: ('unknown_00a_1', 'I'),
        0x0002: ('unknown_00a_2', 'Q'),
        0x0003: ('unknown_00a_3', 'Q'),
        0x0004: ('height_unit', 'utf16'),  # Height unit ('µm')
        0x0005: ('unknown_00a_4', 'I'),
    }

    # Tags inside tag 0x012d (extended metadata container)
    _tag_012d_children = {
        0x0001: ('params_1', 'container'),  # Contains doubles: 0.5, 2.5, etc.
        0x0002: ('params_2', 'container'),
        0x0003: ('params_3', 'container'),
        0x0004: ('params_4', 'container'),
        0x0005: ('params_5', 'container'),
        0x0006: ('params_6', 'container'),
        0x0007: ('params_7', 'container'),
        0x0008: ('params_8', 'container'),
        0x0009: ('measurement_settings', 'container'),  # Uses _tag_012d_009_children
        0x000a: ('height_settings', 'container'),  # Uses _tag_012d_00a_children
        0x000b: ('params_9', 'container'),
        0x000c: ('params_10', 'container'),
        0x000d: ('params_11', 'container'),
        0x000e: ('params_12', 'container'),
        0x000f: ('params_13', 'container'),
    }

    # Tags inside tag 0x0006 (pixel scale factors)
    _tag_0006_children = {
        0x0001: ('pixel_scale_1', 'd'),  # Value 10.0
        0x0002: ('pixel_scale_2', 'd'),  # Value 10.0
        0x0003: ('pixel_scale_3', 'd'),  # Value 10.0
        0x0004: ('pixel_scale_4', 'd'),  # Value 10.0
    }

    # Tags inside tag 0x0003 (dimension parameters)
    _tag_0003_children = {
        0x0001: ('dimension_param_1', 'I'),  # Values: 1123 / 816
        0x0002: ('dimension_param_2', 'I'),  # Values: 1588 / 1054
    }

    # Tags inside tag 0x000a (at main level)
    _tag_000a_children = {
        0x0001: ('unknown_0a_1', 'H'),
        0x0002: ('unknown_0a_2', 'H'),
        0x0003: ('unknown_0a_3', 'container'),
        0x0004: ('unknown_0a_4', 'container'),
        0x0005: ('unknown_0a_5', 'container'),
    }

    # Tags at the main container level (inside tag 0x0001)
    _main_container_tags = {
        0x00c8: ('file_metadata', 'container'),  # Uses _tag_c8_children
        0x00c9: ('unknown_c9', 'I'),
        0x00ca: ('unknown_ca', 'container'),  # Variable-size block
        0x00cb: ('serial_number', 'utf16'),  # DS-XXXXXXXXX
        0xffff: ('marker', 'container'),  # Separator/marker
        0x012d: ('extended_metadata', 'container'),  # Uses _tag_012d_children
        0x0190: ('unknown_190', 'I'),
        0x0001: ('unknown_main_1', 'I'),  # Value 1
        0x0002: ('unknown_main_2', 'I'),  # Value 0
        0x0003: ('dimension_params', 'container'),  # Uses _tag_0003_children
        0x0004: ('unknown_main_3', 'I'),  # Values: 794 / 1054
        0x0005: ('unknown_main_4', 'I'),  # Value 0
        0x0006: ('pixel_scales', 'container'),  # Uses _tag_0006_children
        0x0007: ('unknown_main_5', 'I'),  # Value 0
        0x0008: ('unknown_main_6', 'I'),  # Values: 2 / 1
        0x0009: ('unknown_main_7', 'I'),  # Value 0
        0x000a: ('unknown_main_8', 'container'),  # Uses _tag_000a_children
        0x000b: ('pixel_aspect_ratio', 'd'),  # Values: 0.918347 / 1.0
        0x000c: ('unknown_main_9', 'I'),  # Value 96
        0x0064: ('unknown_main_10', 'I'),  # Value 0
        0x0065: ('unknown_main_11', 'I'),  # Value 1
        0x0066: ('unknown_main_12', 'B'),  # Value 4
        0x000d: ('unknown_main_13', 'I'),  # Values: 10 / 2
        0x0258: ('color_palette', 'container'),  # Color palette data
        0x02bd: ('height_data', 'container'),  # Compressed height blocks
        0x0014: ('trailing_1', 'container'),
        0x0015: ('trailing_2', 'container'),
        0x0016: ('trailing_3', 'container'),
        0x0017: ('trailing_4', 'B'),
        0x0018: ('trailing_5', 'B'),
        0x0019: ('trailing_6', 'container'),
        0x001a: ('trailing_7', 'container'),
        0x001b: ('trailing_8', 'container'),
    }

    def __init__(self, fobj):
        """
        Load Digital Surf Mountains data files.

        Arguments
        ---------
        fobj : filename or file object
            File or data stream to open.
        """
        self._fobj = fobj
        self._channels = []

        with OpenFromAny(fobj, 'rb') as f:
            # Read enough bytes for OLE detection and initial parsing
            header = f.read(8)
            f.seek(0)

            # Check for OLE signature
            if header[:8] != b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
                raise FileFormatMismatch('Not an OLE compound document')

            # Read entire file for OLE parsing
            file_data = f.read()

        # Parse as OLE file
        try:
            ole = olefile.OleFileIO(file_data)
        except Exception as e:
            raise FileFormatMismatch(f'Failed to parse OLE file: {e}')

        # Verify this is an MNT file by checking for expected streams
        if not ole.exists('ScopedContents'):
            ole.close()
            raise FileFormatMismatch('Missing ScopedContents stream')

        # Read ScopedContents
        scoped_contents = ole.openstream('ScopedContents').read()

        # Verify minimum header length for dimension extraction
        if len(scoped_contents) < 0x72:
            ole.close()
            raise CorruptFile('ScopedContents header too short')

        # Extract dimension parameters from header
        # These are at fixed byte offsets within the TLV structure
        num_blocks = scoped_contents[0x3d]  # Number of compressed blocks
        factor_a = scoped_contents[0x63]    # First rows-per-block factor
        factor_b = scoped_contents[0x71]    # Second rows-per-block factor
        rows_per_block = factor_a * factor_b

        if num_blocks == 0 or rows_per_block == 0:
            ole.close()
            raise CorruptFile('Invalid dimension parameters in header')

        # Find all zlib-compressed blocks
        # Each block has a 16-byte prefix:
        #   Bytes 0-7:  uint64 LE - Element offset (for ordering)
        #   Bytes 8-11: uint32 LE - Elements per block
        #   Bytes 12-15: uint32 LE - Compressed size
        zlib_blocks = []
        i = 0
        while i < len(scoped_contents) - 2:
            # Check for zlib header bytes
            if scoped_contents[i] == 0x78 and scoped_contents[i + 1] in [0x01, 0x5e, 0x9c, 0xda]:
                try:
                    decompressed = zlib.decompress(scoped_contents[i:])
                    if len(decompressed) >= 1000:  # Only substantial blocks
                        # Extract block prefix information
                        element_offset = 0
                        elements_per_block = 0
                        if i >= 16:
                            prefix = scoped_contents[i - 16:i]
                            element_offset = struct.unpack('<Q', prefix[0:8])[0]
                            elements_per_block = struct.unpack('<I', prefix[8:12])[0]
                        zlib_blocks.append({
                            'pos': i,
                            'size': len(decompressed),
                            'data': decompressed,
                            'element_offset': element_offset,
                            'elements_per_block': elements_per_block
                        })
                except zlib.error:
                    pass
            i += 1

        if not zlib_blocks:
            ole.close()
            raise CorruptFile('No zlib-compressed data found')

        # Group blocks by elements_per_block to identify height data blocks
        blocks_by_epb = {}
        for block in zlib_blocks:
            epb = block['elements_per_block']
            if epb not in blocks_by_epb:
                blocks_by_epb[epb] = []
            blocks_by_epb[epb].append(block)

        # The largest group of same-sized blocks contains height data
        height_epb = max(blocks_by_epb.keys(), key=lambda k: len(blocks_by_epb[k]))
        height_blocks = blocks_by_epb[height_epb]
        elements_per_block = height_epb

        # Sort blocks by element_offset to get correct ordering
        height_blocks.sort(key=lambda b: b['element_offset'])

        # Calculate dimensions from header parameters and block info
        nx = elements_per_block // rows_per_block
        ny = num_blocks * rows_per_block

        # Verify the calculation
        if elements_per_block % rows_per_block != 0:
            ole.close()
            raise CorruptFile(
                f'Elements per block ({elements_per_block}) not divisible by '
                f'rows per block ({rows_per_block})'
            )

        if len(height_blocks) != num_blocks:
            ole.close()
            raise CorruptFile(
                f'Found {len(height_blocks)} height blocks, expected {num_blocks}'
            )

        # MNT files store data as int16 pairs: (height, secondary)
        # The secondary channel may be:
        #   - A validity flag (only 0 and -1 values) - used to create mask
        #   - Some other small value (e.g., error estimate) - ignored
        # Heights are always in the even indices (first int16 of each pair)
        combined_sample = b''.join(b['data'] for b in height_blocks[:4])
        arr_i16_sample = np.frombuffer(combined_sample, dtype='<i2')
        secondary_sample = arr_i16_sample[1::2]

        # Check if secondary channel is a validity flag (only contains 0 and -1)
        unique_secondary = np.unique(secondary_sample)
        has_validity_flag = set(unique_secondary).issubset({-1, 0})

        # Store metadata for later use
        self._ole_data = file_data
        self._height_blocks = [(b['pos'], b['data']) for b in height_blocks]
        self._nx = nx
        self._ny = ny
        self._has_validity_flag = has_validity_flag

        # Physical sizes are difficult to reliably extract from MNT files
        # Default to pixel count (user can override with physical_sizes parameter)
        physical_size_x = float(nx)
        physical_size_y = float(ny)

        # Create channel info
        self._channels = [
            ChannelInfo(
                self,
                0,
                name='Default',
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(physical_size_x, physical_size_y),
                uniform=True,
                unit='µm',
                info={'raw_metadata': {'nx': nx, 'ny': ny}},
            )
        ]

        ole.close()

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        channel = self._channels[channel_index]
        nx, ny = channel.nb_grid_pts

        # Combine all blocks (already sorted by element_offset)
        combined_data = b''.join(data for pos, data in self._height_blocks)

        # Data is stored as int16 pairs: (height, secondary)
        # Heights are in even indices, secondary channel in odd indices
        arr_i16 = np.frombuffer(combined_data, dtype='<i2')

        # Extract heights (even indices)
        heights_i16 = arr_i16[::2]

        # Reshape to image dimensions
        heights = heights_i16[:nx * ny].reshape(ny, nx).T.astype(float)

        # Create mask based on format
        if self._has_validity_flag:
            # Secondary channel is a validity flag: 0 = valid, -1 = invalid
            flags_i16 = arr_i16[1::2]
            flags = flags_i16[:nx * ny].reshape(ny, nx).T
            invalid_mask = (flags != 0)
        else:
            # No validity flag - only mask extreme int16 values
            invalid_mask = (heights <= -32760) | (heights >= 32760)

        # Apply scale factor if provided
        if height_scale_factor is not None:
            heights = heights * height_scale_factor

        # Check physical sizes
        sx, sy = channel.physical_sizes
        if physical_sizes is not None:
            sx, sy = physical_sizes

        # Build info dict
        _info = channel.info.copy()
        _info.update(info)

        if invalid_mask.any():
            heights = np.ma.masked_array(heights, mask=invalid_mask)

        topography = Topography(
            heights,
            physical_sizes=(sx, sy),
            unit=unit if unit is not None else channel.unit,
            info=_info,
            periodic=periodic,
        )

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
