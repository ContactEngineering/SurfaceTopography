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

    # Block structures are initialized lazily to avoid circular imports
    _block_structures = None
    _tlv_size_bytes = 8  # MNT uses uint64 size fields

    @classmethod
    def _init_block_structures(cls):
        """Initialize block structure definitions using BlockDefinition."""
        # Use importlib to import tlv directly, bypassing package initialization
        # This avoids circular dependency during package initialization
        import importlib.util
        import os
        tlv_path = os.path.join(os.path.dirname(__file__), 'tlv.py')
        spec = importlib.util.spec_from_file_location('tlv', tlv_path)
        tlv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tlv_module)
        BlockDefinition = tlv_module.BlockDefinition

        # Main container children (tag 0x0001)
        main_container_children = {
            0x00c8: BlockDefinition(container=True),  # File metadata
            0x00c9: BlockDefinition(),  # Unknown
            0x00ca: BlockDefinition(container=True),  # Variable-size block
            0x00cb: BlockDefinition(),  # Serial number (UTF-16)
            0xffff: BlockDefinition(container=True),  # Marker
            0x012d: BlockDefinition(container=True),  # Extended metadata
            0x0190: BlockDefinition(),  # Unknown
            0x0001: BlockDefinition(),  # Unknown
            0x0002: BlockDefinition(),  # Unknown
            0x0003: BlockDefinition(container=True),  # Dimension params
            0x0004: BlockDefinition(),  # Unknown
            0x0005: BlockDefinition(),  # Unknown
            0x0006: BlockDefinition(container=True),  # Pixel scales
            0x0007: BlockDefinition(),  # Unknown
            0x0008: BlockDefinition(),  # Unknown
            0x0009: BlockDefinition(),  # Unknown
            0x000a: BlockDefinition(container=True),  # Unknown container
            0x000b: BlockDefinition(),  # Pixel aspect ratio
            0x000c: BlockDefinition(),  # Unknown
            0x0064: BlockDefinition(),  # Unknown
            0x0065: BlockDefinition(),  # Unknown
            0x0066: BlockDefinition(),  # Unknown
            0x000d: BlockDefinition(),  # Unknown
            0x0258: BlockDefinition(container=True),  # Color palette
            0x02bd: BlockDefinition(container=True),  # Height data container
            0x0014: BlockDefinition(container=True),
            0x0015: BlockDefinition(container=True),
            0x0016: BlockDefinition(container=True),
            0x0017: BlockDefinition(),
            0x0018: BlockDefinition(),
            0x0019: BlockDefinition(container=True),
            0x001a: BlockDefinition(container=True),
            0x001b: BlockDefinition(container=True),
        }

        # Top-level block structures (after 8-byte size header)
        cls._block_structures = {
            0x0001: BlockDefinition(container=main_container_children),
            0x0002: BlockDefinition(),  # Format flags
            0x0003: BlockDefinition(),  # Format version
            0x0004: BlockDefinition(),  # Unknown
            0x0005: BlockDefinition(),  # Unknown
        }

    def _tlv_read_header_from_bytes(self, data, offset):
        """Read TLV header (tag + size) from byte buffer."""
        header_size = 2 + self._tlv_size_bytes
        if offset + header_size > len(data):
            return None, None, offset

        tag = struct.unpack_from('<H', data, offset)[0]
        size = struct.unpack_from('<Q', data, offset + 2)[0]
        return tag, size, offset + header_size

    def _tlv_parse_nested_bytes(self, data, start=0, end=None, block_defs=None):
        """
        Parse nested TLV structure from a byte buffer.

        Parameters
        ----------
        data : bytes
            Binary data buffer containing TLV entries.
        start : int, optional
            Start offset within buffer. Default: 0.
        end : int, optional
            End offset within buffer. Default: len(data).
        block_defs : dict, optional
            Block definitions to use. Default: self._block_structures.

        Returns
        -------
        dict
            Dictionary of parsed entries, keyed by tag ID.
        """
        if end is None:
            end = len(data)
        if block_defs is None:
            block_defs = self._block_structures

        entries = {}
        pos = start

        while pos < end:
            tag, size, data_start = self._tlv_read_header_from_bytes(data, pos)
            if tag is None or size is None:
                break

            data_end = data_start + size
            if data_end > end:
                break

            block_def = block_defs.get(tag) if block_defs else None

            if block_def is None:
                # Unknown block - store raw data
                entry = {'_raw': data[data_start:data_end], '_size': size}
            elif block_def.text:
                # ASCII text block
                entry = {
                    'text': data[data_start:data_end].decode(
                        'ascii', errors='replace'
                    ).strip('\x00')
                }
            elif block_def.container:
                # Nested container - use child definitions if provided
                if isinstance(block_def.container, dict):
                    child_defs = block_def.container
                else:
                    child_defs = block_defs
                entry = self._tlv_parse_nested_bytes(
                    data, data_start, data_end, child_defs
                )
            elif block_def.fields:
                # Structured block - parse fields
                entry = self._parse_fields(data[data_start:data_end], block_def.fields)
            else:
                entry = {'_raw': data[data_start:data_end], '_size': size}

            # Handle duplicate tags by converting to list
            if tag in entries:
                if not isinstance(entries[tag], list):
                    entries[tag] = [entries[tag]]
                entries[tag].append(entry)
            else:
                entries[tag] = entry

            pos = data_end

        return entries

    def _parse_fields(self, data, fields):
        """Parse structured fields from byte data."""
        result = {}
        offset = 0
        for name, fmt in fields:
            if fmt == 'B':
                result[name] = data[offset]
                offset += 1
            elif fmt == 'H':
                result[name] = struct.unpack_from('<H', data, offset)[0]
                offset += 2
            elif fmt == 'I':
                result[name] = struct.unpack_from('<I', data, offset)[0]
                offset += 4
            elif fmt == 'Q':
                result[name] = struct.unpack_from('<Q', data, offset)[0]
                offset += 8
            elif fmt == 'd':
                result[name] = struct.unpack_from('<d', data, offset)[0]
                offset += 8
        return result

    def __init__(self, fobj):
        """
        Load Digital Surf Mountains data files.

        Arguments
        ---------
        fobj : filename or file object
            File or data stream to open.
        """
        # Initialize block structures if not already done
        if self._block_structures is None:
            self._init_block_structures()

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

        # Parse TLV structure for metadata storage
        # First 8 bytes are stream size (skip), then TLV entries start
        self._metadata = self._tlv_parse_nested_bytes(scoped_contents, start=8)

        # Extract dimension parameters from fixed byte offsets within TLV structure
        # These offsets are consistent across files and more reliable than
        # parsing nested TLV values which have variable encodings
        num_blocks = scoped_contents[0x3d]   # Number of compressed blocks
        factor_a = scoped_contents[0x63]     # First rows-per-block factor
        factor_b = scoped_contents[0x71]     # Second rows-per-block factor
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
            if (scoped_contents[i] == 0x78 and
                    scoped_contents[i + 1] in [0x01, 0x5e, 0x9c, 0xda]):
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
