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
from io import BytesIO

import numpy as np
import olefile

from ..Exceptions import CorruptFile, FileFormatMismatch
from ..UniformLineScanAndTopography import Topography
from .binary import BinaryStructure, RawBuffer, TextBuffer, TLVContainer
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

    # Block structures and TLV parser are initialized lazily to avoid circular imports
    _block_structures = None
    _tlv_parser = None

    @classmethod
    def _init_block_structures(cls):
        """Initialize TLV block structure definitions.

        MNT File Structure Overview
        ===========================
        The ScopedContents stream contains a hierarchical TLV structure:

        [8 bytes: uint64 stream size]
        [TLV entries...]

        Top-level tags:
        - 0x0001: Main data container (contains all measurement data)
        - 0x0002: Format flags (uint32)
        - 0x0003: Format version (uint32)

        Main container (0x0001) children:
        - 0x00c8: File metadata container (timestamps, software info)
        - 0x00c9: Unknown (small, possibly flags)
        - 0x00ca: Measurement parameters container
        - 0x00cb: Serial number (UTF-16 encoded string)
        - 0x012d: Extended metadata container
        - 0x0003: Dimension parameters (nb_blocks, rows_per_block factors)
        - 0x0006: Pixel scale factors (physical sizes, units)
        - 0x02bd: Height data container (zlib-compressed blocks)
        - 0x0258: Color palette/visualization settings
        - 0xffff: Section delimiter/marker

        Compressed Data Block Prefix (16 bytes before zlib stream):
        - Bytes 0-7:  uint64 LE - Element offset (for block ordering)
        - Bytes 8-11: uint32 LE - Elements per block (nx * rows_per_block)
        - Bytes 12-15: uint32 LE - Compressed size

        Height Data Format:
        - Stored as int16 pairs: (height_value, validity_flag)
        - Height values at even indices, validity at odd indices
        - Validity: 0 = valid, -1 = invalid/masked
        """

        # Helper to create RawBuffer with immediate loading (for unknown blocks)
        def raw(name):
            return RawBuffer(name, size=None, lazy=False)

        # Helper to create BinaryStructure with single uint32 field
        def uint32(name):
            return BinaryStructure([('value', 'I')], name=name)

        # =====================================================================
        # Nested container definitions (deepest first)
        #
        # Legend:
        #   [CONFIRMED] - Verified through hex analysis of multiple files
        #   [LIKELY]    - Strong evidence but not fully confirmed
        #   [GUESS]     - Speculative based on tag position or similar formats
        #   [UNKNOWN]   - Purpose unknown, included for completeness
        # =====================================================================

        SIZE_FMT = '<Q'  # MNT uses uint64 LE size fields

        # Container 0x00c8 children - [CONFIRMED] block dimension parameters
        # Evidence: Tags 0x0001 and 0x0002 contain factor_a and factor_b
        # which multiply to give rows_per_block
        block_params_children = {
            0x0001: uint32('factor_a'),   # [CONFIRMED] uint32
            0x0002: uint32('factor_b'),   # [CONFIRMED] uint32
            0x0003: uint32('param_3'),    # [UNKNOWN] uint32
            0x0004: uint32('param_4'),    # [UNKNOWN] uint32
            0x0005: uint32('param_5'),    # [UNKNOWN] uint32
            0x0006: uint32('param_6'),    # [UNKNOWN] uint32
            0x0007: uint32('param_7'),    # [UNKNOWN] uint32
            0x0008: raw('param_8'),       # [UNKNOWN] variable size
            0x0009: raw('param_9'),       # [UNKNOWN] variable size
            0x000a: raw('param_10'),      # [UNKNOWN] variable size
        }

        # Container 0x0003 children (inside main) - dimension parameters
        dimension_params_children = {
            0x0001: uint32('nx'),         # [LIKELY] X dimension as uint32
            0x0002: uint32('ny'),         # [LIKELY] Y dimension as uint32
        }

        # Container 0x0006 children - [GUESS] possibly pixel scales/units
        # Evidence: Similar structure to SUR format scale blocks
        pixel_scale_children = {
            0x0001: raw('x_scale'),                  # [GUESS] Possibly X scale
            0x0002: raw('y_scale'),                  # [GUESS] Possibly Y scale
            0x0003: raw('z_scale'),                  # [GUESS] Possibly Z scale
            0x0004: TextBuffer('x_unit'),            # [LIKELY] Contains ASCII text (unit?)
            0x0005: TextBuffer('y_unit'),            # [LIKELY] Contains ASCII text (unit?)
            0x0006: TextBuffer('z_unit'),            # [LIKELY] Contains ASCII text (unit?)
        }

        # Container 0x00ca children - [GUESS] possibly measurement parameters
        measurement_params_children = {
            0x0001: raw('measurement_type'),         # [UNKNOWN] 1 byte
            0x0002: raw('scan_params'),              # [UNKNOWN]
            0x0003: raw('instrument_settings'),      # [CONFIRMED] Container with unknown structure
            0x0004: TextBuffer('instrument_name'),   # [LIKELY] Contains ASCII text
            0x0005: TextBuffer('objective'),         # [LIKELY] Contains ASCII text
        }

        # Container 0x02bd children - [CONFIRMED] height data
        # Evidence: Contains zlib-compressed blocks with known prefix structure
        height_data_children = {
            0x0001: raw('data_header'),              # [UNKNOWN] Appears before compressed data
            0x0002: raw('compressed_blocks'),        # [CONFIRMED] Compressed data blocks (zlib stream)
        }

        # Container 0x012d children - [UNKNOWN]
        extended_metadata_children = {
            0x0001: TextBuffer('extended_description'),  # [LIKELY] Contains ASCII text
            0x0002: raw('extended_params'),          # [CONFIRMED] Container with unknown structure
            0x0003: raw('extended_flags'),           # [UNKNOWN]
        }

        # Container 0x0258 children - [GUESS] possibly visualization/palette
        # Evidence: Only present in some files, similar tag range to other formats
        color_palette_children = {
            0x0001: raw('palette_info'),             # [UNKNOWN]
            0x0002: raw('palette_data'),             # [UNKNOWN]
        }

        # =====================================================================
        # Main container children (tag 0x0001)
        # [CONFIRMED] Tag 0x0001 is the main container
        # =====================================================================
        main_container_children = {
            # [CONFIRMED] These tags exist and are containers or leaf nodes as marked
            0x00c8: TLVContainer(block_params_children, name='block_params', size_format=SIZE_FMT),
            0x00c9: raw('file_flags'),               # [UNKNOWN]
            0x00cb: TextBuffer('serial_number'),     # [LIKELY] Contains text (serial number?)

            0x00ca: TLVContainer(measurement_params_children, name='measurement_params', size_format=SIZE_FMT),
            0x012d: TLVContainer(extended_metadata_children, name='extended_metadata', size_format=SIZE_FMT),
            0x0190: raw('acquisition_mode'),         # [UNKNOWN]

            # Grid structure - [CONFIRMED] tag 0x0003 contains dimension info
            0x0001: raw('data_type'),                # [UNKNOWN]
            0x0002: raw('format_flags'),             # [UNKNOWN]
            0x0003: TLVContainer(dimension_params_children, name='dimension_params', size_format=SIZE_FMT),
            0x0004: raw('grid_type'),                # [UNKNOWN]
            0x0005: raw('data_encoding'),            # [UNKNOWN]

            # [GUESS] Tags 0x0006-0x000d might be physical dimensions
            0x0006: TLVContainer(pixel_scale_children, name='pixel_scales', size_format=SIZE_FMT),
            0x0007: raw('origin_x'),                 # [UNKNOWN]
            0x0008: raw('origin_y'),                 # [UNKNOWN]
            0x0009: raw('origin_z'),                 # [UNKNOWN]
            0x000a: raw('coordinate_system'),        # [CONFIRMED] Container with unknown structure
            0x000b: raw('aspect_ratio'),             # [UNKNOWN]
            0x000c: raw('rotation'),                 # [UNKNOWN]
            0x000d: raw('tilt'),                     # [UNKNOWN]

            # [GUESS] Tags 0x0064-0x0066 grouped together, maybe statistics
            0x0064: raw('height_stats'),             # [UNKNOWN]
            0x0065: raw('rms_stats'),                # [UNKNOWN]
            0x0066: raw('higher_moments'),           # [UNKNOWN]

            0x0258: TLVContainer(color_palette_children, name='color_palette', size_format=SIZE_FMT),

            # [CONFIRMED] Height data container
            0x02bd: TLVContainer(height_data_children, name='height_data', size_format=SIZE_FMT),

            # [GUESS] Tags 0x0014-0x001b might be processing/ROI related
            # These are containers but with unknown child structure, so we store raw data
            0x0014: raw('filter_history'),           # [CONFIRMED] Container with unknown structure
            0x0015: raw('leveling_history'),         # [CONFIRMED] Container with unknown structure
            0x0016: raw('form_removal'),             # [CONFIRMED] Container with unknown structure
            0x0017: raw('processing_flags'),         # [UNKNOWN]
            0x0018: raw('quality_score'),            # [UNKNOWN]
            0x0019: raw('roi_definitions'),          # [CONFIRMED] Container with unknown structure
            0x001a: raw('mask_regions'),             # [CONFIRMED] Container with unknown structure
            0x001b: raw('annotations'),              # [CONFIRMED] Container with unknown structure

            0xffff: raw('section_marker'),           # [LIKELY] Delimiter/marker (common pattern)
        }

        # =====================================================================
        # Top-level block structures (after 8-byte size header)
        # [CONFIRMED] TLV structure: uint16 tag + uint64 size + data
        # =====================================================================
        cls._block_structures = {
            0x0001: TLVContainer(main_container_children, name='main', size_format=SIZE_FMT),
            0x0002: raw('format_flags'),             # [UNKNOWN]
            0x0003: raw('format_version'),           # [UNKNOWN]
            0x0004: raw('file_type'),                # [UNKNOWN]
            0x0005: BinaryStructure([('value', 'Q')], name='num_blocks'),  # [CONFIRMED] uint64
        }

        # Create top-level TLV parser
        cls._tlv_parser = TLVContainer(
            cls._block_structures,
            size_format=SIZE_FMT
        )

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
        stream = BytesIO(scoped_contents[8:])
        self._metadata = self._tlv_parser.from_stream(stream, {})

        # Extract dimension parameters from parsed metadata
        try:
            num_blocks = self._metadata['num_blocks']['value']
        except (KeyError, TypeError):
            ole.close()
            raise CorruptFile('Missing num_blocks in metadata')

        try:
            main = self._metadata['main']
            block_params = main['block_params']
            factor_a = block_params['factor_a']['value']
            factor_b = block_params['factor_b']['value']
        except (KeyError, TypeError):
            ole.close()
            raise CorruptFile('Missing block parameters in metadata')

        rows_per_block = factor_a * factor_b

        if num_blocks == 0 or rows_per_block == 0:
            ole.close()
            raise CorruptFile('Invalid dimension parameters in metadata')

        # Extract compressed height data from parsed metadata
        try:
            height_data = main['height_data']
            compressed_blocks_raw = height_data['compressed_blocks']['_raw']
        except (KeyError, TypeError):
            ole.close()
            raise CorruptFile('Missing compressed height data in metadata')

        # Find all zlib-compressed blocks within the compressed data
        # Each block has a 16-byte prefix:
        #   Bytes 0-7:  uint64 LE - Element offset (for ordering)
        #   Bytes 8-11: uint32 LE - Elements per block
        #   Bytes 12-15: uint32 LE - Compressed size
        zlib_blocks = []
        i = 0
        while i < len(compressed_blocks_raw) - 2:
            # Check for zlib header bytes
            if (compressed_blocks_raw[i] == 0x78 and
                    compressed_blocks_raw[i + 1] in [0x01, 0x5e, 0x9c, 0xda]):
                try:
                    decompressed = zlib.decompress(compressed_blocks_raw[i:])
                    if len(decompressed) >= 1000:  # Only substantial blocks
                        # Extract block prefix information
                        element_offset = 0
                        elements_per_block = 0
                        if i >= 16:
                            prefix = compressed_blocks_raw[i - 16:i]
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
            raise CorruptFile('No zlib-compressed data found in height data')

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
