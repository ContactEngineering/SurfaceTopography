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
- ScopedContents: Binary data with zlib-compressed height data
- ScopedResults: Parameter data
- XmlHeader: UTF-16 encoded XML metadata
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
Microsoft Compound Document (OLE) file containing compressed height data
and metadata.
'''

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

        # Extract dimensions from header (at offset 0x4f as uint16 pair)
        if len(scoped_contents) < 0x53:
            ole.close()
            raise CorruptFile('ScopedContents too short')

        nx_header, ny_header = struct.unpack('<HH', scoped_contents[0x4f:0x53])

        # Find all zlib-compressed blocks
        # Each block has a 16-byte prefix containing row offset info
        zlib_blocks = []
        i = 0
        while i < len(scoped_contents) - 2:
            # Check for zlib header bytes
            if scoped_contents[i] == 0x78 and scoped_contents[i + 1] in [0x01, 0x5e, 0x9c, 0xda]:
                try:
                    decompressed = zlib.decompress(scoped_contents[i:])
                    if len(decompressed) >= 1000:  # Only substantial blocks
                        # Extract row_offset from prefix (bytes 1-2 as little-endian uint16)
                        # This indicates the correct ordering of blocks
                        row_offset = 0
                        if i >= 16:
                            row_offset = struct.unpack('<H', scoped_contents[i-15:i-13])[0]
                        zlib_blocks.append((i, len(decompressed), decompressed, row_offset))
                except zlib.error:
                    pass
            i += 1

        if not zlib_blocks:
            ole.close()
            raise CorruptFile('No zlib-compressed data found')

        # Find the most common block size (this is the height data block size)
        block_sizes = {}
        for pos, size, data, row_offset in zlib_blocks:
            if size not in block_sizes:
                block_sizes[size] = []
            block_sizes[size].append((pos, data, row_offset))

        # Get the largest group of same-sized blocks (height data)
        height_block_size = max(block_sizes.keys(), key=lambda s: len(block_sizes[s]))
        height_blocks = block_sizes[height_block_size]
        num_height_blocks = len(height_blocks)

        # Sort blocks by row_offset (not file position) to get correct ordering
        height_blocks.sort(key=lambda x: x[2])

        # Try to determine dimensions and data format
        # Collect all possible interpretations and score them
        candidates = []

        # Helper function to add candidate if valid
        def add_candidate(test_nx, test_ny, test_dtype, elem_size):
            n_elements = height_block_size // elem_size
            if n_elements % test_nx != 0:
                return
            rows = n_elements // test_nx
            total_rows = rows * num_height_blocks

            if total_rows < test_ny:
                return  # Not enough data

            # Score: prefer interpretations where total_rows == test_ny (exact fit)
            # Lower ratio = better fit
            ratio = total_rows / test_ny if test_ny > 0 else float('inf')
            candidates.append((ratio, test_nx, test_ny, test_dtype, rows))

        # Try header dimensions with both dtypes
        if nx_header > 0 and ny_header > 0:
            add_candidate(nx_header, ny_header, '<i2', 2)
            add_candidate(nx_header, ny_header, '<i4', 4)

        # Try common width values to infer dimensions
        candidate_widths = [1280, 1024, 960, 800, 640, 512, 480, 400, 320, 256, 200, 100]
        for test_width in candidate_widths:
            # For int32
            n_elements_i32 = height_block_size // 4
            if n_elements_i32 % test_width == 0:
                rows_i32 = n_elements_i32 // test_width
                total_rows_i32 = rows_i32 * num_height_blocks
                add_candidate(test_width, total_rows_i32, '<i4', 4)

            # For int16
            n_elements_i16 = height_block_size // 2
            if n_elements_i16 % test_width == 0:
                rows_i16 = n_elements_i16 // test_width
                total_rows_i16 = rows_i16 * num_height_blocks
                add_candidate(test_width, total_rows_i16, '<i2', 2)

        if not candidates:
            ole.close()
            raise CorruptFile('Cannot determine data format or dimensions')

        # Prefer ratio=1.0 candidates (exact fit to block structure)
        # This means all compressed data is used exactly once
        ratio_one_candidates = [c for c in candidates if abs(c[0] - 1.0) < 0.01]

        if ratio_one_candidates:
            # Among ratio=1.0 candidates, prefer:
            # 1. int32 over int16 (more precision)
            # 2. Smaller ny (fewer total rows = smaller image)
            # 3. Larger nx (wider images are more common)
            ratio_one_candidates.sort(key=lambda c: (0 if c[3] == '<i4' else 1, c[2], -c[1]))
            candidates = ratio_one_candidates + [c for c in candidates
                                                 if c not in ratio_one_candidates]

        _, nx, ny, dtype, rows_per_block = candidates[0]

        if nx == 0 or ny == 0:
            ole.close()
            raise CorruptFile('Cannot determine data format or dimensions')

        # Store metadata for later use
        self._ole_data = file_data
        self._height_blocks = [(pos, data) for pos, data, row_offset in height_blocks]
        self._dtype = dtype
        self._rows_per_block = rows_per_block
        self._nx = nx
        self._ny = ny

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

        # Combine all blocks (already sorted by row_offset)
        combined_data = b''.join(data for pos, data in self._height_blocks)

        # MNT data is stored as (height_i16, flag_i16) pairs
        # The flag indicates validity: 0 = valid, -1 = invalid/masked
        arr_i16 = np.frombuffer(combined_data, dtype='<i2')

        # Extract heights (even indices) and flags (odd indices)
        heights_i16 = arr_i16[::2]
        flags_i16 = arr_i16[1::2]

        # Reshape to image dimensions
        heights = heights_i16[:nx * ny].reshape(ny, nx).T.astype(float)
        flags = flags_i16[:nx * ny].reshape(ny, nx).T

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

        # Create mask from flag channel (flag != 0 means invalid)
        invalid_mask = (flags != 0)

        # Also mask extreme values (shouldn't happen but just in case)
        invalid_mask |= (heights <= -32760) | (heights >= 32760)

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
