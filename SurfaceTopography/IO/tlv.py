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
TLV (Tag-Length-Value) parsing infrastructure for Digital Surf file formats.

This module provides common TLV parsing functionality used by both FRT and MNT
file format readers. Both formats use a similar structure:
- Tag: uint16 LE
- Size: uint32 LE (FRT 1.00) or uint64 LE (FRT 1.01, MNT)
- Data: variable length

The MNT format uses nested TLV structures where container tags hold additional
TLV sequences.
"""

import struct
from io import BytesIO


class TLVParser:
    """
    Parser for TLV (Tag-Length-Value) encoded binary data.

    Parameters
    ----------
    size_bytes : int
        Number of bytes for the size field (4 for uint32, 8 for uint64).
    byte_order : str
        Byte order ('<' for little-endian, '>' for big-endian).
    """

    def __init__(self, size_bytes=8, byte_order='<'):
        self.size_bytes = size_bytes
        self.byte_order = byte_order
        if size_bytes == 4:
            self._size_format = f'{byte_order}I'
        elif size_bytes == 8:
            self._size_format = f'{byte_order}Q'
        else:
            raise ValueError(f"Unsupported size_bytes: {size_bytes}")
        self._tag_format = f'{byte_order}H'
        self._header_size = 2 + size_bytes

    def parse_header(self, data, offset):
        """
        Parse a TLV header at the given offset.

        Returns
        -------
        tuple : (tag, size, data_start)
            Tag value, data size, and offset where data starts.
        """
        if offset + self._header_size > len(data):
            return None, None, None
        tag = struct.unpack_from(self._tag_format, data, offset)[0]
        size = struct.unpack_from(self._size_format, data, offset + 2)[0]
        return tag, size, offset + self._header_size

    def parse_entries(self, data, start, end, tag_structures=None, max_depth=10, depth=0):
        """
        Parse TLV entries from a byte buffer.

        Parameters
        ----------
        data : bytes
            The binary data to parse.
        start : int
            Start offset in the data.
        end : int
            End offset in the data.
        tag_structures : dict, optional
            Dictionary mapping tag IDs to structure definitions.
        max_depth : int
            Maximum recursion depth for nested containers.
        depth : int
            Current recursion depth.

        Returns
        -------
        dict : Parsed entries keyed by tag ID.
        """
        if tag_structures is None:
            tag_structures = {}

        entries = {}
        pos = start

        while pos < end:
            tag, size, data_start = self.parse_header(data, pos)
            if tag is None or size is None:
                break
            if size > end - data_start:
                break

            data_end = data_start + size
            content = data[data_start:data_end]

            # Check if we have a structure definition for this tag
            if tag in tag_structures:
                structure = tag_structures[tag]
                entry = self._parse_structure(content, structure, tag_structures,
                                              max_depth, depth)
            else:
                # Unknown tag - store raw content
                entry = {'_raw': content, '_size': size}

            # Handle duplicate tags by converting to list
            if tag in entries:
                if not isinstance(entries[tag], list):
                    entries[tag] = [entries[tag]]
                entries[tag].append(entry)
            else:
                entries[tag] = entry

            pos = data_end

        return entries

    def _parse_structure(self, content, structure, tag_structures, max_depth, depth):
        """Parse content according to a structure definition."""
        if structure is None:
            return {'_raw': content, '_size': len(content)}

        if isinstance(structure, str):
            # Simple type: 'B', 'H', 'I', 'Q', 'd', 'f', 'utf16', etc.
            return self._parse_simple_type(content, structure)

        if isinstance(structure, list):
            # List of (name, format) tuples - sequential structure
            return self._parse_sequential(content, structure)

        if isinstance(structure, dict):
            # Nested TLV container
            if '_type' in structure and structure['_type'] == 'container':
                if depth >= max_depth:
                    return {'_raw': content, '_size': len(content)}
                nested_structures = structure.get('_children', {})
                merged_structures = {**tag_structures, **nested_structures}
                return self.parse_entries(content, 0, len(content),
                                          merged_structures, max_depth, depth + 1)
            else:
                # Dictionary of tag -> structure mappings (alternative format)
                return self.parse_entries(content, 0, len(content),
                                          structure, max_depth, depth + 1)

        return {'_raw': content, '_size': len(content)}

    def _parse_simple_type(self, content, type_str):
        """Parse content as a simple type."""
        bo = self.byte_order
        if type_str == 'B' and len(content) >= 1:
            return struct.unpack_from(f'{bo}B', content)[0]
        elif type_str == 'H' and len(content) >= 2:
            return struct.unpack_from(f'{bo}H', content)[0]
        elif type_str == 'I' and len(content) >= 4:
            return struct.unpack_from(f'{bo}I', content)[0]
        elif type_str == 'Q' and len(content) >= 8:
            return struct.unpack_from(f'{bo}Q', content)[0]
        elif type_str == 'd' and len(content) >= 8:
            return struct.unpack_from(f'{bo}d', content)[0]
        elif type_str == 'f' and len(content) >= 4:
            return struct.unpack_from(f'{bo}f', content)[0]
        elif type_str == 'utf16':
            # UTF-16 string with type byte prefix (0x04)
            if len(content) > 1 and content[0] == 0x04:
                try:
                    return content[1:].decode('utf-16-le').rstrip('\x00\uffff')
                except UnicodeDecodeError:
                    pass
            return content
        elif type_str == 'raw':
            return content
        else:
            return content

    def _parse_sequential(self, content, structure):
        """Parse content as a sequential structure of named fields."""
        result = {}
        stream = BytesIO(content)
        bo = self.byte_order

        for field_def in structure:
            name = field_def[0]
            fmt = field_def[1]

            if fmt == 'B':
                data = stream.read(1)
                if len(data) < 1:
                    break
                result[name] = struct.unpack(f'{bo}B', data)[0]
            elif fmt == 'H':
                data = stream.read(2)
                if len(data) < 2:
                    break
                result[name] = struct.unpack(f'{bo}H', data)[0]
            elif fmt == 'I':
                data = stream.read(4)
                if len(data) < 4:
                    break
                result[name] = struct.unpack(f'{bo}I', data)[0]
            elif fmt == 'Q':
                data = stream.read(8)
                if len(data) < 8:
                    break
                result[name] = struct.unpack(f'{bo}Q', data)[0]
            elif fmt == 'd':
                data = stream.read(8)
                if len(data) < 8:
                    break
                result[name] = struct.unpack(f'{bo}d', data)[0]
            elif fmt == 'f':
                data = stream.read(4)
                if len(data) < 4:
                    break
                result[name] = struct.unpack(f'{bo}f', data)[0]
            elif fmt.endswith('s'):
                # Fixed-length string
                length = int(fmt[:-1])
                data = stream.read(length)
                result[name] = data.decode('latin1').rstrip('\x00')
            else:
                # Unknown format, skip
                break

        return result
