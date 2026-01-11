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
Unified TLV (Tag-Length-Value) reader infrastructure for Digital Surf formats.

Both FRT and MNT formats use TLV encoding with:
- Tag: uint16 LE
- Size: uint32 LE (FRT 1.00) or uint64 LE (FRT 1.01, MNT)
- Data: variable length

This module provides:
- BlockDefinition: Declarative block structure definitions
- TLVReaderMixin: Common TLV parsing methods for reader classes
"""

import os
import struct


class BlockDefinition:
    """
    Declarative definition for a TLV block structure.

    Parameters
    ----------
    fields : list of tuples, optional
        Field definitions for decode(): [(name, format), ...]
        Format codes follow struct module plus extensions in binary.py.
    text : bool, optional
        If True, entire block content is ASCII text. Default: False.
    container : bool or dict, optional
        If True, block contains nested TLV entries (parsed generically).
        If dict, maps child tag IDs to BlockDefinition instances for
        structured parsing of nested content. Default: False.
    trailing_data : bool, optional
        If True, record file offset of data following the defined fields.
        Useful for blocks with variable-length data at the end.
        Default: False.
    subblocks : tuple, optional
        (count_field, field_list) for repeated sub-structures.
        count_field is the name of a field containing the repeat count.
        field_list is the structure definition for each subblock.
        Default: None.
    skip_rest : bool, optional
        If True, skip any remaining bytes after parsing fields.
        Useful for blocks with padding or unknown trailing data.
        Default: False.

    Examples
    --------
    Simple structured block:

    >>> BlockDefinition(fields=[
    ...     ('width', 'I'),
    ...     ('height', 'I'),
    ... ])

    Text block:

    >>> BlockDefinition(text=True)

    Block with trailing data:

    >>> BlockDefinition(
    ...     fields=[('count', 'I'), ('flags', 'I')],
    ...     trailing_data=True
    ... )

    Block with repeated subblocks:

    >>> BlockDefinition(
    ...     fields=[('nb_items', 'I')],
    ...     subblocks=('nb_items', [('value', 'd'), ('name', '32s')])
    ... )

    Container with known child structures:

    >>> BlockDefinition(container={
    ...     0x0001: BlockDefinition(fields=[('value', 'I')]),
    ...     0x0002: BlockDefinition(text=True),
    ... })
    """

    def __init__(self, fields=None, text=False, container=False,
                 trailing_data=False, subblocks=None, skip_rest=False):
        self.fields = fields
        self.text = text
        self.container = container
        self.trailing_data = trailing_data
        self.subblocks = subblocks
        self.skip_rest = skip_rest

    def __repr__(self):
        parts = []
        if self.fields:
            parts.append(f'fields={self.fields!r}')
        if self.text:
            parts.append('text=True')
        if self.container:
            parts.append(f'container={self.container!r}')
        if self.trailing_data:
            parts.append('trailing_data=True')
        if self.subblocks:
            parts.append(f'subblocks={self.subblocks!r}')
        if self.skip_rest:
            parts.append('skip_rest=True')
        return f'BlockDefinition({", ".join(parts)})'


class TLVReaderMixin:
    """
    Mixin class providing TLV parsing methods for file format readers.

    Subclasses should define:
    - _block_structures: dict mapping tag IDs to BlockDefinition instances
    - _tlv_size_bytes: 4 for uint32, 8 for uint64 size fields (default: 8)

    The mixin provides methods for:
    - Reading TLV headers
    - Parsing individual blocks according to their definitions
    - Parsing flat block sequences (FRT style)
    - Parsing nested containers (MNT style)
    """

    _block_structures = {}  # Override in subclass
    _tlv_size_bytes = 8  # Override in subclass if needed

    def _tlv_read_header(self, f):
        """
        Read a TLV header (tag + size) from a file stream.

        Parameters
        ----------
        f : file-like object
            Binary file stream positioned at start of TLV entry.

        Returns
        -------
        tag : int or None
            Tag value (uint16), or None if end of data.
        size : int or None
            Size of data section in bytes, or None if end of data.
        """
        tag_data = f.read(2)
        if len(tag_data) < 2:
            return None, None
        tag = struct.unpack('<H', tag_data)[0]

        if self._tlv_size_bytes == 4:
            size_data = f.read(4)
            if len(size_data) < 4:
                return None, None
            size = struct.unpack('<I', size_data)[0]
        else:
            size_data = f.read(8)
            if len(size_data) < 8:
                return None, None
            size = struct.unpack('<Q', size_data)[0]

        return tag, size

    def _tlv_read_header_from_bytes(self, data, offset):
        """
        Read a TLV header from a byte buffer.

        Parameters
        ----------
        data : bytes
            Binary data buffer.
        offset : int
            Offset within buffer to read from.

        Returns
        -------
        tag : int or None
            Tag value, or None if insufficient data.
        size : int or None
            Size value, or None if insufficient data.
        next_offset : int
            Offset after the header (start of data section).
        """
        header_size = 2 + self._tlv_size_bytes
        if offset + header_size > len(data):
            return None, None, offset

        tag = struct.unpack_from('<H', data, offset)[0]
        if self._tlv_size_bytes == 4:
            size = struct.unpack_from('<I', data, offset + 2)[0]
        else:
            size = struct.unpack_from('<Q', data, offset + 2)[0]

        return tag, size, offset + header_size

    def _tlv_parse_block(self, f, block_size, block_def):
        """
        Parse a single TLV block according to its definition.

        Parameters
        ----------
        f : file-like object
            Binary file stream positioned at start of block data.
        block_size : int
            Size of block data in bytes.
        block_def : BlockDefinition or None
            Block structure definition, or None for unknown blocks.

        Returns
        -------
        dict
            Parsed block metadata.
        """
        if block_def is None:
            # Unknown block - store offset and skip
            meta = {'block_size': block_size, 'block_offset': f.tell()}
            f.seek(block_size, os.SEEK_CUR)
            return meta

        # Text block
        if block_def.text:
            text_data = f.read(block_size)
            return {'text': text_data.decode('ascii', errors='replace').strip('\x00')}

        # Container block (nested TLV)
        if block_def.container:
            return self._tlv_parse_container(f, block_size, block_def.container)

        # Structured block with fields
        if block_def.fields is None:
            # No fields defined - skip block
            meta = {'block_size': block_size, 'block_offset': f.tell()}
            f.seek(block_size, os.SEEK_CUR)
            return meta

        # Lazy import to avoid circular dependency during package initialization
        from .binary import decode

        meta, size = decode(f, block_def.fields, return_size=True)

        # Handle trailing data
        if block_def.trailing_data:
            meta['data_offset'] = f.tell()
            remaining = block_size - size
            if remaining > 0:
                f.seek(remaining, os.SEEK_CUR)
            size = block_size

        # Handle subblocks
        if block_def.subblocks:
            count_field, subblock_fields = block_def.subblocks
            subblocks = []
            for _ in range(meta[count_field]):
                sub_meta, sub_size = decode(f, subblock_fields, return_size=True)
                subblocks.append(sub_meta)
                size += sub_size
            meta['subblocks'] = subblocks

        # Handle skip_rest
        if block_def.skip_rest:
            remaining = block_size - size
            if remaining > 0:
                f.seek(remaining, os.SEEK_CUR)

        return meta

    def _tlv_parse_container(self, f, container_size, child_defs):
        """
        Parse a container block with nested TLV entries.

        Parameters
        ----------
        f : file-like object
            Binary file stream positioned at start of container data.
        container_size : int
            Size of container data in bytes.
        child_defs : dict or bool
            If dict, maps child tag IDs to BlockDefinition instances.
            If True, parse children generically (store offsets).

        Returns
        -------
        dict
            Dictionary of parsed child entries, keyed by tag ID.
        """
        entries = {}
        end_pos = f.tell() + container_size

        while f.tell() < end_pos:
            tag, size = self._tlv_read_header(f)
            if tag is None or size is None:
                break

            # Bounds check
            if f.tell() + size > end_pos:
                break

            # Get child block definition
            if isinstance(child_defs, dict):
                child_def = child_defs.get(tag)
            else:
                child_def = None

            # Parse child block
            entry = self._tlv_parse_block(f, size, child_def)

            # Handle duplicate tags by converting to list
            if tag in entries:
                if not isinstance(entries[tag], list):
                    entries[tag] = [entries[tag]]
                entries[tag].append(entry)
            else:
                entries[tag] = entry

        return entries

    def _tlv_parse_flat_blocks(self, f, num_blocks):
        """
        Parse a flat sequence of TLV blocks (FRT style).

        Parameters
        ----------
        f : file-like object
            Binary file stream positioned at start of block sequence.
        num_blocks : int
            Number of blocks to parse.

        Returns
        -------
        dict
            Dictionary of parsed blocks, keyed by hex tag string.
        """
        metadata = {}

        for _ in range(num_blocks):
            tag, size = self._tlv_read_header(f)
            if tag is None or size is None:
                break

            block_def = self._block_structures.get(tag)
            meta = self._tlv_parse_block(f, size, block_def)
            metadata[hex(tag)] = meta

        return metadata

    def _tlv_parse_nested_bytes(self, data, start=0, end=None, block_defs=None):
        """
        Parse nested TLV structure from a byte buffer (MNT style).

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

            block_def = block_defs.get(tag)

            if block_def is None:
                # Unknown block
                entry = {'_raw': data[data_start:data_end], '_size': size}
            elif block_def.text:
                # Text block
                entry = {
                    'text': data[data_start:data_end].decode(
                        'ascii', errors='replace'
                    ).strip('\x00')
                }
            elif block_def.container:
                # Nested container
                if isinstance(block_def.container, dict):
                    child_defs = block_def.container
                else:
                    child_defs = block_defs
                entry = self._tlv_parse_nested_bytes(
                    data, data_start, data_end, child_defs
                )
            elif block_def.fields:
                # Structured block - parse from BytesIO
                from io import BytesIO

                # Lazy import to avoid circular dependency during package initialization
                from .binary import decode
                entry, _ = decode(
                    BytesIO(data[data_start:data_end]),
                    block_def.fields,
                    return_size=True
                )
            else:
                entry = {'_raw': data[data_start:data_end], '_size': size}

            # Handle duplicate tags
            if tag in entries:
                if not isinstance(entries[tag], list):
                    entries[tag] = [entries[tag]]
                entries[tag].append(entry)
            else:
                entries[tag] = entry

            pos = data_end

        return entries
