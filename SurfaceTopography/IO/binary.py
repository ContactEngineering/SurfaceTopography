#
# Copyright 2022-2023 Lars Pastewka
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

import math
import numbers
import os
import struct
import zlib
from struct import calcsize, unpack

import numpy as np


class ValidationError(Exception):
    pass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def decode(stream_obj, structure_format, byte_order='@', return_size=False, context={}):
    """
    Decode a binary stream given the sequence of binary entries. Strings are
    stripped of zeros and white spaces.

    Parameters
    ----------
    stream_obj : stream-like object
        Binary stream to decode.
    structure_format : list of tuples
        List of tuples describing the sequence of entries in the binary
        stream. Each tuple consists of two entries
            (name, format)
        that give the name of the entry and the format. We support the format
        defined in the `struct` module, plus 'u' for UTF-8, 'U' for UTF-16,
        't' for a Pascal string with 16-bit length and 'T' for a Pascal string
        with 32-bit length. Decoder also supports per-entry endianness.
    byte_order : str, optional
        Byte order (see `struct.unpack`). (Default: '@')
    return_size : bool, optional
        Return the total size the structure in addition to the decoded data.
        (Default: False)
    context : dict, optional
        Context dictionary passed to validation and converter functions.
        (Default: {})

    Returns
    -------
    data : dict
        Dictionary with decoded data entries.
    size : int
        Size of the structure in the native binary form. (Only returned
        if `return_size` is True.)
    """

    def mogrify_format(format):
        """Convert format string into something that struct.unpack can parse"""
        if format.endswith('b'):  # bytes
            return format[:-1] + 's'
        elif format.endswith('u'):  # UTF-8
            return format[:-1] + 's'
        elif format.endswith('U'):  # UTF-16
            return str(2 * int(format[:-1])) + 's'
        else:
            if format.startswith('>') or format.startswith('<'):
                return format
            else:
                return byte_order + format

    def decode_data(data, format):
        """Perform additional decoding step on top of struct.unpack"""
        if len(data) == 1:
            # Data is always a tuple or list
            data, = data
        if format.endswith('s'):
            return data.decode('latin1').strip('\x00').strip(' ')
        elif format.endswith('u'):
            return data.decode('utf-8').strip('\x00').strip(' ')
        elif format.endswith('U'):
            return data.decode('utf-16').strip('\x00').strip(' ')
        else:
            # We need to sanitize NaNs. This is because NaN == NaN returns
            # false. If NaNs show up in our metadata dictionaries, then
            # equality tests for those dictionaries will *always fail*. We
            # sanitize NaNs to Python `None`, which translates to `null` in
            # JSON.
            if type(data) is tuple or type(data) is list:
                return [None if isinstance(x, numbers.Number) and math.isnan(x) else x for x in data]
            else:
                return None if isinstance(data, numbers.Number) and math.isnan(data) else data

    local_context = AttrDict()
    total_size = 0
    for entry in structure_format:
        # Check if entry is a layout class (has from_stream method)
        if hasattr(entry, 'from_stream'):
            start_pos = stream_obj.tell()
            result = entry.from_stream(stream_obj, AttrDict({**local_context, **context}))
            total_size += stream_obj.tell() - start_pos
            if isinstance(result, dict):
                local_context.update(result)
            continue

        entry = list(entry)
        name, format = entry[:2]

        # Special formats
        if format == 't':
            strlen, = unpack(byte_order + 'H', stream_obj.read(2))
            data = stream_obj.read(strlen).decode('ascii').strip('\x00')
            size = strlen + 2
        elif format == 'T':
            strlen, = unpack(byte_order + 'I', stream_obj.read(4))
            data = stream_obj.read(strlen).decode('ascii').strip('\x00')
            size = strlen + 4
        else:
            native_format = mogrify_format(format)
            size = calcsize(native_format)
            data = decode_data(unpack(native_format, stream_obj.read(size)), format)

        # Track size of data structure
        total_size += size

        for converter in entry[2:]:
            data = converter(data, AttrDict({**local_context, **context}), name=name,
                             file_offset=stream_obj.tell() - size)

        if name is not None:
            local_context[name] = data

    if return_size:
        return local_context, total_size
    else:
        return local_context


class Convert:
    def __init__(self, fun, exception=None):
        self._fun = fun
        self._exception = exception

    def __call__(self, data, context, **kwargs):
        if self._exception is None:
            return self._fun(data)
        else:
            try:
                return self._fun(data)
            except Exception as exc:
                raise self._exception(f"Conversion of entry `{kwargs['name']}` at file offset {kwargs['file_offset']} "
                                      f"failed.") from exc


class Validate:
    def __init__(self, value, exception=ValidationError):
        self._value = value
        self._exception = exception

    def __call__(self, data, context, **kwargs):
        if self._value is not None:
            # If a validator is given, then check the validity of this entry
            if callable(self._value):
                if not self._value(data, context):
                    raise self._exception(f"Structure entry `{kwargs['name']}` has invalid value '{data}' at file "
                                          f"offset {kwargs['file_offset']}.")
            else:
                if data != self._value:
                    raise self._exception(f"Structure entry `{kwargs['name']}` has invalid value '{data}' at file "
                                          f"offset {kwargs['file_offset']}. The expected value is '{self._value}'.")

        return data


class DebugOutput:
    def __init__(self, prefix='', context=False):
        self._prefix = prefix
        self._context = context

    def __call__(self, data, context, **kwargs):
        if self._context:
            print(f'{kwargs["file_offset"]}: {self._prefix}{kwargs["name"]} = {data}; context = {context}')
        else:
            print(f'{kwargs["file_offset"]}: {self._prefix}{kwargs["name"]} = {data}')
        return data


class LayoutWithNameBase:
    """Base class for file layout classes"""

    _name = None

    def name(self, context):
        if callable(self._name):
            return self._name(context)
        else:
            return self._name


class BinaryStructure(LayoutWithNameBase):
    def __init__(self, structure_format, byte_order='@', name=None):
        """
        Define a binary stream given the sequence of binary entries.

        Parameters
        ----------
        structure_format : list of tuples
            List of tuples describing the sequence of entries in the binary
            stream. Each tuple consists of two entries
                (name, format)
            that give the name of the entry and the format. We support the format
            defined in the `struct` module, plus 'u' for UTF-8, 'U' for UTF-16,
            't' for a Pascal string with 16-bit length and 'T' for a Pascal string
            with 32-bit length. Decoder also supports per-entry endianness.
        byte_order : str, optional
            Byte order (see `struct.unpack`). (Default: '@')
        name : str, optional
            Name of this structure. (Default: None)
        """
        self._structure_format = structure_format
        self._byte_order = byte_order
        self._name = name

    def from_stream(self, stream_obj, context):
        """
        Decode stream into dictionary.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        data : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        decoded_data : dict
            Dictionary with decoded data entries.
        """
        local_context = decode(stream_obj, self._structure_format, byte_order=self._byte_order, context=context)
        name = self.name(context)
        if name is None:
            return local_context
        else:
            return {name: local_context}


class BinaryArray:
    def __init__(self, name, shape, dtype, conversion_fun=lambda x: x, mask_fun=None):
        """
        Defines flat binary data to be read into a numpy array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : function or tuple
            Function that returns the shape and takes a input the current data
            dictionary.
        dtype : function or dtype
            Function that returns the dtype and takes a input the current data
            dictionary.
        conversion_fun : function
            Function that converts the array after reading. This can be useful
            for example to change the data format or transpose the array.
        mask_fun : function
            Function that returns a mask with undefined data points.
        """
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._conversion_fun = conversion_fun
        self._mask_fun = mask_fun

    def name(self, context):
        return self._name

    def from_stream(self, stream_obj, context):
        """
        Skip over data block and return reader for block.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        context : dict
            Context dictionary with file reader.
        """

        class ReaderProxy:
            def __init__(self, binary_array, data, file_pos):
                self._binary_array = binary_array
                self._data = data
                self._file_pos = file_pos

            def __call__(self, stream_obj):
                stream_obj.seek(self._file_pos)
                return self._binary_array.read(stream_obj, self._data)

        if callable(self._shape):
            shape = self._shape(context)
        else:
            shape = self._shape
        if callable(self._dtype):
            dtype = self._dtype(context)
        else:
            dtype = self._dtype

        file_pos = stream_obj.tell()

        stream_obj.seek(np.prod(shape) * dtype.itemsize, os.SEEK_CUR)

        return {self.name(context): ReaderProxy(self, context, file_pos)}

    def read(self, stream_obj, context):
        """
        Read data block into numpy array.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        data : numpy.ndarray
            Nunpy array containing the data from the file.
        """
        if callable(self._shape):
            shape = self._shape(context)
        else:
            shape = self._shape
        if callable(self._dtype):
            dtype = self._dtype(context)
        else:
            dtype = self._dtype

        buffer = stream_obj.read(np.prod(shape) * dtype.itemsize)
        arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        if self._mask_fun is not None:
            arr = np.ma.masked_array(arr, mask=self._mask_fun(arr, context))
        return self._conversion_fun(arr)


class RawBuffer:
    def __init__(self, name, size=None, lazy=True):
        """
        Defines a raw binary data block.

        Parameters
        ----------
        name : str
            Name of the data block.
        size : int, callable, or None
            Size of the data block in bytes. Can be:
            - An integer for fixed size
            - A callable that takes context and returns size
            - None to read size from context['_block_size'] (for TLV parsing)
        lazy : bool, optional
            If True, return a ReaderProxy for deferred reading.
            If False, read data immediately. Default: True.
        """
        self._name = name
        self._size = size
        self._lazy = lazy

    def name(self, context):
        return self._name

    def _get_size(self, context):
        """Get the size from the configured source."""
        if self._size is None:
            return context.get('_block_size', 0)
        elif callable(self._size):
            return self._size(context)
        else:
            return self._size

    def from_stream(self, stream_obj, context):
        """
        Read or skip over data block.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        dict
            Dictionary with the data or a reader proxy.
        """
        size = self._get_size(context)
        name = self.name(context)

        if self._lazy:
            # Lazy loading - skip data and return proxy
            class ReaderProxy:
                def __init__(self, raw_buffer, data, file_pos):
                    self._raw_buffer = raw_buffer
                    self._data = data
                    self._file_pos = file_pos

                def __call__(self, stream_obj):
                    stream_obj.seek(self._file_pos)
                    return self._raw_buffer.read(stream_obj, self._data)

            file_pos = stream_obj.tell()
            stream_obj.seek(size, os.SEEK_CUR)
            return {name: ReaderProxy(self, context, file_pos)}
        else:
            # Immediate loading - read data now
            data = stream_obj.read(size)
            return {name: {'_raw': data, '_size': size}}

    def read(self, stream_obj, context):
        """
        Read data block.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        buffer : bytes
            Buffer containing the raw data.
        """
        size = self._get_size(context)
        return stream_obj.read(size)


class TextBuffer(LayoutWithNameBase):
    """
    Reads a text block from a binary stream.

    Parameters
    ----------
    name : str
        Name of the text entry.
    size : int, callable, or None
        Size of the text block in bytes. Can be:
        - An integer for fixed size
        - A callable that takes context and returns size
        - None to read size from context['_block_size'] (for TLV parsing)
    encoding : str, optional
        Text encoding. Default: 'ascii'.
    """

    def __init__(self, name, size=None, encoding='ascii'):
        self._name = name
        self._size = size
        self._encoding = encoding

    def name(self, context):
        return self._name

    def _get_size(self, context):
        """Get the size from the configured source."""
        if self._size is None:
            return context.get('_block_size', 0)
        elif callable(self._size):
            return self._size(context)
        else:
            return self._size

    def from_stream(self, stream_obj, context):
        size = self._get_size(context)
        data = stream_obj.read(size)
        text = data.decode(self._encoding, errors='replace').strip('\x00')
        return {self.name(context): text}


class TLVContainer(LayoutWithNameBase):
    """
    Reads TLV (Tag-Length-Value) encoded blocks from a binary stream.

    TLV format:
    - Tag: identifies the block type (uint16 by default)
    - Length: size of the data section
    - Value: the actual data

    Parameters
    ----------
    tag_map : dict
        Maps tag IDs to layout classes. Each layout class must have a
        `from_stream(stream_obj, context)` method. Special values:
        - None or missing: store raw bytes
        - 'text': treat as ASCII text
        - Layout class instance: use that layout to parse
    name : str, optional
        Name for this container in the result dict. Default: None.
    tag_format : str, optional
        Format for tag field. Default: '<H' (uint16 LE).
    size_format : str, optional
        Format for size field. Default: '<Q' (uint64 LE).
        Use '<I' for uint32.
    count : int or callable, optional
        Number of TLV entries to read. If None, reads until end of
        container (requires knowing the container size). Default: None.
    container_size : int or callable, optional
        Total size of the container in bytes. Used when count is None
        to determine when to stop reading. Default: None.
    store_by_name : bool, optional
        If True, also store entries by their layout's name (if available)
        in addition to by tag ID. Default: True.
    entry_prefix_format : str, optional
        Format for a prefix field before each TLV entry. If set, this
        many bytes are read and skipped before each entry. Used for
        structures where each TLV entry is preceded by a size hint.
        Default: None (no prefix).
    """

    def __init__(self, tag_map, name=None, tag_format='<H', size_format='<Q',
                 count=None, container_size=None, store_by_name=True,
                 entry_prefix_format=None):
        self._tag_map = tag_map
        self._name = name
        self._tag_format = tag_format
        self._size_format = size_format
        self._count = count
        self._container_size = container_size
        self._store_by_name = store_by_name
        self._entry_prefix_format = entry_prefix_format

    def name(self, context):
        return self._name

    def _read_header(self, stream_obj):
        """Read TLV header (tag + size) from stream."""
        tag_size = calcsize(self._tag_format)
        tag_data = stream_obj.read(tag_size)
        if len(tag_data) < tag_size:
            return None, None
        tag = unpack(self._tag_format, tag_data)[0]

        size_size = calcsize(self._size_format)
        size_data = stream_obj.read(size_size)
        if len(size_data) < size_size:
            return None, None
        size = unpack(self._size_format, size_data)[0]

        return tag, size

    def _parse_entry(self, stream_obj, tag, size, context):
        """Parse a single TLV entry based on its tag."""
        layout = self._tag_map.get(tag)

        if layout is None:
            # Unknown tag - store raw bytes
            return {'_raw': stream_obj.read(size), '_size': size}, None

        if layout == 'text':
            # Text block shorthand
            data = stream_obj.read(size)
            text = data.decode('ascii', errors='replace').strip('\x00')
            return text, None

        if isinstance(layout, TLVContainer):
            # Nested container - create context with size info
            nested_context = AttrDict({**context, '_container_size': size})
            # Override container_size for nested parsing
            old_size = layout._container_size
            layout._container_size = size
            result = layout.from_stream(stream_obj, nested_context)
            layout._container_size = old_size
            # Extract the inner dict if it was wrapped with a name
            layout_name = layout.name(context)
            if layout_name and layout_name in result:
                return result[layout_name], layout_name
            return result, layout_name

        if hasattr(layout, 'from_stream'):
            # Layout class with from_stream method
            # Create a sub-context with size information
            sub_context = AttrDict({**context, '_block_size': size})
            result = layout.from_stream(stream_obj, sub_context)
            layout_name = layout.name(context) if hasattr(layout, 'name') else None
            # Handle layouts that return {name: value}
            if layout_name and isinstance(result, dict) and layout_name in result:
                return result[layout_name], layout_name
            return result, layout_name

        # Fallback - store raw bytes
        return {'_raw': stream_obj.read(size), '_size': size}, None

    def from_stream(self, stream_obj, context):
        """
        Parse TLV entries from stream.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        dict
            Dictionary with parsed entries keyed by tag ID.
        """
        entries = {}
        start_pos = stream_obj.tell()

        # Determine how many entries to read
        count = self._count(context) if callable(self._count) else self._count
        container_size = (self._container_size(context)
                          if callable(self._container_size)
                          else self._container_size)

        entries_read = 0
        while True:
            # Check termination conditions
            if count is not None and entries_read >= count:
                break
            if container_size is not None:
                if stream_obj.tell() - start_pos >= container_size:
                    break

            # Skip entry prefix if configured (e.g., look-ahead size field)
            if self._entry_prefix_format is not None:
                prefix_size = calcsize(self._entry_prefix_format)
                stream_obj.seek(prefix_size, os.SEEK_CUR)

            # Read header
            tag, size = self._read_header(stream_obj)
            if tag is None or size is None:
                break

            # Bounds check for container
            if container_size is not None:
                if stream_obj.tell() + size > start_pos + container_size:
                    break

            # Parse entry
            entry, layout_name = self._parse_entry(stream_obj, tag, size, context)

            # Store by tag ID
            if tag in entries:
                if not isinstance(entries[tag], list):
                    entries[tag] = [entries[tag]]
                entries[tag].append(entry)
            else:
                entries[tag] = entry

            # Also store by name if available and enabled
            if self._store_by_name and layout_name:
                if layout_name in entries:
                    if not isinstance(entries[layout_name], list):
                        entries[layout_name] = [entries[layout_name]]
                    entries[layout_name].append(entry)
                else:
                    entries[layout_name] = entry

            entries_read += 1

        # Wrap in name if provided
        name = self.name(context)
        if name is not None:
            return {name: entries}
        return entries


class LayoutWithTrailingData(LayoutWithNameBase):
    """
    Layout that parses structured fields then stores remaining bytes as raw data.

    This is useful for TLV blocks that have a known header structure followed
    by variable-length data (e.g., compressed data blocks).

    Parameters
    ----------
    name : str
        Name for this block in the result dict.
    fields : list
        Field definitions for the header. Can include:
        - Tuples: (name, format) following struct module conventions
        - Layout classes: Objects with from_stream() method (like For)
    """

    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def from_stream(self, stream_obj, context):
        block_size = context.get('_block_size', 0)

        # Parse fields (decode handles layout classes in fields)
        result, fields_size = decode(
            stream_obj, self._fields, byte_order='<', return_size=True,
            context=context
        )

        # Store remaining bytes as raw data
        remaining = block_size - fields_size
        if remaining > 0:
            result['_raw'] = stream_obj.read(remaining)

        name = self.name(context)
        if name:
            return {name: result}
        return result


class ZlibBlockChain(LayoutWithNameBase):
    """
    Reads a chain of sequential zlib-compressed blocks from a binary stream.

    This class is designed for file formats like MNT that store height data
    as a series of zlib-compressed blocks with a fixed prefix structure.
    The blocks are stored sequentially in memory and can be chained together
    using the compressed_size field in each prefix.

    Block structure:
    ```
    [prefix][zlib data][prefix][zlib data]...
    ```

    Default prefix format (16 bytes):
    - Bytes 0-7:  uint64 LE - element_offset (for logical ordering)
    - Bytes 8-11: uint32 LE - elements_per_block
    - Bytes 12-15: uint32 LE - compressed_size

    The class scans for the first zlib block by looking for the zlib magic
    byte (0x78), then chains through all subsequent blocks using the
    compressed_size field.

    Parameters
    ----------
    name : str
        Name for the parsed result in the context dictionary.
    prefix_format : str, optional
        Struct format for the block prefix. Default: '<QII' (uint64 + 2*uint32).
        The last field must be the compressed_size.
    min_decompressed_size : int, optional
        Minimum decompressed size to consider a valid block. This filters
        out false positive zlib matches. Default: 1000.

    Examples
    --------
    >>> layout = ZlibBlockChain('height_blocks')
    >>> result = layout.from_stream(stream, {})
    >>> blocks = result['height_blocks']
    >>> for block in blocks:
    ...     elem_offset = block['elem_offset']
    ...     data = block['data']  # decompressed bytes
    """

    # Valid zlib compression level markers
    ZLIB_COMPRESSION_LEVELS = [0x01, 0x5E, 0x9C, 0xDA]

    def __init__(self, name, prefix_format='<QII', min_decompressed_size=1000):
        self._name = name
        self._prefix_format = prefix_format
        self._prefix_size = struct.calcsize(prefix_format)
        self._min_decompressed_size = min_decompressed_size

    def from_stream(self, stream_obj, context):
        """
        Parse zlib-compressed blocks from stream.

        Parameters
        ----------
        stream_obj : stream-like object
            Binary stream to decode.
        context : dict
            Dictionary with data that has been decoded at this point.

        Returns
        -------
        dict
            Dictionary with {name: list of block dicts}, where each block
            dict contains 'elem_offset', 'elem_per_block', and 'data' keys.
        """
        data = stream_obj.read()
        blocks = self._find_and_chain_blocks(data)
        return {self._name: blocks}

    def _find_first_zlib(self, data):
        """
        Scan for the first valid zlib block.

        Parameters
        ----------
        data : bytes
            Raw data to search.

        Returns
        -------
        int or None
            Position of the first zlib stream, or None if not found.
        """
        # Need at least prefix + 2 bytes for zlib header
        for i in range(self._prefix_size, len(data) - 2):
            # Check for zlib magic byte (0x78)
            if data[i] == 0x78 and data[i + 1] in self.ZLIB_COMPRESSION_LEVELS:
                # Verify there's a valid prefix before this position
                prefix = data[i - self._prefix_size:i]
                prefix_values = struct.unpack(self._prefix_format, prefix)
                comp_size = prefix_values[-1]  # Last field is compressed_size

                # Sanity check on compressed size
                if 0 < comp_size < 10000000:
                    try:
                        decompressed = zlib.decompress(data[i:i + comp_size])
                        if len(decompressed) >= self._min_decompressed_size:
                            return i
                    except zlib.error:
                        pass
        return None

    def _find_and_chain_blocks(self, data):
        """
        Find the first zlib block, then chain through all blocks.

        Parameters
        ----------
        data : bytes
            Raw data containing the blocks.

        Returns
        -------
        list of dict
            List of block dictionaries, each containing:
            - 'elem_offset': element offset for logical ordering
            - 'elem_per_block': number of elements in this block
            - 'data': decompressed data bytes
        """
        first_zlib = self._find_first_zlib(data)
        if first_zlib is None:
            return []

        blocks = []
        pos = first_zlib - self._prefix_size  # Start at first prefix

        while pos < len(data) - self._prefix_size:
            # Read prefix
            prefix = data[pos:pos + self._prefix_size]
            prefix_values = struct.unpack(self._prefix_format, prefix)

            # Default prefix format is <QII: elem_offset, elem_per_block, comp_size
            elem_offset = prefix_values[0]
            elem_per_block = prefix_values[1] if len(prefix_values) > 2 else 0
            comp_size = prefix_values[-1]

            # Check for end of blocks
            if comp_size == 0 or pos + self._prefix_size + comp_size > len(data):
                break

            # Decompress block
            zlib_start = pos + self._prefix_size
            try:
                decompressed = zlib.decompress(
                    data[zlib_start:zlib_start + comp_size]
                )
                blocks.append({
                    'elem_offset': elem_offset,
                    'elem_per_block': elem_per_block,
                    'data': decompressed
                })
            except zlib.error:
                break

            # Move to next prefix
            pos = zlib_start + comp_size

        return blocks
