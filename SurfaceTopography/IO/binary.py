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
    def __init__(self, name, size):
        """
        Defines flat binary data to be read into a numpy array.

        Parameters
        ----------
        name : str
            Name of the array.
        size : function or int
            Function that returns the size and takes a input the current data
            dictionary.
        """
        self._name = name
        self._size = size

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
            def __init__(self, raw_buffer, data, file_pos):
                self._raw_buffer = raw_buffer
                self._data = data
                self._file_pos = file_pos

            def __call__(self, stream_obj):
                stream_obj.seek(self._file_pos)
                return self._raw_buffer.read(stream_obj, self._data)

        size = self._size(context) if callable(self._size) else self._size

        file_pos = stream_obj.tell()

        stream_obj.seek(size, os.SEEK_CUR)

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
        buffer : bytes
            Buffer containing the raw data.
        """
        size = self._size(context) if callable(self._size) else self._size
        return stream_obj.read(size)
