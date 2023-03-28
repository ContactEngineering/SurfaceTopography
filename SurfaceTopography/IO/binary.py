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

from struct import calcsize, unpack


def decode(stream_obj, structure_format, byte_order='@', return_size=False):
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
        defined in the `struct` module, plus 'u' for UTF-8 and 'U' for UTF-16.
        Decoder also supports per-entry endianness.
    byte_order : str, optional
        Byte order (see `struct.unpack`). (Default: '@')
    return_size : bool, optional
        Return the total size the structure in addition to the decoded data.
        (Default: False)

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
            return str(2*int(format[:-1])) + 's'
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
            return data

    data_dict = {}
    total_size = 0
    for name, format in structure_format:
        native_format = mogrify_format(format)
        size = calcsize(native_format)
        total_size += size

        data = decode_data(unpack(native_format, stream_obj.read(size)), format)
        data_dict[name] = data

    if return_size:
        return data_dict, total_size
    else:
        return data_dict
