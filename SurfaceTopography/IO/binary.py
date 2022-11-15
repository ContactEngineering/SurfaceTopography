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


def decode(stream_obj, structure_format, byte_order='@'):
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
    byte_order : str
        Byte order (see `struct.unpack`).

    Returns
    -------
    data : dict
        Dictionary with decoded data entries.
    """
    def mogrify_format(format):
        return_format = format
        if format.endswith('b'):  # bytes
            return_format = format[:-1] + 's'
        elif format.endswith('u'):  # UTF-8
            return_format = format[:-1] + 's'
        elif format.endswith('U'):  # UTF-16
            return_format = str(2*int(format[:-1])) + 's'
        return return_format

    def decode_data(data, format):
        if format.endswith('s'):
            return data.decode('latin1').strip('\x00').strip(' ')
        elif format.endswith('u'):
            return data.decode('utf-8').strip('\x00').strip(' ')
        elif format.endswith('U'):
            return data.decode('utf-16').strip('\x00').strip(' ')
        else:
            return data

    format_str = byte_order + ''.join([mogrify_format(format) for name, format in structure_format])
    unpacked_data = unpack(format_str, stream_obj.read(calcsize(format_str)))
    return {name: decode_data(data, format) for (name, format), data in zip(structure_format, unpacked_data)}
