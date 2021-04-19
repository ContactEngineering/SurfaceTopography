#
# Copyright 2021 Lars Pastewka
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

import io

###

height_units = {'m': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'nm': 1e-9, 'Å': 1e-10}
voltage_units = {'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9}

units = dict(height=height_units, voltage=voltage_units)

CHANNEL_NAME_INFO_KEY = 'channel_name'

###


def get_unit_conversion_factor(unit1_str, unit2_str):
    """
    Compute factor for conversion from unit1 to unit2. Return None if units are
    incompatible.
    """
    if unit1_str == unit2_str:
        return 1
    unit1_kind = None
    unit2_kind = None
    unit_scales = None
    for key, values in units.items():
        if unit1_str in values:
            unit1_kind = key
            unit_scales = values
        if unit2_str in values:
            unit2_kind = key
            unit_scales = values
    if unit1_kind is None or unit2_kind is None or unit1_kind != unit2_kind:
        return None
    return unit_scales[unit1_str] / unit_scales[unit2_str]


def mangle_length_unit_utf8(unit):
    if isinstance(unit, str):
        unit = unit.strip()
    else:
        unit = unit.decode('utf-8').strip()
    if unit == '':
        return None
    elif unit == 'A':
        return 'Å'
    elif unit == 'μm' or unit == 'um' or unit == '~m':
        return 'µm'
    else:
        return unit


def mangle_length_unit_ascii(unit):
    unit = unit.strip()
    if unit == '':
        return None
    elif unit == 'Å':
        return 'A'
    elif unit == 'μm' or unit == 'µm' or unit == '~m':
        return 'um'
    else:
        return unit


###

def is_binary_stream(fobj):
    """Check whether fobj is a binary stream"""
    return isinstance(fobj, io.BytesIO) or (
            hasattr(fobj, 'mode') and 'b' in fobj.mode)


def text(func):
    """Decorator that turns the first argument into a binary stream"""

    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'r', encoding='utf-8')
            fobj_text = fobj
            close_file = True
        elif is_binary_stream(fobj):
            fobj_text = io.TextIOWrapper(fobj, encoding='utf-8')
        else:
            fobj_text = fobj

        try:
            retvals = func(fobj_text, *args, **kwargs)
        finally:
            if is_binary_stream(fobj):
                fobj_text.detach()
                fobj_text = fobj
            if close_file:
                fobj_text.close()
        return retvals

    return func_wrapper


class OpenFromAny(object):
    """
    Context manager for turning file names or already open streams into a
    single stream format for subsequent reading. The file is left in an
    identical state when the context manager returns.
    """

    def __init__(self, fobj, mode='r'):
        """
        Open file

        Arguments
        ---------
        fobj : str or stream
            The file to be opened, specified either as a file name or a
            stream object.
        mode : str
            Open as text ('r') or binary ('rb').
        """
        self._fobj = fobj
        self._mode = mode
        self._fstream = None
        self._stream_position = None
        self._already_open = None

    def __enter__(self):
        # depending from where this function is called, self._fobj might already
        # be a filestream
        self._already_open = False
        if not hasattr(self._fobj, 'read'):
            # This is a string
            self._fstream = open(self._fobj, self._mode)
        else:
            # This is a stream that is already open
            self._already_open = True
            if self._mode == 'rb':
                # Turn this into a binary stream, if it is a text stream
                if isinstance(self._fobj, io.TextIOBase):
                    # file was opened without the 'b' option, so read its buffer to get the binary data
                    self._fstream = self._fobj.buffer
                else:
                    self._fstream = self._fobj
            elif self._mode == 'r':
                if isinstance(self._fobj, io.TextIOBase):
                    # file was opened without the 'b' option, so read its buffer to get the binary data
                    self._fstream = self._fobj
                else:
                    raise ValueError("Don't know how turn a binary stream into a text stream")
            else:
                return ValueError(f"Unknown file open mode '{self._mode}'.")
        self._stream_position = self._fstream.tell()
        return self._fstream

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Close file, unless it was already open when this object was constructed
        if not self._already_open:
            self._fstream.close()

        # Reset internal state
        self._fstream = None
        self._stream_position = None
        self._already_open = None
