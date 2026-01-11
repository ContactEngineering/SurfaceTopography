#
# Copyright 2021-2024 Lars Pastewka
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
import zipfile

###

CHANNEL_NAME_INFO_KEY = 'channel_name'


###

def is_binary_stream(fobj):
    """Check whether fobj is a binary stream"""
    return isinstance(fobj, io.BytesIO) or \
        isinstance(fobj, zipfile.ZipExtFile) or \
        (hasattr(fobj, 'mode') and 'b' in fobj.mode)


# Hint: We code use UnicodeDammit or chardet to guess the encoding
def text(encoding='utf-8'):
    def decorator(func):
        """Decorator that turns the first argument into a binary stream"""

        def func_wrapper(fobj, *args, **kwargs):
            close_file = False
            if not hasattr(fobj, 'read'):
                fobj = open(fobj, mode='r', encoding=encoding)
                fobj_text = fobj
                close_file = True
            elif is_binary_stream(fobj):
                fobj_text = io.TextIOWrapper(fobj, encoding=encoding)
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

    return decorator


def _get_binary_stream(fstream):
    if isinstance(fstream, io.TextIOBase):
        # file was opened without the 'b' option, so read its buffer to get the binary data
        if hasattr(fstream, 'buffer'):
            return fstream.buffer
        else:
            return io.BytesIO(fstream.read().encode())
    else:
        return fstream


class OpenFromAny(object):
    """
    Context manager for turning file names, callables that open streams or
    already open streams into a single stream format (binary or text with
    specific encoding) for subsequent reading. The file is left in an
    identical state, including its cursor position, when the context manager
    returns.
    """

    def __init__(self, fobj, mode='r', encoding=None):
        """
        Open file

        Arguments
        ---------
        fobj : str or stream
            The file to be opened, specified either as a file name or a
            stream object.
        mode : str
            Open as text ('r') or binary ('rb'). (Default: None)
        encoding : str
            Character encoding when opening text files. (Default: None)
        """
        self._fobj = fobj
        self._mode = mode
        self._encoding = encoding

    def __enter__(self):
        # Depending from where this function is called, self._fobj might already
        # be a filestream
        self._fstream = None
        self._already_open = False
        self._prior_position = None
        self._need_detach = False
        if hasattr(self._fobj, 'read'):
            # This is a stream that is already open
            self._already_open = True
            if hasattr(self._fobj, 'tell'):
                self._prior_position = self._fobj.tell()
            if self._mode == 'rb':
                # Turn this into a binary stream, if it is a text stream
                self._fstream = _get_binary_stream(self._fobj)
            elif self._mode == 'r':
                # file was opened without the 'b' option, we just need to make sure the encoding is correct
                if isinstance(self._fobj, io.TextIOBase) and (self._encoding is None or
                                                              self._fobj.encoding == self._encoding):
                    self._fstream = self._fobj
                else:
                    self._need_detach = True
                    self._fstream = io.TextIOWrapper(_get_binary_stream(self._fobj), encoding=self._encoding)
            else:
                return ValueError(f"Unknown file open mode '{self._mode}'.")
        elif callable(self._fobj):
            # This is a function that returns the file stream
            fobj = self._fobj()
            if self._mode == 'rb':
                # Turn this into a binary stream, if it is a text stream
                self._fstream = _get_binary_stream(fobj)
            elif self._mode == 'r':
                # file was opened without the 'b' option, we just need to make sure the encoding is correct
                if isinstance(fobj, io.TextIOBase) and (self._encoding is None or fobj.encoding == self._encoding):
                    self._fstream = fobj
                else:
                    self._fstream = io.TextIOWrapper(_get_binary_stream(fobj), encoding=self._encoding)
        else:
            # This is a string, just open the file
            self._fstream = open(self._fobj, mode=self._mode, encoding=self._encoding)
        return self._fstream

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Detach streams so file does not get closed
        if self._need_detach:
            self._fstream.detach()

        # Close file, unless it was already open when this object was constructed
        if self._already_open:
            if self._prior_position is not None:
                self._fobj.seek(self._prior_position)
        else:
            self._fstream.close()

        # Reset internal state
        self._fstream = None
        self._already_open = None
        self._need_detach = False
