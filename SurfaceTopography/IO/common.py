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

from io import TextIOBase


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
                if isinstance(self._fobj, TextIOBase):
                    # file was opened without the 'b' option, so read its buffer to get the binary data
                    self._fstream = self._fobj.buffer
                else:
                    self._fstream = self._fobj
            elif self._mode == 'r':
                if isinstance(self._fobj, TextIOBase):
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
