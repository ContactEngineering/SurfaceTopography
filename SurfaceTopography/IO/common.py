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

import gzip
import io
import zipfile

###

CHANNEL_NAME_INFO_KEY = 'channel_name'


###

def stream_size(fobj):
    """Total size of a seekable stream in bytes; the position is kept."""
    position = fobj.tell()
    fobj.seek(0, io.SEEK_END)
    size = fobj.tell()
    fobj.seek(position)
    return size


def check_stream_size(fobj, expected_size):
    """
    Raise `CorruptFile` if the stream is shorter than `expected_size` bytes,
    i.e. if the file is truncated.
    """
    size = stream_size(fobj)
    if size < expected_size:
        from ..Exceptions import CorruptFile
        raise CorruptFile(
            f"The file has {size} bytes, but its header describes {expected_size} bytes; it is "
            f"probably truncated."
        )


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def check_hdf5_size(fobj):
    """
    Raise `CorruptFile` if an HDF5 file is shorter than the end-of-file
    address stored in its superblock, i.e. if it is truncated. The
    superblock is searched at offset 0 and at 512, 1024, 2048, ... bytes
    (after a user block, e.g. of MAT-files). Streams without HDF5 signature
    are ignored.
    """
    size = stream_size(fobj)
    location = 0
    while location + 8 <= size:
        fobj.seek(location)
        if fobj.read(8) == _HDF5_SIGNATURE:
            break
        location = 512 if location == 0 else 2 * location
    else:
        fobj.seek(0)
        return
    superblock = fobj.read(56)
    fobj.seek(0)
    version = superblock[0]
    if version in (0, 1):
        size_of_offsets = superblock[5]
        # Base address, free-space info, end-of-file address, driver info
        first_address = 16 if version == 0 else 20
    else:
        size_of_offsets = superblock[1]
        # Base address, superblock extension, end-of-file address
        first_address = 4
    if len(superblock) < first_address + 3 * size_of_offsets:
        from ..Exceptions import CorruptFile
        raise CorruptFile("The file ended within the HDF5 superblock; it is probably truncated.")
    addresses = [int.from_bytes(superblock[first_address + i * size_of_offsets:
                                           first_address + (i + 1) * size_of_offsets], "little")
                 for i in range(3)]
    # The end-of-file address is absolute, i.e. it includes a user block
    _, _, end_of_file_address = addresses
    check_stream_size(fobj, end_of_file_address)


def is_binary_stream(fobj):
    """Check whether fobj is a binary stream"""
    return isinstance(fobj, io.BytesIO) or \
        isinstance(fobj, zipfile.ZipExtFile) or \
        isinstance(fobj, gzip.GzipFile) or \
        (hasattr(fobj, 'mode') and isinstance(fobj.mode, str) and 'b' in fobj.mode)


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
            Open as text ('r') or binary ('rb'). (Default: 'r')
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
                raise ValueError(f"Unknown file open mode '{self._mode}'.")
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
