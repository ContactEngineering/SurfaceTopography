#
# Copyright 2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
from PyCo.Topography.IO.FromFile import DiReader, IbwReader, MatReader, \
    X3pReader, XyzReader, OpdReader, AscReader

from PyCo.Topography.IO.NPY import NPYReader
from PyCo.Topography.IO.H5 import H5Reader
from PyCo.Topography.IO.OPDx import OPDxReader

from .Reader import UnknownFileFormatGiven, CannotDetectFileFormat, \
    FileFormatMismatch, CorruptFile

def detect_format(fn, comm=None):
    """
    Detect file format based on its content.

    Keyword Arguments:
    fobj : filename or file object
    comm : mpi communicator, optional
    """

    for name, reader in readers.items():
        try:
            if comm is not None:
                reader(fn, comm)
            else:
                reader(fn)
            return name
        except :
            raise CannotDetectFileFormat()

readers = {
        "asc": AscReader,
        "npy": NPYReader,
        "h5":  H5Reader,
        "OPDx":OPDxReader,
        'di':  DiReader,
        'ibw': IbwReader,
        'mat': MatReader,
        'opd': OpdReader,
        'x3p': X3pReader,
        'xyz': XyzReader
    }

def read(fn, format=None, comm=None):
    """

    Parameters
    ----------
    fn
    format
    comm

    Returns
    -------

    """
    if comm is not None:
        kwargs = {"comm":comm}
    else: kwargs= {}

    if not os.path.isfile(fn):
        raise FileExistsError("file {} not found".format(fn))
    
    if format is None:
        for name, reader in readers.items():
            try:
                return reader(fn, **kwargs)
            except:
                pass
        raise CannotDetectFileFormat()
    else:
        if format not in readers.keys():
            raise UnknownFileFormatGiven("{} not in registered file formats {}".format(fn, readers.keys()))
        return readers[format](fn, **kwargs)