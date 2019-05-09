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
# Old-style readers
from PyCo.Topography.IO.FromFile import DiReader, IbwReader, MatReader, \
    X3pReader, XyzReader, OpdReader, AscReader

# New-style readers
from PyCo.Topography.IO.NPY import NPYReader
from PyCo.Topography.IO.H5 import H5Reader
from PyCo.Topography.IO.OPDx import OPDxReader

from .Reader import UnknownFileFormatGiven, CannotDetectFileFormat, \
    FileFormatMismatch, CorruptFile

readers = {
    'asc': AscReader,
    'di': DiReader,
    'h5': H5Reader,
    'ibw': IbwReader,
    'mat': MatReader,
    'npy': NPYReader,
    'opd': OpdReader,
    'opdx': OPDxReader,
    'x3p': X3pReader,
    'xyz': XyzReader,
}


def detect_format(fobj, comm=None):
    """
    Detect file format based on its content.

    Keyword Arguments:
    fobj : filename or file object
    comm : mpi communicator, optional
    """
    msg = ""
    for name, reader in readers.items():
        try:
            if comm is not None:
                reader(fobj, comm)
            else:
                reader(fobj)
            return name
        except Exception as err:
            msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
        finally:
            if hasattr(fobj, 'seek'):
                # if the reader crashes the cursor in the file-like object
                # has to be set back to the top of the file
                fobj.seek(0)
    raise CannotDetectFileFormat(msg)


def open_topography(fobj, format=None, comm=None):
    r"""
    Returns a reader for the file `fobj``


    Parameters
    ----------
    fobj: str or filelike object
        path of the file or filelike object

    format: str, optional
        specify in which format the file should be interpreted

    comm: MPI communicator or MPIStub.comm, optional
        Only relevant for MPI code. MPI is only supported for `format = "npy"`

    Returns
    -------
    Instance of a `ReaderBase` subclass according to the format

    Examples
    --------
    simplest read workflow
    >>> reader = open_topography("filename")
    >>> top = reader.topography()

    if the file contains several channels you can check their metadata with
    `reader.channels()` (returns a list of dicts containing attributes `size`,
    `unit` and
    `height_scale_factor)
    >>> top = reader.topography(channel=2)

    You can also prescribe some attributes when creating the topography
    >>> top = reader.topography(channel=2, size=(10.,10.), info={"unit":"µm"})
    """
    if comm is not None:
        kwargs = {"comm": comm}
    else:
        kwargs = {}

    if not hasattr(fobj, 'read'):
        if not os.path.isfile(fobj):
            raise FileExistsError("file {} not found".format(fobj))

    if format is None:
        msg = ""
        for name, reader in readers.items():
            try:
                return reader(fobj, **kwargs)
            except Exception as err:
                msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
            finally:
                if hasattr(fobj, 'seek'):
                    # if the reader crashes the cursos in the file-like object
                    # have to be set back to the top of the file
                    fobj.seek(0)
        raise CannotDetectFileFormat(msg)
    else:
        if format not in readers.keys():
            raise UnknownFileFormatGiven("{} not in registered file formats {}".format(fobj, readers.keys()))
        return readers[format](fobj, **kwargs)


def read_topography(fn, *args, **kwargs):
    return open_topography(fn, *args, **kwargs).topography()
