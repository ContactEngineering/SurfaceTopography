
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
            CannotDetectFileFormat()

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