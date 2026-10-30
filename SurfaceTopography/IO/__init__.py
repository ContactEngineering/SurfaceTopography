#
# Copyright 2019-2024 Lars Pastewka
#           2022 Johannes Hörmann
#           2020-2021 Michael Röttger
#           2019 Kai Haase
#           2019 Antoine Sanner
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

import inspect
import os

# Registers DZI writers with Topography class
import SurfaceTopography.IO.DZI  # noqa: F401

from ..Exceptions import UnknownFileFormat  # noqa: F401
from ..Exceptions import (  # noqa: F401
    CannotDetectFileFormat,
    CorruptFile,
    MetadataAlreadyFixedByFile,
    ReadFileError,
)

from .AL3D import AL3DReader
from .BCR import BCRReader
from .DATX import DATXReader
from .DI import DIReader
from .EZD import EZDReader
from .FromFile import HGTReader
from .FRT import FRTReader
from .GWY import GWYReader
from .H5 import H5Reader
from .IBW import IBWReader
from .JPK import JPKReader
from .LEXT import LEXTReader
from .Matlab import MatReader
from .MDT import MDTReader
from .MetroPro import MetroProReader
from .MI import MIReader
from .Mitutoyo import MitutoyoReader
from .MNT import MNTReader
from .NC import NCReader
from .NMM import NMMReader
from .NMS import NMSReader
from .NPY import NPYReader, NPZReader
from .OIR import OIRReader, POIRReader
from .OPD import OPDReader
from .OPDx import OPDxReader
from .OS3D import OS3DReader
from .PLU import PLUReader
from .PLUX import PLUXReader
from .PS import PSReader
from .QEP import QEPReader
from .Reader import MagicMatch, ReaderBase  # noqa: F401
from .SDF import SDFReader
from .SUR import SURReader
from .Text import AscReader
from .TMD import TMDReader
from .VK import VKReader
from .WSXM import WSXMReader
from .X3P import X3PReader
from .XYZ import XYZReader
from .ZMG import ZMGReader
from .ZON import ZONReader

readers = [
    # SDFReader must come before ASC, because SDF ASCII is readable as ASC
    SDFReader,
    # XYZ must come before ASC, because 2D XYZ is a specialized ASC
    XYZReader,
    AscReader,
    DIReader,
    MatReader,
    MDTReader,
    OPDReader,
    OPDxReader,
    X3PReader,
    IBWReader,
    MIReader,
    MitutoyoReader,
    # NCReader must come before H5Reader, because NC4 *is* a specialized form of HDF5
    NCReader,
    # DATXReader must come before H5Reader, because DATX *is* a specialized form of HDF5
    DATXReader,
    H5Reader,
    NPYReader,
    PSReader,
    SURReader,
    TMDReader,
    VKReader,
    ZMGReader,
    ZONReader,
    OS3DReader,
    AL3DReader,
    EZDReader,
    BCRReader,
    MetroProReader,
    GWYReader,
    PLUReader,
    FRTReader,
    LEXTReader,
    OIRReader,
    POIRReader,
    WSXMReader,
    PLUXReader,
    JPKReader,
    MNTReader,
    QEPReader,
    NMMReader,
    # NPZReader must come after the readers of other ZIP-based formats,
    # because it may accept any ZIP archive
    NPZReader,
    # NMS and HGT readers should come last as there is no file magic
    NMSReader,
    HGTReader,
]

lookup_reader_by_format = {}
for reader in readers:
    lookup_reader_by_format[reader.format()] = reader


# Buffer size for magic-based format detection
MAGIC_BUFFER_SIZE = 512


def _read_magic_buffer(fobj):
    """
    Read the first N bytes of a file for magic-based format detection.

    Parameters
    ----------
    fobj : str, file-like object, or callable
        File path, file object, or callable that returns a file object.

    Returns
    -------
    bytes
        First MAGIC_BUFFER_SIZE bytes of the file.
    """
    # Handle callable (e.g., CEFileOpener that returns file handle when called)
    if callable(fobj) and not hasattr(fobj, 'read'):
        try:
            f = fobj()
            buffer = f.read(MAGIC_BUFFER_SIZE)
            f.close()
            return buffer if isinstance(buffer, bytes) else b''
        except Exception:
            return b''  # Fall back to full parsing on any error

    if hasattr(fobj, 'read'):
        # File-like object
        # Check if file is in text mode - if so, we can't read binary magic
        # Note: gzip.GzipFile has mode as an int, not a string
        if hasattr(fobj, 'mode') and isinstance(fobj.mode, str) and 'b' not in fobj.mode:
            return b''  # Return empty buffer, fall back to full parsing
        pos = fobj.tell() if hasattr(fobj, 'tell') else 0
        try:
            buffer = fobj.read(MAGIC_BUFFER_SIZE)
            # If read returns a string, file is in text mode
            if isinstance(buffer, str):
                if hasattr(fobj, 'seek'):
                    fobj.seek(pos)
                return b''  # Return empty buffer, fall back to full parsing
        except UnicodeDecodeError:
            # Binary content in text-mode file
            if hasattr(fobj, 'seek'):
                fobj.seek(pos)
            return b''  # Return empty buffer, fall back to full parsing
        if hasattr(fobj, 'seek'):
            fobj.seek(pos)
        return buffer
    else:
        # File path (string or PathLike)
        try:
            with open(fobj, 'rb') as f:
                return f.read(MAGIC_BUFFER_SIZE)
        except (TypeError, OSError):
            return b''  # Fall back to full parsing on any error


def detect_format(fobj, comm=None):
    """
    Detect file format based on its content.

    The readers are tried in the order of the list `readers`. A reader is
    skipped if the first bytes of the file rule out its format
    (`can_read` returns `MagicMatch.NO`); otherwise it attempts to open the
    file, and the format of the first reader that succeeds is returned.

    Arguments
    ---------
    fobj : str, file-like object or callable
        Path of the file, a file-like object or a callable that returns a
        file-like object when called.
    comm : mpi communicator, optional
        Communicator passed to the readers. (Default: None)

    Returns
    -------
    format : str
        Format identifier of the file.

    Raises
    ------
    CannotDetectFileFormat
        If no registered reader can open the file.
    """
    # Read magic buffer once for fast pre-filtering
    magic_buffer = _read_magic_buffer(fobj)

    msg = ""
    for reader in readers:
        # Fast magic-based rejection
        magic_result = reader.can_read(magic_buffer)
        if magic_result == MagicMatch.NO:
            continue  # Skip this reader

        # Try full instantiation
        try:
            if comm is not None:
                reader(fobj, comm)
            else:
                reader(fobj)
            return reader.format()
        except Exception as err:
            msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
        finally:
            if hasattr(fobj, 'seek'):
                # if the reader crashes the cursor in the file-like object
                # has to be set back to the top of the file
                fobj.seek(0)
    raise CannotDetectFileFormat(msg)


def open_topography(fobj, format=None, communicator=None):
    r"""
    Open a topography file and return a reader object for it. The reader
    gives access to the channels in the file and their metadata (number of
    grid points, physical sizes, units, etc.) without reading the height data
    into memory. The data is read when calling `topography` on the reader or
    on one of its channels.

    Parameters
    ----------
    fobj : str, file-like object or callable
        Path of the file, a (binary) file-like object or a callable that
        returns a file-like object when called.
    format : str, optional
        Format identifier (e.g. 'di', 'sdf'; see `readers` for the list of
        registered readers). (Default: None, which means the format is
        detected from the file contents)
    communicator : mpi4py or NuMPI communicator object, optional
        MPI communicator handling inter-process communication. It is passed
        to readers that support parallel I/O. If the communicator has more
        than one process, readers without parallel I/O support are rejected:
        with an explicit `format`, a `ValueError` is raised; during format
        detection, the reader is skipped. (Default: None)

    Returns
    -------
    reader : :class:`SurfaceTopography.IO.Reader.ReaderBase`
        Instance of the reader subclass for the format of the file.

    Raises
    ------
    FileNotFoundError
        If `fobj` is a path and the file does not exist.
    CannotDetectFileFormat
        If no registered reader can open the file.
    UnknownFileFormat
        If `format` is not a registered format identifier.

    Examples
    --------
    Simplest read workflow, which returns the default channel:

    >>> reader = open_topography("filename")
    >>> top = reader.topography()

    The channels and their metadata are listed in `reader.channels`, a list
    of :class:`SurfaceTopography.IO.Reader.ChannelInfo` objects:

    >>> for channel in reader.channels:
    ...     print(channel.name, channel.nb_grid_pts, channel.physical_sizes,
    ...           channel.unit)

    A specific channel is loaded by its index in this list:

    >>> top = reader.topography(channel_index=1)

    Metadata that is not stored in the file can be specified when reading
    the topography:

    >>> top = reader.topography(physical_sizes=(10., 10.), unit="µm")

    Specifying metadata that is stored in the file raises
    `MetadataAlreadyFixedByFile`.
    """  # noqa: E501
    def reader_accepts_communicator(reader_class):
        """Check if a reader's __init__ accepts a 'communicator' argument."""
        try:
            sig = inspect.signature(reader_class.__init__)
            return 'communicator' in sig.parameters
        except (ValueError, TypeError):
            return False

    def check_parallel_support(reader_class):
        """Raise an error if parallel I/O is requested but not supported."""
        if communicator is not None and communicator.size > 1:
            if not reader_accepts_communicator(reader_class):
                raise ValueError(
                    f"{reader_class.__name__} does not support parallel I/O, "
                    f"but communicator has size {communicator.size}."
                )

    if not hasattr(fobj, 'read') and not callable(fobj):  # fobj is a path
        if not os.path.isfile(fobj):
            raise FileNotFoundError("file {} not found".format(fobj))

    if format is None:
        # Read magic buffer once for fast pre-filtering
        magic_buffer = _read_magic_buffer(fobj)

        msg = ""
        for reader in readers:
            # Fast magic-based rejection
            magic_result = reader.can_read(magic_buffer)
            if magic_result == MagicMatch.NO:
                continue  # Skip this reader

            kwargs = {}
            if communicator is not None and reader_accepts_communicator(reader):
                kwargs["communicator"] = communicator
            try:
                r = reader(fobj, **kwargs)
                check_parallel_support(reader)
                return r
            except Exception as err:
                msg += "tried {}: \n {}\n\n".format(reader.__name__, err)
            finally:
                if hasattr(fobj, 'seek'):
                    # if the reader crashes the cursor in the file-like object
                    # has to be set back to the top of the file
                    fobj.seek(0)
        raise CannotDetectFileFormat(msg)
    else:
        if format not in lookup_reader_by_format.keys():
            raise UnknownFileFormat(
                f"{format} not in registered file formats {lookup_reader_by_format.keys()}.")
        reader = lookup_reader_by_format[format]
        check_parallel_support(reader)
        kwargs = {}
        if communicator is not None and reader_accepts_communicator(reader):
            kwargs["communicator"] = communicator
        return reader(fobj, **kwargs)


def read_topography(fn, format=None, communicator=None, **kwargs):
    r"""
    Read a topography from a file. If there are multiple data channels in
    the file, the default channel is returned unless a channel is selected.
    The default channel depends on the file format; see the documentation of
    the respective reader.

    This is a shortcut for `open_topography(fn, ...).topography(**kwargs)`.

    Parameters
    ----------
    fn : str, file-like object or callable
        Path of the file, a (binary) file-like object or a callable that
        returns a file-like object when called.
    format : str, optional
        Format identifier. (Default: None, which means the format is
        detected from the file contents)
    communicator : mpi4py or NuMPI communicator object, optional
        MPI communicator handling inter-process communication.
        (Default: None)
    channel_index : int, optional
        Index of the channel to load in the list `reader.channels`.
        (Default: None, which loads the default channel)
    physical_sizes : tuple of floats, optional
        Physical size of the topography. It is necessary to specify this
        if no physical size is found in the data file. If there is a
        physical size in the file, then specifying this parameter raises
        `MetadataAlreadyFixedByFile`.
    height_scale_factor : float, optional
        Factor by which the heights are multiplied, if not found in the data
        file.
    unit : str, optional
        Length unit, if not found in the data file.
    info : dict, optional
        This dictionary will be appended to the info dictionary returned
        by the reader.
    periodic : bool, optional
        Whether the topography should be interpreted as one period of a
        periodic surface. This affects the PSD and autocorrelation
        calculations (windowing).
    subdomain_locations : tuple of ints, optional
        Origin (location) of the subdomain handled by the present MPI process.
    nb_subdomain_grid_pts : tuple of ints, optional
        Number of grid points within the subdomain handled by the present
        MPI process.
    **kwargs
        All keyword arguments other than `format` and `communicator` are
        passed to the `topography` method of the reader. Some readers
        additionally accept `channel_id` and `height_channel_index`.

    Returns
    -------
    topography : subclass of :obj:`HeightContainer`
        The object containing the actual topography data.

    Raises
    ------
    MetadataAlreadyFixedByFile
        Raised if `physical_sizes`, `unit` or `height_scale_factor` are
        given although they are already defined in the file.
    """
    with open_topography(fn, format=format,
                         communicator=communicator) as reader:
        t = reader.topography(**kwargs)
    return t
