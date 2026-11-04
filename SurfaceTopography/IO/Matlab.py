#
# Copyright 2019-2023 Lars Pastewka
#           2021 Michael Röttger
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

import struct

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

from ..Exceptions import CorruptFile, FileFormatMismatch
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny, check_hdf5_size, check_stream_size, stream_size
from .Reader import ReaderBase, ChannelInfo

# MATLAB classes that hold numerical data that can be interpreted as heights.
# Other classes (char, cell, struct, sparse, objects, function handles) are
# ignored.
_NUMERIC_MATLAB_CLASSES = {
    "double",
    "single",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "logical",
}

# Version 7.3 MAT-files are HDF5 files with a 512-byte MATLAB header
_MAT73_MAGIC = b"MATLAB 7.3 MAT-file"


def _is_2d_array(shape):
    try:
        nx, ny = shape
    except (TypeError, ValueError):
        return False
    return nx > 0 and ny > 0


# Data type of compressed data elements of version 5 MAT-files
_MI_COMPRESSED = 15


# Size of the header of version 5 (and later) MAT-files
_MAT5_HEADER_SIZE = 128


def _check_mat5_size(f):
    """
    Raise `CorruptFile` if a version 5 MAT-file ends within its header or if
    a top-level data element extends beyond the end of the file (scipy
    raises IndexError or OSError then). The file consists of a 128-byte
    header and data elements with an 8-byte tag (data type, number of
    bytes). The data of an element is padded to a multiple of 8 bytes,
    except for compressed elements, which are not padded, so that the
    elements after a compressed one may start at unaligned offsets.
    """
    header = f.read(_MAT5_HEADER_SIZE)
    f.seek(0)
    if not header.startswith(b"MATLAB"):
        return  # Not a MAT-file of version 5 or later; scipy reports this
    if len(header) < _MAT5_HEADER_SIZE:
        raise CorruptFile("The file ended within the MAT-file header; it is probably truncated.")
    if header[126:128] not in (b"IM", b"MI"):
        return  # Unknown byte order indicator; scipy reports this
    byte_order = "<" if header[126:128] == b"IM" else ">"
    size = stream_size(f)
    offset = _MAT5_HEADER_SIZE
    while offset < size:
        f.seek(offset)
        tag = f.read(8)
        if len(tag) < 8:
            raise CorruptFile("The file ended within the tag of a data element; it is probably truncated.")
        data_type, nb_bytes = struct.unpack(byte_order + "II", tag)
        if data_type >> 16:
            # Small data element: the data is stored within the tag
            offset += 8
            continue
        end = offset + 8 + nb_bytes
        check_stream_size(f, end, size=size)
        offset = end if data_type == _MI_COMPRESSED else end + (-nb_bytes % 8)
    f.seek(0)


class MatReader(ReaderBase):
    _format = 'mat'
    _mime_types = ['application/x-matlab-data']
    _file_extensions = ['mat']

    _name = 'MATLAB'
    _description = '''
Imports topography data stored in MATLAB workspace files (including the
HDF5-based version 7.3 files). The reader automatically extracts all
two-dimensional numerical arrays stored in the file and interprets those as
height information. The first (row) index of the MATLAB matrix runs along x,
the second (column) index along y. Matlab files do not store units or physical
sizes. These need to be manually provided by the user.
    '''

    def __init__(self, fobj):
        """
        Reads a surface profile from a Matlab file and presents it in a
        SurfaceTopography-conformant manner.

        All two-dimensional numerical arrays present in the matlab data file
        are returned.

        Parameters
        ----------
        fobj : filename or file object
             File to read.
        """
        self._fobj = fobj
        self._channels = []
        with OpenFromAny(self._fobj, 'rb') as f:
            self._hdf5 = f.read(len(_MAT73_MAGIC)) == _MAT73_MAGIC
            f.seek(0)
            if self._hdf5:
                header = self._whosmat_hdf5(f)
            else:
                _check_mat5_size(f)
                header = whosmat(f)  # Only read header
        for name, shape, data_class in header:
            if data_class not in _NUMERIC_MATLAB_CLASSES or not _is_2d_array(shape):
                continue
            channel_info = ChannelInfo(self,
                                       len(self._channels),
                                       name=name,
                                       dim=len(shape),
                                       uniform=True,
                                       nb_grid_pts=shape)
            # no height scale factor given in mat file

            self._channels.append(channel_info)

    @staticmethod
    def _whosmat_hdf5(f):
        """
        Equivalent of `scipy.io.whosmat` for version 7.3 (HDF5) MAT-files.
        Returns a list of (name, shape, class) tuples for all top-level
        variables. MATLAB stores matrices in column-major order, hence the
        shape of the HDF5 dataset is the transpose of the MATLAB shape.
        """
        header = []
        # The HDF5 data follows the 512-byte user block with the MATLAB header
        if stream_size(f) < 512 + 8:
            raise CorruptFile("The file ended within the MAT-file header; it is probably truncated.")
        check_hdf5_size(f)
        try:
            h5 = h5py.File(f, 'r')
        except OSError:
            raise FileFormatMismatch("MAT-file 7.3 does not contain HDF5 data.")
        with h5:
            for name, node in h5.items():
                if not isinstance(node, h5py.Dataset):
                    # Structures and sparse matrices are stored as groups
                    continue
                data_class = node.attrs.get("MATLAB_class", b"")
                if isinstance(data_class, bytes):
                    data_class = data_class.decode("ascii", errors="replace")
                if "MATLAB_empty" in node.attrs or node.dtype.fields is not None:
                    # Empty arrays store their dimensions as data; complex
                    # arrays are stored as compound (real, imag) types
                    continue
                header += [(name, tuple(int(n) for n in node.shape[::-1]), data_class)]
        return header

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        name = self.channels[channel_index].name

        info = info.copy()

        with OpenFromAny(self._fobj, 'rb') as f:
            if self._hdf5:
                with h5py.File(f, 'r') as h5:
                    # Transpose from column-major storage to MATLAB shape
                    heights = np.asarray(h5[name][...]).T
            else:
                heights = loadmat(f, variable_names=[name])[name]

        topography = Topography(
            heights, physical_sizes=self._check_physical_sizes(physical_sizes), unit=unit,
            info=info, periodic=periodic)

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
