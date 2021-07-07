#
# Copyright 2018-2021 Lars Pastewka
#           2018-2021 Michael Röttger
#           2019-2020 Antoine Sanner
#           2019 Kai Haase
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

"""
SurfaceTopography profile from file input
"""

import xml.etree.ElementTree as ElementTree
from struct import unpack
from zipfile import ZipFile
import numpy as np
import numpy.ma as ma

from .common import CHANNEL_NAME_INFO_KEY
from .Reader import ReaderBase, ChannelInfo, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


###

def binary(func):
    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        try:
            retvals = func(fobj, *args, **kwargs)
        finally:
            if close_file:
                fobj.close()
        return retvals

    return func_wrapper


def mask_undefined(data, maxval=1e32):
    """
    If data contains undefined points, then return a masked array with all
    undefined points masked.

    The following heuristics is applied to identify undefined points:
    - Remove points that are +/-inf or nan
    - Remove points that are >+maxval or <-maxval
    """
    # First, we mask all points that are infinite or nan
    mask = np.logical_and(np.isfinite(data), np.abs(data) < maxval)
    if mask.sum() < len(mask):
        return ma.masked_array(data, mask=np.logical_not(mask))
    else:
        return data


def make_wrapped_reader(reader_func, class_name='WrappedReader', format=None,
                        name=None, description=None):
    class WrappedReader(ReaderBase):
        """
        emulates the new implementation of the readers
        """

        _format = format
        _name = name
        _description = description

        def __init__(self, fobj):
            self._fobj = fobj
            self._file_position = 0
            if hasattr(fobj, 'tell'):
                self._file_position = fobj.tell()
            self._topography = reader_func(fobj)
            if CHANNEL_NAME_INFO_KEY in self._topography.info:
                self._channel_name = self._topography.info[CHANNEL_NAME_INFO_KEY]
                del self._topography.info[CHANNEL_NAME_INFO_KEY]
            else:
                self._channel_name = "Default"

        @property
        def channels(self):
            try:
                height_scale_factor = self._topography.height_scale_factor
            except AttributeError:
                height_scale_factor = None
                # None means: Not available in file

            return [ChannelInfo(
                self, 0,
                name=self._channel_name,
                dim=self._topography.dim,
                unit=self._topography.unit,
                info=self._topography.info,
                nb_grid_pts=self._topography.nb_grid_pts,
                physical_sizes=self._topography.physical_sizes,
                height_scale_factor=height_scale_factor)]

        def topography(self, channel_index=None, physical_sizes=None,
                       height_scale_factor=None, unit=None, info={}, periodic=False,
                       subdomain_locations=None, nb_subdomain_grid_pts=None):
            if channel_index is None:
                channel_index = self._default_channel_index

            if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
                raise RuntimeError(
                    'This reader does not support MPI parallelization.')

            if channel_index != 0:
                raise RuntimeError('Reader supports only a single channel 0.')

            physical_sizes = self._check_physical_sizes(physical_sizes, self._topography.physical_sizes)

            # Rewind to position where the data is. Otherwise this method
            # cannot be called twice.
            if hasattr(self._fobj, 'seek'):
                self._fobj.seek(self._file_position)

            # Read again, but this time with physical_sizes and unit set (if not
            # specified in file)
            reader_kwargs = dict(height_scale_factor=height_scale_factor,
                                 unit=unit, info=info.copy(), periodic=periodic)
            if self._topography.physical_sizes is None:
                # file does not have physical sizes
                reader_kwargs['physical_sizes'] = physical_sizes
                # otherwise we won't add the argument, because that is not allowed any more

            return reader_func(self._fobj, **reader_kwargs)

        channels.__doc__ = ReaderBase.channels.__doc__
        topography.__doc__ = ReaderBase.topography.__doc__

    WrappedReader.__name__ = class_name
    return WrappedReader


def read_x3p(fobj, physical_sizes=None, height_scale_factor=None, unit=None, info={},
             periodic=False):
    """
    Load x3p-file.
    See: http://opengps.eu

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

    # Data types of binary container
    # See: https://sourceforge.net/p/open-gps/mwiki/X3p/
    dtype_map = {'I': np.dtype('<u2'),
                 'L': np.dtype('<u4'),
                 'F': np.dtype('f4'),
                 'D': np.dtype('f8')}

    with ZipFile(fobj, 'r') as x3p:
        xmlroot = ElementTree.parse(x3p.open('main.xml')).getroot()
        record1 = xmlroot.find('Record1')
        record3 = xmlroot.find('Record3')

        if record1 is None:
            raise IOError("'Record1' not found in XML.")
        if record3 is None:
            raise IOError("'Record3' not found in XML.")

        # Parse record1

        feature_type = record1.find('FeatureType')
        if feature_type.text != 'SUR':
            raise ValueError("FeatureType must be 'SUR'.")
        axes = record1.find('Axes')
        cx = axes.find('CX')
        cy = axes.find('CY')
        cz = axes.find('CZ')

        if cx.find('AxisType').text != 'I':
            raise ValueError(
                "CX AxisType is not 'I'. Don't know how to handle "
                "this.")
        if cy.find('AxisType').text != 'I':
            raise ValueError(
                "CY AxisType is not 'I'. Don't know how to handle "
                "this.")
        if cz.find('AxisType').text != 'A':
            raise ValueError(
                "CZ AxisType is not 'A'. Don't know how to handle "
                "this.")

        xinc = float(cx.find('Increment').text)
        yinc = float(cy.find('Increment').text)

        datatype = cz.find('DataType').text
        dtype = dtype_map[datatype]

        # Parse record3
        matrix_dimension = record3.find('MatrixDimension')
        nx = int(matrix_dimension.find('SizeX').text)
        ny = int(matrix_dimension.find('SizeY').text)
        nz = int(matrix_dimension.find('SizeZ').text)

        if nz != 1:
            raise ValueError('Z dimension has extend != 1. Volumetric data '
                             'is not supported.')

        data_link = record3.find('DataLink')
        binfn = data_link.find('PointDataLink').text

        rawdata = x3p.open(binfn).read(nx * ny * dtype.itemsize)
        data = np.frombuffer(rawdata, count=nx * ny * nz, dtype=dtype).reshape(nx, ny).T

    if physical_sizes is not None:
        raise MetadataAlreadyFixedByFile('physical_sizes')
    t = Topography(data, (xinc * nx, yinc * ny), unit=unit, info=info, periodic=periodic)
    if height_scale_factor is not None:
        t = t.scale(height_scale_factor)
    return t


X3PReader = make_wrapped_reader(
    read_x3p, class_name="X3PReader", format='x3p',
    name='XML 3D surface profile (X3P)', description='''
X3P is a container format conforming to the ISO 5436-2 (Geometrical Product
Specifications — Surface texture) standard. The format is defined in ISO
25178 and is a standardized format for the exchange of surface topography
data. The full specification of the format can be found
[here](http://www.opengps.eu/).
''')


@binary
def read_opd(fobj, physical_sizes=None, height_scale_factor=None, unit=None, info={},
             periodic=False):
    """
    Load Wyko Vision OPD file.

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

    if unit is not None:
        raise MetadataAlreadyFixedByFile('unit')

    BLOCK_SIZE = 24

    def read_block(fobj):
        blkname = fobj.read(16).split(b'\0', 1)[0].decode('latin-1')
        blktype, blklen, blkattr = unpack('<hlH', fobj.read(8))
        return blkname, blktype, blklen, blkattr

    # Skip header
    fobj.read(2)

    # Read directory block
    dirname, dirtype, dirlen, dirattr = read_block(fobj)
    if dirname != 'Directory':
        raise IOError("Error reading directory block. "
                      "Header is '{}', expected 'Directory'".format(dirname))
    num_blocks = dirlen // BLOCK_SIZE
    if num_blocks * BLOCK_SIZE != dirlen:
        raise IOError(
            'Directory length is not a multiple of the block physical_sizes.')

    blocks = []
    for i in range(num_blocks - 1):
        blocks += [read_block(fobj)]

    data = None
    nx = None
    ny = None
    pixel_size = 1.0
    aspect = 1.0
    mult = 1.0
    for n, t, L, a in blocks:
        if L <= 0:
            continue
        if n == 'RAW DATA' or n == 'RAW_DATA' or n == 'OPD' or n == 'Raw':
            if data is not None:
                raise IOError('Multiple data blocks encountered.')

            nx, ny, elsize = unpack('<HHH', fobj.read(6))
            if elsize == 1:
                dtype = np.dtype('c')
            elif elsize == 2:
                dtype = np.dtype('<i2')
            elif elsize == 4:
                dtype = np.dtype('f4')
            else:
                raise IOError("Don't know how to handle element of size {}."
                              .format(elsize))
            rawdata = fobj.read(nx * ny * dtype.itemsize)
            data = np.frombuffer(rawdata, count=nx * ny, dtype=dtype)
        elif n == 'Wavelength':
            wavelength, = unpack('<f', fobj.read(4))
        elif n == 'Mult':
            mult, = unpack('<H', fobj.read(2))
        elif n == 'Aspect':
            aspect, = unpack('<f', fobj.read(4))
        elif n == 'Pixel_size':
            pixel_size, = unpack('<f', fobj.read(4))
        else:
            fobj.read(L)

    if data is None:
        raise IOError('No data block encountered.')

    data = mask_undefined(data)
    data.shape = (nx, ny)

    # Height are in nm, width in mm
    if physical_sizes is not None:
        raise MetadataAlreadyFixedByFile('physical_sizes')
    surface = Topography(np.fliplr(data),
                         (nx * pixel_size, ny * pixel_size * aspect),
                         unit='mm', info=info, periodic=periodic)
    height_scale_factor_from_file = wavelength / mult * 1e-6
    if height_scale_factor is not None:
        raise MetadataAlreadyFixedByFile('height_scale_factor')
    surface = surface.scale(height_scale_factor_from_file)
    return surface


OPDReader = make_wrapped_reader(read_opd, class_name="OPDReader", format='opd',
                                name='Wyko OPD', description='''
Files generated by the Vision software of the Bruker Wyko white-light
interferometer.
''')


@binary
def read_hgt(fobj, physical_sizes=None, height_scale_factor=None, unit=None, info={},
             periodic=False):
    """
    Read Shuttle Radar SurfaceTopography Mission (SRTM) topography data
    (.hgt extension).

    Keyword Arguments:
    fobj -- filename or file object
    """
    fobj.seek(0, 2)
    fsize = fobj.tell()
    fobj.seek(0)

    dim = int(np.sqrt(fsize / 2))
    if dim * dim * 2 != fsize:
        raise RuntimeError(
            'File physical_sizes of {0} bytes does not match file '
            'physical_sizes for a map of dimension {1}x{1}.'.format(fsize, dim))
    data = np.fromfile(fobj, dtype=np.dtype('>i2'),
                       count=dim * dim).reshape((dim, dim))

    if physical_sizes is None:
        topography = Topography(data, physical_sizes=data.shape, unit=unit, info=info, periodic=periodic)
    else:
        topography = Topography(data, physical_sizes=physical_sizes, unit=unit, info=info, periodic=periodic)
    if height_scale_factor is not None:
        topography = topography.scale(height_scale_factor)
    return topography


HGTReader = make_wrapped_reader(
    read_hgt, class_name="HGTReader", format='hgt',
    name='NASA shuttle radar topography mission', description='''
Data format of the NASA shuttle radar topography mission that recorded the '
earths topography. More information can be found
[here](https://www2.jpl.nasa.gov/srtm/).
                                ''')
