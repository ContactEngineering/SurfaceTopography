#
# Copyright 2018, 2020 Lars Pastewka
#           2019-2020 Antoine Sanner
#           2018, 2020 Michael Röttger
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

import io
import re
import xml.etree.ElementTree as ElementTree
from io import TextIOWrapper
from struct import unpack
from zipfile import ZipFile
import numpy as np
import numpy.ma as ma

from .Reader import ReaderBase, ChannelInfo
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..NonuniformLineScan import NonuniformLineScan

###

height_units = {'m': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'nm': 1e-9, 'Å': 1e-10}
voltage_units = {'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9}

units = dict(height=height_units, voltage=voltage_units)

CHANNEL_NAME_INFO_KEY = 'channel_name'


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


def text(func):
    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'r', encoding='utf-8')
            fobj_text = fobj
            close_file = True
        elif is_binary_stream(fobj):
            fobj_text = TextIOWrapper(fobj, encoding='utf-8')
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


def is_binary_stream(fobj):
    """

    :param fobj:
    :return:
    """
    return isinstance(fobj, io.BytesIO) or (
            hasattr(fobj, 'mode') and 'b' in fobj.mode)


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


def mangle_height_unit(unit):
    unit = unit.strip()
    if unit == '':
        return None
    elif unit == 'A':
        return 'Å'
    elif unit == 'μm' or unit == 'um' or unit == '~m':
        return 'µm'
    else:
        return unit


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
                self._channel_name = self._topography.info[
                    CHANNEL_NAME_INFO_KEY]
                del self._topography.info[CHANNEL_NAME_INFO_KEY]
            else:
                self._channel_name = "Default"

        @property
        def channels(self):
            return [ChannelInfo(
                self, 0,
                name=self._channel_name,
                dim=self._topography.dim,
                info=self._topography.info,
                nb_grid_pts=self._topography.nb_grid_pts,
                physical_sizes=self._topography.physical_sizes)]

        def topography(self, channel_index=None, physical_sizes=None,
                       height_scale_factor=None, info={}, periodic=False,
                       subdomain_locations=None, nb_subdomain_grid_pts=None):
            if channel_index is None:
                channel_index = self._default_channel_index

            if subdomain_locations is not None or \
                    nb_subdomain_grid_pts is not None:
                raise RuntimeError(
                    'This reader does not support MPI parallelization.')

            if channel_index != 0:
                raise RuntimeError('Reader supports only a single channel 0.')

            # Rewind to position where the data is. Otherwise this method
            # cannot be called twice.
            if hasattr(self._fobj, 'seek'):
                self._fobj.seek(self._file_position)
            return reader_func(self._fobj, physical_sizes=physical_sizes,
                               height_scale_factor=height_scale_factor,
                               info=info.copy(), periodic=periodic)

        channels.__doc__ = ReaderBase.channels.__doc__
        topography.__doc__ = ReaderBase.topography.__doc__

    WrappedReader.__name__ = class_name
    return WrappedReader


@text
def read_matrix(fobj, physical_sizes=None, factor=None, periodic=False):
    """
    Reads a surface profile from a text file and presents in in a
    SurfaceTopography-conformant manner. No additional parsing of
    meta-information is carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    arr = np.loadtxt(fobj)
    if physical_sizes is None:
        surface = Topography(arr, arr.shape, periodic=periodic)
    else:
        surface = Topography(arr, physical_sizes, periodic=periodic)
    if factor is not None:
        surface = surface.scale(factor)
    return surface


MatrixReader = make_wrapped_reader(
    read_matrix, class_name="MatrixReader", format='matrix',
    name='Plain text (matrix)')


@text
def read_asc(fobj, physical_sizes=None, height_scale_factor=None, x_factor=1.0,
             z_factor=None, info={}, periodic=False):
    # pylint: disable=too-many-branches,too-many-statements,invalid-name
    """
    Reads a surface profile (topography) from an generic asc file and presents
    it in a surface-conformant manner. Applies some heuristic to extract
    meta-information for different file formats.

    The info dict of the topography returned is a copy from the given info dict
    and may have some extra keys added:
    - "unit": a common unit for the data, for dimensions and heights
    - "channel_name": the name of the channel (if unknown "Default" is used")

    Keyword Arguments:
    fobj_in -- filename or file object
    unit -- name of surface units, one of m, mm, μm/um, nm, A
    x_factor -- multiplication factor for physical_sizes
    z_factor -- multiplication factor for height
    """
    unit = info['unit'] if 'unit' in info else None

    _float_regex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

    checks = list()
    # Resolution keywords
    checks.append((re.compile(r"\b(?:x-pixels|h)\b\s*=\s*([0-9]+)"), int,
                   "yres"))
    checks.append((re.compile(r"\b(?:y-pixels|w)\b\s*=\s*([0-9]+)"), int,
                   "xres"))

    # Size keywords
    checks.append(
        (re.compile(r"\b(?:x-length|Width|Breite)\b\s*(?:=|\:)\s*(?P<value>" +
                    _float_regex + ")(?P<unit>.*)"), float, "xsiz"))
    checks.append(
        (re.compile(r"\b(?:y-length|Height|Höhe)\b\s*(?:=|\:)\s*(?P<value>" +
                    _float_regex + ")(?P<unit>.*)"), float, "ysiz"))

    # Unit keywords
    checks.append(
        (re.compile(r"\b(?:x-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "xunit"))
    checks.append(
        (re.compile(r"\b(?:y-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "yunit"))
    checks.append(
        (re.compile(r"\b(?:z-unit|Value units)\b\s*(?:=|\:)\s*(\w+)"),
         str, "zunit"))

    # Scale factor keywords
    checks.append(
        (re.compile(r"(?:pixel\s+size)\s*=\s*(?P<value>" + _float_regex +
                    ")(?P<unit>.*)"), float, "xfac"))
    checks.append((
        re.compile((
                r"(?:height\s+conversion\s+factor\s+\(->\s+(?P<unit>.*)\))\s*="
                r"\s*(?P<value>" + _float_regex + ")")),
        float, "zfac"))
    # Channel name keywords
    checks.append((
        re.compile(r"\b(?:Channel|Kanal)\b\s*(?:=|\:)\s*([\w|\s]+)"),
        str, "channel_name"))

    xres = yres = xsiz = ysiz = xunit = yunit = zunit = xfac = yfac = None
    zfac = None
    channel_name = "Default"

    def process_comment(line):
        "Find and interpret known comments in the header of the asc file"
        nonlocal xres, yres, xsiz, ysiz, xunit, yunit, zunit, data, xfac, yfac
        nonlocal zfac
        nonlocal channel_name
        # TODO Is 'fun' needed here? No used so far..
        for reg, fun, key in checks:
            match = reg.search(line)
            if match is None:
                continue
            if key == 'xres':
                xres = int(match.group(1))
            elif key == 'yres':
                yres = int(match.group(1))
            elif key == 'xsiz':
                xsiz = float(match.group('value'))
                x = match.group('unit')
                if x:
                    xunit = mangle_height_unit(x)
            elif key == 'ysiz':
                ysiz = float(match.group('value'))
                y = match.group('unit')
                if y:
                    yunit = mangle_height_unit(y)
            elif key == 'xunit':
                xunit = mangle_height_unit(match.group(1))
            elif key == 'yunit':
                yunit = mangle_height_unit(match.group(1))
            elif key == 'zunit':
                zunit = mangle_height_unit(match.group(1))
            elif key == 'xfac':
                xfac = float(match.group('value'))
                xunit = mangle_height_unit(match.group('unit'))
            elif key == 'zfac':
                zfac = float(match.group('value'))
                zunit = mangle_height_unit(match.group('unit'))
            elif key == 'channel_name':
                channel_name = match.group(1).strip()

    data = []
    for line in fobj:
        line_elements = line.strip().split()
        if len(line_elements) > 0:
            try:
                data += [[float(strval) for strval in line_elements]]
            except ValueError:
                process_comment(line)

    data = np.array(data).T
    nx, ny = data.shape
    if nx == 2 or ny == 2:
        raise Exception(
            "This file has just two rows or two columns and is more likely a "
            "line scan than a map.")
    if xres is not None and xres != nx:
        raise Exception(
            "The number of rows (={}) open_topography from the file '{}' "
            "does not match the nb_grid_pts in the file's metadata (={})."
            .format(nx, fobj, xres))
    if yres is not None and yres != ny:
        raise Exception(
            "The number of columns (={}) open_topography from the file '{}' "
            "does not match the nb_grid_pts in the file's metadata "
            "(={}).".format(ny, fobj, yres))

    # Handle scale factors
    if xfac is not None and yfac is None:
        yfac = xfac
    elif xfac is None and yfac is not None:
        xfac = yfac
    if xfac is not None:
        if xsiz is None:
            xsiz = xfac * nx
        else:
            xsiz *= xfac
    if yfac is not None:
        if ysiz is None:
            ysiz = yfac * ny
        else:
            ysiz *= yfac
    if z_factor is not None:
        zfac = z_factor if zfac is None else zfac * z_factor

    info = info.copy()

    # Handle units -> convert to target unit
    if xunit is None and zunit is not None:
        xunit = zunit
    if yunit is None and zunit is not None:
        yunit = zunit

    if unit is None:
        unit = zunit
    if unit is not None:
        if xunit is not None:
            xsiz *= height_units[xunit] / height_units[unit]
        if yunit is not None:
            ysiz *= height_units[yunit] / height_units[unit]
        if zunit is not None:
            if zfac is None:
                zfac = height_units[zunit] / height_units[unit]
            else:
                zfac *= height_units[zunit] / height_units[unit]
        info['unit'] = unit

    # handle channel name
    # we use the info dict here to transfer the channel name
    info[CHANNEL_NAME_INFO_KEY] = channel_name

    # calculate physical sizes and generate topography
    if xsiz is not None and ysiz is not None and physical_sizes is None:
        physical_sizes = (x_factor * xsiz, x_factor * ysiz)
    if data.shape[0] == 1:
        if physical_sizes is not None and len(physical_sizes) > 1:
            physical_sizes = physical_sizes[0]
        surface = UniformLineScan(data[0, :], physical_sizes, info=info,
                                  periodic=periodic)
    else:
        surface = Topography(data, physical_sizes, info=info,
                             periodic=periodic)
    if height_scale_factor is not None:
        zfac = height_scale_factor
    if zfac is not None and zfac != 1:
        surface = surface.scale(zfac)
    return surface


AscReader = make_wrapped_reader(read_asc, class_name="AscReader", format='asc',
                                name='Plain text (with headers)',
                                description='''
SurfaceTopography data stored in plain text (ASCII) format needs to be stored
in a matrix format. Each row contains the height information for subsequent
points in x-direction separated by a whitespace. The next row belong to the
following y-coordinate. Note that if the file has three or less columns, it
will be interpreted as a topography stored in a coordinate format (the three
columns contain the x, y and z coordinates of the same points). The smallest
topography that can be provided in this format is therefore 4 x 1.

The reader supports parsing file headers for additional metadata. This allows
to specify the physical size of the topography and the unit. In particular, it
supports reading ASCII files exported from SPIP and Gwyddion.

When writing your own ASCII files, we recommend to prepent the header with a
'#'. The following file is an example that contains 4 x 3 data points:
```
# Channel: Main
# Width: 10 µm
# Height: 10 µm
# Value units: m
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
 9.0 10.0 11.0 12.0
```
''')


@text
def read_xyz(fobj, physical_sizes=None, height_scale_factor=None, info={},
             periodic=False, tol=1e-6):
    """
    Load xyz-file. These files contain line scan information in terms of
    (x,y)-positions.

    Parameters
    ----------
    fobj : str or file object
         File name or stream.
    unit : str
         Physical unit.
    tol : float
         Tolerance for detecting uniform grids

    Returns
    -------
    topography : Topography
        SurfaceTopography object.
    """
    # pylint: disable=invalid-name
    data = np.loadtxt(fobj, unpack=True)

    if len(data) == 2:
        # This is a line scan.
        x, z = data
        x -= np.min(x)

        d_uniform = (x[-1] - x[0]) / (len(x) - 1)
        if np.max(np.abs(np.diff(x) - d_uniform)) < tol:
            if physical_sizes is None:
                physical_sizes = d_uniform * len(x)
            t = UniformLineScan(z, physical_sizes, info=info,
                                periodic=periodic)
        else:
            if physical_sizes is not None:
                raise ValueError(
                    'XYZ reader found nonuniform data. Manually setting the'
                    'physical size is not possible for this type of data.')

            t = NonuniformLineScan(x, z, info=info, periodic=periodic)
    elif len(data) == 3:
        # This is a topography map.
        x, y, z = data

        # Sort x-values into bins. Assume points on surface are equally spaced.
        dx = x[1] - x[0]
        binx = np.array(x / dx + 0.5, dtype=int)
        n = np.bincount(binx)
        ny = n[0]
        assert np.all(n == ny)  # FIXME: Turn assert into exception

        # Sort y-values into bins.
        dy = y[binx == 0][1] - y[binx == 0][0]
        biny = np.array(y / dy + 0.5, dtype=int)
        n = np.bincount(biny)
        nx = n[0]
        assert np.all(n == nx)  # FIXME: Turn assert into exception

        # Sort data into bins.
        data = np.zeros((nx, ny))
        data[binx, biny] = z

        # Sanity check. Should be covered by above asserts.
        value_present = np.zeros((nx, ny), dtype=bool)
        value_present[binx, biny] = True
        assert np.all(value_present)  # FIXME: Turn assert into exception

        if physical_sizes is None:
            physical_sizes = (dx * nx, dy * ny)
        t = Topography(data, physical_sizes, info=info, periodic=periodic)
    else:
        raise Exception(
            'Expected two or three columns for topography that is a list of '
            'positions and heights.')

    if height_scale_factor is not None:
        t = t.scale(height_scale_factor)
    return t


XYZReader = make_wrapped_reader(
    read_xyz, class_name="XYZReader", format='xyz',
    name='Plain text (x,y,z coordinates)', description='''
SurfaceTopography information can be provided as coordinate data. This is a
text file that contains either two columns (for line scans) or three columns
(for two-dimensional topographies) of data. The parser does not support
reading header information. Units can therefore not be provided directly
within this file format.

Line scans can be provided on a non-uniform grid. The x-coordinates do not
need to be equally spaced and the surface specified in this format can be
reentrant. The code interprets such topographies as piecewise linear between
the points that are specified in the file.

Two-dimensional topography maps need to reside on a regular grid. The x- and
y-coordinates need to be equally spaced.
''')


def read_x3p(fobj, physical_sizes=None, height_scale_factor=None, info={},
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
        data = np.frombuffer(rawdata, count=nx * ny * nz,
                             dtype=dtype).reshape(nx, ny).T
    if physical_sizes is None:
        physical_sizes = (xinc * nx, yinc * ny)
    t = Topography(data, physical_sizes, info=info, periodic=periodic)
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
def read_opd(fobj, physical_sizes=None, height_scale_factor=None, info={},
             periodic=False):
    """
    Load Wyko Vision OPD file.

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

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
    if physical_sizes is None:
        physical_sizes = (nx * pixel_size, ny * pixel_size * aspect)
    surface = Topography(np.fliplr(data), physical_sizes,
                         info={**info, **dict(unit='mm')}, periodic=periodic)
    if height_scale_factor is None:
        surface = surface.scale(wavelength / mult * 1e-6)
    else:
        surface = surface.scale(height_scale_factor)
    return surface


OPDReader = make_wrapped_reader(read_opd, class_name="OPDReader", format='opd',
                                name='Wyko OPD', description='''
Files generated by the Vision software of the Bruker Wyko white-light
interferometer.
''')


@binary
def read_hgt(fobj, physical_sizes=None, height_scale_factor=None, info={},
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
            'physical_sizes for a map of dimension {1}x{1}.'
            .format(fsize, dim))
    data = np.fromfile(fobj, dtype=np.dtype('>i2'),
                       count=dim * dim).reshape((dim, dim))

    if physical_sizes is None:
        topography = Topography(data, physical_sizes=data.shape, info=info,
                                periodic=periodic)
    else:
        topography = Topography(data, physical_sizes=physical_sizes, info=info,
                                periodic=periodic)
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
