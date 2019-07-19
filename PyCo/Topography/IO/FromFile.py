#
# Copyright 2019 Lars Pastewka
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

"""
Topography profile from file input
"""

import io
import re
import xml.etree.ElementTree as ElementTree
from io import TextIOWrapper
from struct import unpack
from zipfile import ZipFile
import copy
import numpy as np

from PyCo.Topography.UniformLineScanAndTopography import Topography, UniformLineScan
from PyCo.Topography.NonuniformLineScan import NonuniformLineScan
from PyCo.Topography.IO.Reader import ReaderBase

###

height_units = {'m': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'nm': 1e-9, 'Å': 1e-10}
voltage_units = {'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9}

units = dict(height=height_units, voltage=voltage_units)

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
    return isinstance(fobj, io.BytesIO) or (hasattr(fobj, 'mode') and 'b' in fobj.mode)

###

def get_unit_conversion_factor(unit1_str, unit2_str):
    """
    Compute factor for conversion from unit1 to unit2. Return None if units are
    incompatible.
    """
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


def make_wrapped_reader(reader_func, name="wrappedReader"):
    class wrappedReader(ReaderBase):
        """
        emulates the new implementation of the readers
        """
        def __init__(self, fn):
            self._topography = reader_func(fn)

            super().__init__(nb_grid_pts=self._topography.nb_grid_pts,
                             physical_sizes=self._topography.physical_sizes,
                             info=self._topography.info)

        def topography(self, physical_sizes=None, info = {}):
            size = self._process_size(physical_sizes)
            info = self._process_info(info)
            if self._topography.is_uniform:
                if self._topography.dim == 2:
                    return Topography(self._topography.heights(), physical_sizes=size, info=info)
                elif self._topography.dim == 1:
                    return UniformLineScan(self._topography.heights(), physical_sizes=size, info=info)
            else:
                top = copy.copy(self._topography)
                top._info = info

                return NonuniformLineScan(self._topography.positions(), self._topography.heights())
    wrappedReader.__name__=name
    return wrappedReader

@text
def read_matrix(fobj, physical_sizes=None, factor=None):
    """
    Reads a surface profile from a text file and presents in in a
    Topography-conformant manner. No additional parsing of meta-information is
    carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    arr = np.loadtxt(fobj)
    if physical_sizes is None:
        surface = Topography(arr, arr.shape)
    else:
        surface = Topography(arr, physical_sizes)
    if factor is not None:
        surface = surface.scale(factor)
    return surface
MatrixReader = make_wrapped_reader(read_matrix, name="MatrixReader")

@text
def read_asc(fobj, physical_sizes=None, unit=None, x_factor=1.0, z_factor=None):
    # pylint: disable=too-many-branches,too-many-statements,invalid-name
    """
    Reads a surface profile from an generic asc file and presents it in a
    surface-conformant manner. Applies some heuristic to extract
    meta-information for different file formats.

    Keyword Arguments:
    fobj_in -- filename or file object
    unit -- name of surface units, one of m, mm, μm/um, nm, A
    x_factor -- multiplication factor for physical_sizes
    z_factor -- multiplication factor for height
    """

    _float_regex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

    checks = list()
    # Resolution keywords
    checks.append((re.compile(r"\b(?:x-pixels|h)\b\s*=\s*([0-9]+)"), int,
                   "xres"))
    checks.append((re.compile(r"\b(?:y-pixels|w)\b\s*=\s*([0-9]+)"), int,
                   "yres"))

    # Size keywords
    checks.append((re.compile(r"\b(?:x-length|Width)\b\s*(?:=|\:)\s*(?P<value>" +
                              _float_regex + ")(?P<unit>.*)"), float, "xsiz"))
    checks.append((re.compile(r"\b(?:y-length|Height)\b\s*(?:=|\:)\s*(?P<value>" +
                              _float_regex + ")(?P<unit>.*)"), float, "ysiz"))

    # Unit keywords
    checks.append((re.compile(r"\b(?:x-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "xunit"))
    checks.append((re.compile(r"\b(?:y-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "yunit"))
    checks.append((re.compile(r"\b(?:z-unit|Value units)\b\s*(?:=|\:)\s*(\w+)"),
                   str, "zunit"))

    # Scale factor keywords
    checks.append((re.compile(r"(?:pixel\s+size)\s*=\s*(?P<value>" + _float_regex +
                              ")(?P<unit>.*)"), float, "xfac"))
    checks.append((re.compile(
        (r"(?:height\s+conversion\s+factor\s+\(->\s+(?P<unit>.*)\))\s*=\s*(?P<value>" +
         _float_regex + ")")),
                   float, "zfac"))

    xres = yres = xsiz = ysiz = xunit = yunit = zunit = xfac = yfac = None
    zfac = None

    def process_comment(line):
        "Find and interpret known comments in the header of the asc file"
        nonlocal xres, yres, xsiz, ysiz, xunit, yunit, zunit, data, xfac, yfac
        nonlocal zfac
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

    data = []
    for line in fobj:
        line_elements = line.strip().split()
        if len(line) > 0:
            try:
                dummy = float(line_elements[0])
                data += [[float(strval) for strval in line_elements]]
            except ValueError:
                process_comment(line)

    data = np.array(data)
    nx, ny = data.shape
    if nx == 2 or ny == 2:
        raise Exception("This file has just two rows or two columns and is more likely a line scan than a map.")
    if xres is not None and xres != nx:
        raise Exception(
            "The number of rows (={}) open_topography from the file '{}' does "
            "not match the nb_grid_pts in the file's metadata (={})."
                .format(nx, fobj, xres))
    if yres is not None and yres != ny:
        raise Exception("The number of columns (={}) open_topography from the file '{}' "
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
        zfac = z_factor if zfac is None else zfac*z_factor

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

    if xsiz is not None and ysiz is not None and physical_sizes is None:
        physical_sizes = (x_factor * xsiz, x_factor * ysiz)
    if physical_sizes is None:
        physical_sizes = data.shape
    if data.shape[1] == 1:
        if physical_sizes is not None and len(physical_sizes) > 1:
            physical_sizes = physical_sizes[0]
        surface = UniformLineScan(data[:, 0], physical_sizes, info=dict(unit=unit))
    else:
        surface = Topography(data, physical_sizes, info=dict(unit=unit))
    if zfac is not None:
        surface = surface.scale(zfac)
    return surface
AscReader = make_wrapped_reader(read_asc, name="AscReader")

@text
def read_xyz(fobj, unit=None, tol=1e-6):
    """
    Load xyz-file. These files contain line scan information in terms of (x,y)-positions.

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
        Topography object.
    """
    # pylint: disable=invalid-name
    data = np.loadtxt(fobj, unpack=True)

    if len(data) == 2:
        # This is a line scan.
        x, z = data
        x -= np.min(x)

        d_uniform = (x[-1] - x[0])/(len(x) - 1)
        if np.max(np.abs(np.diff(x) - d_uniform)) < tol:
            return UniformLineScan(z, d_uniform*len(x), info=dict(unit=unit))
        else:
            return NonuniformLineScan(x, z, info=dict(unit=unit))
    elif len(data) == 3:
        # This is a topography map.
        x, y, z = data

        # Sort x-values into bins. Assume points on surface are equally spaced.
        dx = x[1] - x[0]
        binx = np.array(x / dx + 0.5, dtype=int)
        n = np.bincount(binx)
        ny = n[0]
        assert np.all(n == ny) # FIXME: Turn assert into exception

        # Sort y-values into bins.
        dy = y[binx == 0][1] - y[binx == 0][0]
        biny = np.array(y / dy + 0.5, dtype=int)
        n = np.bincount(biny)
        nx = n[0]
        assert np.all(n == nx) # FIXME: Turn assert into exception

        # Sort data into bins.
        data = np.zeros((nx, ny))
        data[binx, biny] = z

        # Sanity check. Should be covered by above asserts.
        value_present = np.zeros((nx, ny), dtype=bool)
        value_present[binx, biny] = True
        assert np.all(value_present) # FIXME: Turn assert into exception

        return Topography(data, (dx * nx, dy * ny), info=dict(unit=unit))
    else:
        raise Exception('Expected two or three columns for topgraphy that is a list of positions and heights.')
XYZReader = make_wrapped_reader(read_xyz, name="XyzReader")

def read_x3p(fobj):
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
            raise ValueError("CX AxisType is not 'I'. Don't know how to handle "
                             "this.")
        if cy.find('AxisType').text != 'I':
            raise ValueError("CY AxisType is not 'I'. Don't know how to handle "
                             "this.")
        if cz.find('AxisType').text != 'A':
            raise ValueError("CZ AxisType is not 'A'. Don't know how to handle "
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
            raise ValueError('Z dimension has extend != 1. Volumetric data is '
                             'not supported.')

        data_link = record3.find('DataLink')
        binfn = data_link.find('PointDataLink').text

        rawdata = x3p.open(binfn).read(nx * ny * dtype.itemsize)
        data = np.frombuffer(rawdata, count=nx * ny * nz,
                             dtype=dtype).reshape(nx, ny).T

    return Topography(data, (xinc * nx, yinc * ny))
X3PReader = make_wrapped_reader(read_x3p, name="X3pReader")

@binary
def read_mat(fobj, physical_sizes=None, factor=None, unit=None):
    """
    Reads a surface profile from a matlab file and presents in in a
    Topography-conformant manner.

    All two-dimensional arrays present in the matlab data file are returned.

    Keyword Arguments:
    fobj -- filename or file object
    physical_sizes -- physical_sizes of the surface
    factor -- scaling factor for height
    unit -- physical_sizes and height unit
    """
    from scipy.io import loadmat
    data = loadmat(fobj)
    surfaces = []
    for key, value in data.items():
        is_2darray = False
        try:
            nx, ny = value.shape
            is_2darray = True
        except (AttributeError, ValueError):
            pass
        if is_2darray:
            if physical_sizes is None:
                surface = Topography(value, value.shape, info=dict(unit=unit))
            else:
                surface = Topography(value, physical_sizes, info=dict(unit=unit))
            if factor is not None:
                surface = surface.scale(factor)
            surfaces += [surface]
    if len(surfaces) == 1:
        return surfaces[0]
    else:
        return surfaces

class MatReader(ReaderBase):
    def __init__(self, fobj):
        """
            Reads a surface profile from a matlab file and presents in in a
            Topography-conformant manner.

            All two-dimensional arrays present in the matlab data file are returned.

            Parameters
            ----------

            fobj: filename or file object

        """
        super().__init__()
        from scipy.io import loadmat

        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        try:
            data = loadmat(fobj)
            surfaces = []
            self._channels = []
            self._height_data=[]
            for key, value in data.items():
                is_2darray = False
                try:
                    nx, ny = value.shape
                    is_2darray = True
                except (AttributeError, ValueError):
                    pass
                if is_2darray:
                    channelinfo = {"name": key,
                                   "nb_grid_pts":value.shape,
                                   "height_scale_factor": 1.,
                                   "unit": "",
                                   "physical_sizes": None}

                    self._channels.append(channelinfo)
                    self._height_data.append(value)
        finally:
            if close_file:
                fobj.close()

    @property
    def channels(self):
        return self._channels

    @property
    def nb_grid_pts(self, channel=None):
        if channel is None:
            channel = self._default_channel
        return self.channels[channel]["nb_grid_pts"]

    @property
    def physical_sizes(self, channel=None):
        if channel is None:
            channel = self._default_channel

        return self.channels[channel]["physical_sizes"]

    def topography(self, channel=None, physical_sizes=None, info={}):
        if channel is None:
            channel=self._default_channel
        info_dict = dict(data_source=self.channels[channel]["name"],
                         unit=self.channels[channel]["unit"])
        info_dict.update(info)
        return Topography(self._height_data[channel],
                          physical_sizes=self._process_size(physical_sizes),
                          info=info_dict)

@binary
def read_opd(fobj):
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

    # Header
    tmp = fobj.read(2)

    # Read directory block
    dirname, dirtype, dirlen, dirattr = read_block(fobj)
    if dirname != 'Directory':
        raise IOError("Error reading directory block. "
                      "Header is '{}', expected 'Directory'".format(dirname))
    num_blocks = dirlen // BLOCK_SIZE
    if num_blocks * BLOCK_SIZE != dirlen:
        raise IOError('Directory length is not a multiple of the block physical_sizes.')

    blocks = []
    for i in range(num_blocks - 1):
        blocks += [read_block(fobj)]

    data = None
    nx = None
    ny = None
    pixel_size = 1.0
    aspect = 1.0
    mult = 1.0
    for n, t, l, a in blocks:
        if l <= 0:
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
                raise IOError("Don't know how to handle element physical_sizes {}."
                              .format(elsize))
            rawdata = fobj.read(nx * ny * dtype.itemsize)
            data = np.frombuffer(rawdata, count=nx * ny,
                                 dtype=dtype).reshape(nx, ny)
        elif n == 'Wavelength':
            wavelength, = unpack('<f', fobj.read(4))
        elif n == 'Mult':
            mult, = unpack('<H', fobj.read(2))
        elif n == 'Aspect':
            aspect, = unpack('<f', fobj.read(4))
        elif n == 'Pixel_size':
            pixel_size, = unpack('<f', fobj.read(4))
        else:
            fobj.read(l)

    if data is None:
        raise IOError('No data block encountered.')

    # Height are in nm, width in mm
    surface = Topography(data, (nx * pixel_size, ny * pixel_size * aspect), info=dict(unit='mm'))
    surface = surface.scale(wavelength / mult * 1e-6)
    return surface
OPDReader = make_wrapped_reader(read_opd, name="OpdReader")

@binary
def read_ibw(fobj):
    """
    Read IGOR Binary Wave files.

    Keyword Arguments:
    fobj -- filename or file object
    """
    from igor.binarywave import load

    wave = load(fobj)['wave']

    channel = 0
    data = wave['wData'][:, :, channel].copy()
    # This is just a wild guess...
    z_unit = wave['wave_header']['dataUnits'][channel].decode('latin-1')
    xy_unit = wave['wave_header']['dimUnits'][channel, channel].decode('latin-1')
    assert z_unit == xy_unit

    sfA = wave['wave_header']['sfA']
    nx, ny = data.shape

    surface = Topography(data, (nx * sfA[0], ny * sfA[1]), info=dict(unit=z_unit))

    return surface
IBWReader = make_wrapped_reader(read_ibw, name="IbwReader")

@binary
def read_hgt(fobj, size=None):
    """
    Read Shuttle Radar Topography Mission (SRTM) topography data
    (.hgt extension).

    Keyword Arguments:
    fobj -- filename or file object
    """
    fobj.seek(0, 2)
    fsize = fobj.tell()
    fobj.seek(0)

    dim = int(np.sqrt(fsize / 2))
    if dim * dim * 2 != fsize:
        raise RuntimeError('File physical_sizes of {0} bytes does not match file physical_sizes '
                           'for a map of dimension {1}x{1}.'.format(fsize, dim))
    data = np.fromfile(fobj, dtype=np.dtype('>i2'),
                       count=dim * dim).reshape((dim, dim))

    if size is None:
        return Topography(data, data.shape)
    else:
        return Topography(data, size)
HGTReader = make_wrapped_reader(read_hgt, name="HgtReader")
