#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   FromFile.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Topography profile from file input

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import io
import re
import xml.etree.ElementTree as ElementTree
from io import TextIOWrapper
from struct import unpack
from zipfile import ZipFile

import numpy as np

from .TopographyUniform import UniformNumpyTopography
from .TopographyNonuniform import NonuniformNumpyTopography
from .TopographyPipeline import ScaledTopography

###

height_units = {'m': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'nm': 1e-9, 'A': 1e-10}
voltage_units = {'kV': 1000.0, 'V': 1.0, 'mV': 1e-3, 'µV': 1e-6, 'nV': 1e-9}

units = dict(height=height_units, voltage=voltage_units)

###

def binary(func):
    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        retvals = func(fobj, *args, **kwargs)
        if close_file:
            fobj.close()
        return retvals

    return func_wrapper


def text(func):
    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'r')
            fobj_text = fobj
            close_file = True
        elif is_binary_stream(fobj):
            fobj_text = TextIOWrapper(fobj)
        else:
            fobj_text = fobj

        try:
            retvals = func(fobj_text, *args, **kwargs)
        except:
            # This is iffy. We need to catch exceptions that happen during loadtxt, because if fobj_text is a
            # TextIOWrapper, it will close the file when it is deleted whenver the function returns through an
            # exception. We need to detach the TextIOWrapper before exiting.
            if is_binary_stream(fobj):
                fobj_text.detach()
            raise

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
    elif unit == 'μm' or unit == 'um' or unit == '~m':
        return 'µm'
    else:
        return unit


@text
def read_matrix(fobj, size=None, factor=None):
    """
    Reads a surface profile from a text file and presents in in a
    Topography-conformant manner. No additional parsing of meta-information is
    carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    surface = UniformNumpyTopography(np.loadtxt(fobj), size=size)
    if factor is not None:
        surface = ScaledTopography(surface, factor)
    return surface


@text
def read_asc(fobj, unit=None, x_factor=1.0, z_factor=1.0):
    # pylint: disable=too-many-branches,too-many-statements,invalid-name
    """
    Reads a surface profile from an generic asc file and presents it in a
    surface-conformant manner. Applies some heuristic to extract
    meta-information for different file formats. All units of the returned
    surface are in meters.

    Keyword Arguments:
    fobj_in -- filename or file object
    unit -- name of surface units, one of m, mm, μm/um, nm, A
    x_factor -- multiplication factor for size
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
            "The number of rows (={}) read from the file '{}' does "
            "not match the resolution in the file's metadata (={})."
                .format(nx, fname, xres))
    if yres is not None and yres != ny:
        raise Exception("The number of columns (={}) read from the file '{}' "
                        "does not match the resolution in the file's metadata "
                        "(={}).".format(ny, fname, yres))

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
    if zfac is None:
        zfac = 1.0
    zfac *= z_factor

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
            zfac *= height_units[zunit] / height_units[unit]

    if xsiz is None or ysiz is None:
        surface = UniformNumpyTopography(data, unit=unit)
    else:
        surface = UniformNumpyTopography(data, size=(x_factor * xsiz, x_factor * ysiz),
                                         unit=unit)
    surface = ScaledTopography(surface, zfac)
    return surface


@text
def read_xyz(fobj, unit=None):
    """
    Load xyz-file. These files contain line scan information in terms of (x,y)-positions.

    Parameters
    ----------
    fobj : str or file object
         File name or stream.

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

        return NonuniformNumpyTopography(x, z, size=np.max(x), unit=unit)
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

        return UniformNumpyTopography(data, size=(dx * nx, dy * ny), unit=unit)
    else:
        raise Exception('Expected two or three columns for topgraphy that is a list of positions and heights.')


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

    return UniformNumpyTopography(data, size=(xinc * nx, yinc * ny))


def read_mat(fobj, size=None, factor=None, unit=None):
    """
    Reads a surface profile from a matlab file and presents in in a
    Topography-conformant manner.

    All two-dimensional arrays present in the matlab data file are returned.

    Keyword Arguments:
    fobj -- filename or file object
    size -- size of the surface
    factor -- scaling factor for height
    unit -- size and height unit
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
            surface = UniformNumpyTopography(value, size=size, unit=unit)
            if factor is not None:
                surface = ScaledTopography(surface, factor)
            surfaces += [surface]
    if len(surfaces) == 1:
        return surfaces[0]
    else:
        return surfaces


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
        raise IOError('Directory length is not a multiple of the block size.')

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
                raise IOError("Don't know how to handle element size {}."
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
    surface = UniformNumpyTopography(data, size=(nx * pixel_size, ny * pixel_size * aspect),
                                     unit='mm')
    surface = ScaledTopography(surface, wavelength / mult * 1e-6)
    return surface


@binary
def read_di(fobj):
    """
    Load Digital Instrument's Nanoscope files.

    FIXME: Descriptive error messages. Probably needs to be made more robust.

    Keyword Arguments:
    fobj -- filename or file object
    """

    parameters = []
    section_name = None
    section_dict = {}

    l = fobj.readline().decode('latin-1').strip()
    while l and l.lower() != '\*file list end':
        if l.startswith('\\*'):
            if section_name is not None:
                parameters += [(section_name, section_dict)]
            section_name = l[2:].lower()
            section_dict = {}
        elif l.startswith('\\'):
            s = l[1:].split(': ', 1)
            try:
                key, value = s
            except ValueError:
                key, = s
                value = ''
            section_dict[key.lower()] = value.strip()
        else:
            raise IOError("Header line '{}' does not start with a slash."
                          "".format(l))
        l = fobj.readline().decode('latin-1').strip()
    parameters += [(section_name, section_dict)]

    surfaces = []
    scanner = {}
    for n, p in parameters:
        if n == 'scanner list' or n == 'ciao scan list':
            scanner.update(p)
        elif n == 'ciao image list':
            image_data_key = re.match('^S \[(.*?)\] ',
                                      p['@2:image data']).group(1)

            nx = int(p['samps/line'])
            ny = int(p['number of lines'])
            s = p['scan size'].split(' ', 2)
            sx = float(s[0])
            sy = float(s[1])
            xy_unit = mangle_height_unit(s[2])
            offset = int(p['data offset'])
            length = int(p['data length'])
            elsize = int(p['bytes/pixel'])
            if elsize == 2:
                dtype = np.dtype('<i2')
            else:
                raise IOError("Don't know how to handle {} bytes per pixel "
                              "data.".format(elsize))
            if nx * ny * elsize != length:
                raise IOError('Data block size differs from extend of surface.')
            fobj.seek(offset)
            rawdata = fobj.read(nx * ny * dtype.itemsize)
            unscaleddata = np.frombuffer(rawdata, count=nx * ny,
                                         dtype=dtype).reshape(nx, ny)

            scale_re = re.match('^V \[(.*?)\] \(([0-9\.]+) (.*)\/LSB\) (.*) '
                                '(.*)', p['@2:z scale'])
            quantity = scale_re.group(1).lower()
            hard_scale = float(scale_re.group(4)) / 65536
            hard_unit = scale_re.group(5)

            s = scanner['@' + quantity].split()
            if s[0] != 'V' or len(s) < 2:
                raise ValueError('Malformed Nanoscope DI file.')
            soft_scale = float(s[1])

            height_unit = None
            hard_to_soft = 1.0
            if len(s) > 2:
                # Check units
                height_unit, soft_unit = s[2].split('/')
                hard_to_soft = get_unit_conversion_factor(hard_unit, soft_unit)
                if hard_to_soft is None:
                    raise ValueError("Units for hard (={}) and soft (={}) "
                                     "scale differ for '{}'. Don't know how "
                                     "to handle this.".format(hard_unit,
                                                              soft_unit,
                                                              image_data_key))

            if height_unit in height_units:
                height_unit = mangle_height_unit(height_unit)
                if xy_unit != height_unit:
                    fac = get_unit_conversion_factor(xy_unit, height_unit)
                    sx *= fac
                    sy *= fac
                    xy_unit = height_unit
                unit = height_unit
            else:
                unit = (xy_unit, height_unit)

            surface = UniformNumpyTopography(unscaleddata.T, size=(sx, sy), unit=unit)
            surface.info.update(dict(data_source=image_data_key))
            surface = ScaledTopography(surface, hard_scale * hard_to_soft * soft_scale)
            surfaces += [surface]

    if len(surfaces) == 1:
        return surfaces[0]
    else:
        return surfaces


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

    surface = UniformNumpyTopography(data, size=(nx * sfA[0], ny * sfA[1]), unit=z_unit)

    return surface


@binary
def read_hgt(fobj):
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
        raise RuntimeError('File size of {0} bytes does not match file size '
                           'for a map of dimension {1}x{1}.'.format(fsize,
                                                                    dim))
    data = np.fromfile(fobj, dtype=np.dtype('>i2'),
                       count=dim * dim).reshape((dim, dim))

    return UniformNumpyTopography(data)


def read_h5(fobj):
    import h5py
    h5 = h5py.File(fobj)
    return UniformNumpyTopography(h5['surface'][...])


def detect_format(fobj):
    """
    Detect file format based on its content.

    Parameters
    ----------
    fobj : string or file object
        File name or open stream. Note that the stream should be opened as
        binary, otherwise the binary formats will not be detected.
    """

    close_file = False
    if not hasattr(fobj, 'read'):
        try:
            import h5py
            h5 = h5py.File(fobj, 'r')
            return 'h5'
        except:
            pass

        fname = fobj
        fobj = open(fobj, 'rb')
        close_file = True

    magic_len = 20
    file_pos = fobj.tell()
    magic = fobj.read(magic_len)
    fobj.seek(file_pos)

    # Check for magic string
    if is_binary_stream(fobj):
        if magic.startswith(b'\*File list'):
            if close_file:
                fobj.close()
            return 'di'
        elif magic.startswith(b'\001\000Directory'):
            if close_file:
                fobj.close()
            return 'opd'
        else:
            # Try opening at matlab and see if it fails
            try:
                from scipy.io import loadmat
                loadmat(fobj)
                if close_file:
                    fobj.close()
                else:
                    fobj.seek(file_pos)
                return 'mat'
            except:
                pass

            # Try opening zip and see if it fails
            try:
                with ZipFile(fobj, 'r') as zipfile:
                    if 'main.xml' in zipfile.namelist():
                        if close_file:
                            fobj.close()
                        else:
                            fobj.seek(file_pos)
                        return 'x3p'
            except:
                pass

            # Try opening igor binary wave and see if it fails
            fobj.seek(file_pos)
            import igor.binarywave as ibw
            try:
                ibw.load(fobj)
                if close_file:
                    fobj.close()
                else:
                    fobj.seek(file_pos)
                return 'ibw'
            except:
                pass

    # Finally, this could be a line scan in text format
    try:
        read_xyz(fobj)
        if close_file:
            fobj.close()
        else:
            fobj.seek(file_pos)
        return 'xyz'
    except:
        pass

    if close_file:
        fobj.close()
    else:
        fobj.seek(file_pos)
    return None


def read(fobj, format=None):
    if format is None:
        format = detect_format(fobj)
        if format is None:
            format = 'asc'

    readers = {'di': read_di,
               'h5': read_h5,
               'ibw': read_ibw,
               'mat': read_mat,
               'opd': read_opd,
               'x3p': read_x3p,
               'xyz': read_xyz}

    format = format.lower()
    if format not in readers:
        return read_asc(fobj)
    else:
        return readers[format](fobj)
