#
# Copyright 2019-2021 Lars Pastewka
#           2019-2021 Michael Röttger
#           2019-2020 Antoine Sanner
#           2019-2020 Kai Haase
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

import datetime
from collections import OrderedDict

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo

MAGIC = "VCA DATA\x01\x00\x00\x55"

DEKTAK_MATRIX = 0x00  # Too lazy to assign an actual type id?
DEKTAK_BOOLEAN = 0x01  # Takes value 0 and 1
DEKTAK_SINT32 = 0x06
DEKTAK_UINT32 = 0x07
DEKTAK_SINT64 = 0x0a
DEKTAK_UINT64 = 0x0b
DEKTAK_FLOAT = 0x0c  # Single precision float
DEKTAK_DOUBLE = 0x0d  # Double precision float
DEKTAK_TYPE_ID = 0x0e  # Compound type holding some kind of type id
DEKTAK_STRING = 0x12  # Free-form string value
DEKTAK_QUANTITY = 0x13  # Value with units (compound type)
DEKTAK_TIME_STAMP = 0x15  # Datetime (string/9-byte binary)
DEKTAK_UNITS = 0x18  # Units (compound type)
DEKTAK_DOUBLE_ARRAY = 0x40  # Raw data array, in XML Base64-encoded
DEKTAK_STRING_LIST = 0x42  # List of Str
DEKTAK_RAW_DATA = 0x46  # Parent/wrapper tag of raw data
DEKTAK_RAW_DATA_2D = 0x47  # Parent/wrapper tag of raw data
# Base64-encoded positions, not sure how it differs from 64
DEKTAK_POS_RAW_DATA = 0x7c
DEKTAK_CONTAINER = 0x7d  # General nested data structure
# Always the last item. Usually a couple of 0xff bytes inside.
DEKTAK_TERMINATOR = 0x7f

TIMESTAMP_SIZE = 9
UNIT_EXTRA = 12
DOUBLE_ARRAY_EXTRA = 5


class OPDxReader(ReaderBase):
    _format = 'opdx'
    _name = 'Dektak OPDx'
    _description = '''
File format of the Bruker Dektak XT* series stylus profilometer.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        with OpenFromAny(file_path, 'rb') as f:
            # read topography in file as hexadecimal
            self.buffer = [chr(byte) for byte in f.read()]

            # length of file
            physical_sizes = len(self.buffer)

            # check if correct header
            if physical_sizes < len(MAGIC) or ''.join(self.buffer[:len(MAGIC)]) != MAGIC:
                raise ValueError('Invalid file format for Dektak OPDx.')

            pos = len(MAGIC)
            self.manifest = {}
            while pos < physical_sizes:
                buf, pos, manifest, path = read_item(buf=self.buffer, pos=pos, manifest=self.manifest, path="")

            # Read channel information
            self._channels = []
            data_kind = manifest['/MetaData/DataKind'].data
            if data_kind == 'Surface Profile':
                self.read_linescan_channel_infos(0)
            elif data_kind == 'Surface Height':
                self.read_topography_channel_infos(0)
            else:
                raise ValueError(f"Cannot load data of kind '{data_kind}'.")

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI '
                               'parallelization.')

        channel_info = self._channels[channel_index]
        prefix = channel_info.info['opdx_prefix']
        if channel_info.unit is not None:
            if unit is not None:
                raise MetadataAlreadyFixedByFile('unit')
            unit = channel_info.unit
        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if channel_info.dim == 1:
            position, length = self.manifest[f'{prefix}/Array'].data
            data = read_array(channel_info.nb_grid_pts[0], self.buffer[position:position + length])

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return UniformLineScan(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                                   info=channel_info.info).scale(channel_info.height_scale_factor)
        elif channel_info.dim == 2:
            some_int, another_name, position, length, xres, yres = self.manifest[f'{prefix}/Matrix'].data
            assert channel_info.nb_grid_pts == (xres, yres)
            assert length == 4 * xres * yres
            data = read_matrix(xres, yres, self.buffer[position:position + length], 1.0).T
            assert data.shape == (xres, yres)

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return Topography(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                              info=channel_info.info).scale(channel_info.height_scale_factor)
        else:
            raise RuntimeError(f"Don't know how to read a {channel_info.dim}-dimensional dataset.")

    @property
    def channels(self):
        return self._channels

    def read_linescan_channel_infos(self, channel_index):
        """
        Read line scan (profile) information
        """
        channel_names = self.manifest['/MetaData/1D_Channels/Height'].data

        for channel_name in channel_names:
            prefix = f'/1D_Data/{channel_name}'

            nb_grid_pts = self.manifest[f'{prefix}/NumPoints'].data
            physical_size = self.manifest[f'{prefix}/Extent'].data.value
            unit = self.manifest[f'{prefix}/Extent'].data.symbol

            height_scale_factor = self.manifest[f'{prefix}/DataScale'].data.value
            height_unit = self.manifest[f'{prefix}/DataScale'].data.symbol

            height_scale_factor *= get_unit_conversion_factor(height_unit, unit)

            acquisition_time = datetime.datetime.strptime(
                self.manifest['/MetaData/Date'].data + ' ' + self.manifest['/MetaData/Time'].data,
                '%d/%m/%Y %I:%M:%S %p')
            info = {
                'opdx_prefix': prefix,
                'acquisition_time': acquisition_time,
                'instrument': {
                    'name': self.manifest['/MetaData/MeasurementSettings/InstrumentName'].data
                }
            }

            self._channels += [ChannelInfo(self, channel_index,
                                           name=self.manifest[f'{prefix}/DataKind'].data,
                                           dim=1,
                                           nb_grid_pts=nb_grid_pts,
                                           physical_sizes=physical_size,
                                           unit=unit,
                                           height_scale_factor=height_scale_factor,
                                           info=info)]

            channel_index += 1

    def read_topography_channel_infos(self, channel_index):
        """
        Read topography (2D map) information
        """
        channel_name = self.manifest['/MetaData/PrimaryData2D'].data
        prefix = f'/2D_Data/{channel_name}'

        nb_grid_pts_y = self.manifest[f'{prefix}/Dimension1Points'].data
        nb_grid_pts_x = self.manifest[f'{prefix}/Dimension2Points'].data

        physical_size_y = self.manifest[f'{prefix}/Dimension1Extent'].data.value
        unit_y = self.manifest[f'{prefix}/Dimension1Extent'].data.symbol

        physical_size_x = self.manifest[f'{prefix}/Dimension2Extent'].data.value
        unit_x = self.manifest[f'{prefix}/Dimension2Extent'].data.symbol

        physical_size_y *= get_unit_conversion_factor(unit_y, unit_x)

        height_scale_factor = self.manifest[f'{prefix}/DataScale'].data.value
        height_unit = self.manifest[f'{prefix}/DataScale'].data.symbol

        height_scale_factor *= get_unit_conversion_factor(height_unit, unit_x)

        acquisition_time = datetime.datetime.strptime(
            self.manifest['/MetaData/Date'].data + ' ' + self.manifest['/MetaData/Time'].data,
            '%d/%m/%Y %I:%M:%S %p')
        info = {
            'opdx_prefix': prefix,
            'acquisition_time': acquisition_time,
            'instrument': {
                'name': self.manifest['/MetaData/MeasurementSettings/InstrumentName'].data
            }
        }

        self._channels += [ChannelInfo(self, channel_index,
                                       name=self.manifest[f'{prefix}/DataKind'].data,
                                       dim=2,
                                       nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                                       physical_sizes=(physical_size_x, physical_size_y),
                                       unit=unit_x,
                                       height_scale_factor=height_scale_factor,
                                       info=info)]

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


class DektakItem:
    def __init__(self):
        self.typename = None
        self.typeid = None
        self.data = None


class DektakQuantUnit:
    def __init__(self):
        self.name = None
        self.symbol = None
        self.value = None
        self.extra = None


def read_item(buf, pos, manifest, path, abspos=0):
    """
    Reads in the next item out of the buffer and saves it in the hash table.
    May recursively call itself for containers.

    Parameters
    ----------
    buf : bytes
        The raw data buffer
    pos : int
        Current position in the buffer
    manifest : OrderedDict
        The output hash table
    path : str
        Current name to save
    abspos : int, optional
        Absolute position in buffer to keep track when calling itself
        (Default: 0)

    Returns
    -------
    but : bytes,
        Output buffer
    newpos : int
        New position
    manifest : OrderedDict
        Hash table with new item in it
    new_path : str
        New path
    """
    orig_path_len = len(path)
    item = DektakItem()
    itempos = 0
    name, pos = read_name(buf, pos)

    path += '/'
    path += name

    item.typeid, pos = read_with_check(buf, pos, 1)
    item.typeid = ord(item.typeid[0])

    # simple types
    if item.typeid == DEKTAK_BOOLEAN:
        b8, pos = read_with_check(buf, pos, 1)
        if b8 == '\x01':
            item.data = True
        elif b8 == '\x00':
            item.data = False
        else:
            raise ValueError("Something went wrong.")

    elif item.typeid == DEKTAK_SINT32:
        item.data, pos = read_int32(buf, pos, signed=True)
    elif item.typeid == DEKTAK_UINT32:
        item.data, pos = read_int32(buf, pos, signed=False)
    elif item.typeid == DEKTAK_SINT64:
        item.data, pos = read_int64(buf, pos, signed=True)
    elif item.typeid == DEKTAK_UINT64:
        item.data, pos = read_int64(buf, pos, signed=False)
    elif item.typeid == DEKTAK_FLOAT:
        item.data, pos = read_float(buf, pos)
    elif item.typeid == DEKTAK_DOUBLE:
        item.data, pos = read_double(buf, pos)
    elif item.typeid == DEKTAK_TIME_STAMP:
        time, pos = read_with_check(buf, pos, TIMESTAMP_SIZE)
        item.data = time
    elif item.typeid == DEKTAK_STRING:
        item.data, pos = read_string(buf, pos)
    elif item.typeid == DEKTAK_QUANTITY:
        content, _, _, pos = read_structured(buf, pos)
        item.data, itempos = read_quantunit_content(content, itempos, False)
    elif item.typeid == DEKTAK_UNITS:
        content, _, _, pos = read_structured(buf, pos)
        item.data, itempos = read_quantunit_content(content, itempos, True)
    elif item.typeid == DEKTAK_TERMINATOR:
        pos = len(buf)
    # Container types.
    elif item.typeid == DEKTAK_CONTAINER or item.typeid == DEKTAK_RAW_DATA or item.typeid == DEKTAK_RAW_DATA_2D:
        # TODO find out if maybe better place somewhere else
        content, start, _, pos = read_structured(buf, pos)
        abspos += start
        while itempos < len(content):
            content, itempos, manifest, path = read_item(
                buf=content, pos=itempos, manifest=manifest,
                path=path, abspos=abspos)
    # Types with string type name
    elif item.typeid == DEKTAK_DOUBLE_ARRAY:
        item.typename, content, start, length, pos = read_named_struct(buf, pos)
        item.data = (start + abspos, length)
    elif item.typeid == DEKTAK_STRING_LIST:
        item.typename, content, start, _, pos = read_named_struct(buf, pos)
        item.data = []
        while itempos < len(content):
            s, itempos = read_name(content, itempos)
            item.data += [s]
    elif item.typeid == DEKTAK_TYPE_ID:
        item.typename, item.data, _, _, pos = read_named_struct(buf, pos)
    elif item.typeid == DEKTAK_POS_RAW_DATA:
        if path.startswith('/2D_Data'):
            item.typename, content, _, _, pos = read_named_struct(buf, pos)
            unitx, divisorx, itempos = read_dimension2d_content(content, itempos)
            unity, divisory, itempos = read_dimension2d_content(content, itempos)
            item.data = (unitx, divisorx, unity, divisory)
        elif path.startswith('/1D_Data'):
            item.typename, content, _, _, pos = read_named_struct(buf, pos)
            unit, itempos = read_quantunit_content(content, itempos, True)
            count, itempos = read_int64(content, itempos)
            item.data = (unit, count)
        else:
            # TODO check if should assume 1D here like Gwyddion
            raise ValueError
    elif item.typeid == DEKTAK_MATRIX:
        item.typename, pos = read_name(buf, pos)
        some_int, pos = read_int32(buf, pos)
        another_name, pos = read_name(buf, pos)
        length, pos = read_varlen(buf, pos)
        yres, pos = read_int32(buf, pos)
        xres, pos = read_int32(buf, pos)
        if length < 8:  # 2 * sizeof int32
            raise ValueError
        length -= 8
        p = pos + abspos
        item.data = (some_int, another_name, p, length, xres, yres)

        if len(buf) - pos < length:
            raise ValueError
        pos += length
    else:
        raise ValueError
    manifest[path] = item
    path = path[:orig_path_len]
    return buf, pos, manifest, path


def read_quantunit_content(buf, pos, is_unit):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    buf : bytes
        The input buffer
    pos : integer
        The position in the buffer
    is_unit : boot
        Whether or not it is a unit

    Returns
    -------
    quantunit : DektakQuantUnit
        A quantunit item, filled with value, name and symbol
    new_pos : int
        New position in input buffer
    """
    quantunit = DektakQuantUnit()
    quantunit.extra = []

    if not is_unit:
        quantunit.value, pos = read_double(buf, pos)

    quantunit.name, pos = read_name(buf, pos)
    quantunit.symbol, pos = read_name(buf, pos)
    quantunit.symbol = mangle_length_unit_utf8(quantunit.symbol)

    if is_unit:
        quantunit.value, pos = read_double(buf, pos)
        res, pos = read_with_check(buf, pos, UNIT_EXTRA)
        quantunit.extra += res

    return quantunit, pos


def read_dimension2d_content(buf, pos):
    """
    Reads in information about a 2d dimension.

    Parameters
    ----------
    buf : bytes
        The input buffer
    pos : integer
        Position in the buffer

    Returns
    -------
    unit : DektakQuantUnit
        The unit (is passed through)
    divisor : float
        Divisor
    new_pos : int
        New position in input buffer
    """
    unit = DektakQuantUnit()
    unit.value, pos = read_double(buf, pos)
    unit.name, pos = read_name(buf, pos)
    unit.symbol, pos = read_name(buf, pos)
    divisor, pos = read_double(buf, pos)
    unit.extra, pos = read_with_check(buf, pos, UNIT_EXTRA)
    return unit, divisor, pos


def read_matrix(xres, yres, data, q=1):
    """
    Reads a float matrix of given dimensions and multiplies with a scale.

    Parameters
    ----------
    xres : int
        Resolution along x-axis
    yres : int
        Resolution along y-axis
    data : bytes
        The raw hex data
    q : float
        The scale of the data

    Returns
    -------
    data : np.ndarray
        Matrix
    """
    data = ''.join(data)

    # build correct type: 4byte flat, little endian
    dt = np.dtype('f4')  # double
    dt = dt.newbyteorder('<')  # little-endian

    data = np.frombuffer(str.encode(data, "raw_unicode_escape"), dt)
    data = data.copy().reshape((yres, xres))
    data *= q
    return data


def read_array(nb_points, data):
    """
    Reads a float matrix of given dimensions and multiplies with a scale.

    Parameters
    ----------
    xres : int
        Resolution along x-axis
    yres : int
        Resolution along y-axis
    data : bytes
        The raw hex data
    q : float
        The scale of the data

    Returns
    -------
    data : np.ndarray
        Matrix
    """
    data = ''.join(data[DOUBLE_ARRAY_EXTRA:])

    assert len(data) == 8 * nb_points

    dt = np.dtype('<f8')  # double, little-endian
    data = np.frombuffer(str.encode(data, "raw_unicode_escape"), dt)
    return data


def read_name(buf, pos):
    """
    Reads a name.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    name : str
        Name
    new_pos : int
        New position in input buffer
    """

    # Names always have a size of 4 bytes
    length, pos = read_int32(buf, pos)
    if len(buf) < length or pos > len(buf) - length:
        raise ValueError("Some sizes went wrong.")
    position = pos

    name = buf[position:position + length]
    name = "".join(s for s in name)
    pos += length
    return name, pos


def read_structured(buf, pos):
    """
    Reads a length and returns a part of the buffer that long.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    out : int
        Output buffer
    start : int
        Start position within input buffer
    pos : int
        New position in input buffer
    """
    length, pos = read_varlen(buf, pos)
    if len(buf) < length or pos > len(buf) - length:
        raise ValueError("Some sizes went wrong.")
    start = pos
    pos += length
    return buf[start:start + length], start, length, pos


def read_string(buf, pos):
    """
    Reads a string.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    out : int
        Output buffer
    pos : int
        New position in input buffer
    """
    buf, start, length, pos = read_structured(buf, pos)
    return ''.join(buf), pos


def read_named_struct(buf, pos):
    """
    Same as `read_structured` but there is a name to it.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    name : str
        Name of buffer
    out : int
        Output buffer
    start : int
        Start position within input buffer
    pos : int
        New position in input buffer
    """
    typename, pos = read_name(buf, pos)
    content, start, length, pos = read_structured(buf, pos)
    return typename, content, start, length, pos


def read_varlen(buf, pos):
    """
    Reads a length of variable length itself

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    out : int
        Length value
    new_pos : int
        New position in input buffer
    """
    lenlen, pos = read_with_check(buf, pos, 1)
    lenlen = np.frombuffer(str.encode(lenlen, "raw_unicode_escape"), "<u1")[0]
    if lenlen == 1:
        length, pos = read_with_check(buf, pos, 1)
        length = \
            np.frombuffer(str.encode(length, "raw_unicode_escape"), "<u1")[0]
    elif lenlen == 2:
        length, pos = read_int16(buf, pos)
    elif lenlen == 4:
        length, pos = read_int32(buf, pos)
    else:
        raise ValueError
    return length, pos


def read_int64(buf, pos, signed=False):
    """
    Reads a 64bit int.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position
    signed : bool, optional
        Signed integer (Default: false)

    Returns
    -------
    out : int
        Output value
    new_pos : int
        New position in input buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=8)
    out = ''.join(out)
    dt = "<i8" if signed else "<u8"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_int32(buf, pos, signed=False):
    """
    Reads a 32bit int.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position
    signed : bool, optional
        Signed integer (Default: false)

    Returns
    -------
    out : int
        Output value
    new_pos : int
        New position in input buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = "<i4" if signed else "<u4"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_int16(buf, pos, signed=False):
    """
    Reads a 16bit integer.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position
    signed : bool, optional
        Signed integer (Default: false)

    Returns
    -------
    out : int
        Output value
    new_pos : int
        New position in input buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=2)
    out = ''.join(out)
    dt = "<i2" if signed else "<u2"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_double(buf, pos):
    """
    Reads a double (64bit)

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    out : float
        Output value
    new_pos : int
        New position in input buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=8)
    out = ''.join(out)
    dt = np.dtype('d')  # double
    dt = dt.newbyteorder('<')  # little-endian
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_float(buf, pos):
    """
    Reads a float (32bit)

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position

    Returns
    -------
    out : float
        Output value
    new_pos : int
        New position in input buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = np.dtype('f')  # double
    dt = dt.newbyteorder('<')  # little-endian
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_with_check(buf, pos, nbytes):
    """
    Reads and returns n bytes.

    Parameters
    ----------
    buf : bytes
        Input buffer
    pos : int
        The current position
    nbytes : int
        Number of bytes to read in

    Returns
    -------
    out : bytes
        Output buffer
    new_pos : int
        New position in input buffer
    """

    if len(buf) < nbytes or len(buf) - nbytes < pos:
        raise ValueError("Some sizes went wrong.")

    out = buf[pos:pos + nbytes]
    pos += int(nbytes)

    out = out[0] if nbytes == 1 else out
    return out, pos
