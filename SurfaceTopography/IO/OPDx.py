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
from struct import unpack

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo

MAGIC = b'VCA DATA\x01\x00\x00\x55'

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
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Check OPDx file magic
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError('File magic does not match. This is not a Dektak OPDx file.')

            # Read OPDx file manifest (without reading arrays and matrices)
            self.manifest = {}
            while read_item(f, self.manifest):
                pass

            # Read channel information
            self._channels = []
            data_kind = self.manifest['/MetaData/DataKind'].data
            if data_kind == 'Surface Profile':
                # This file contains a line scan
                self.read_linescan_channel_infos(0)
            elif data_kind == 'Surface Height':
                # This file contains a topography scan
                self.read_topography_channel_infos(0)
            else:
                raise ValueError(f"Don't know how to read data of kind '{data_kind}'.")

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
            #position = 48556
            assert length == 8 * channel_info.nb_grid_pts[0] + DOUBLE_ARRAY_EXTRA
            with OpenFromAny(self.file_path, 'rb') as f:
                f.seek(position + DOUBLE_ARRAY_EXTRA)
                data = np.frombuffer(f.read(length - DOUBLE_ARRAY_EXTRA), np.dtype('<f8'))

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return UniformLineScan(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                                   info=channel_info.info).scale(channel_info.height_scale_factor)
        elif channel_info.dim == 2:
            some_int, another_name, position, length, xres, yres = self.manifest[f'{prefix}/Matrix'].data
            assert channel_info.nb_grid_pts == (xres, yres)
            assert length == 4 * xres * yres

            with OpenFromAny(self.file_path, 'rb') as f:
                f.seek(position)
                data = np.frombuffer(f.read(length), np.dtype('<f4')).reshape((yres, xres)).T

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return Topography(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                              info=channel_info.info).scale(channel_info.height_scale_factor)
        else:
            raise RuntimeError(f"Don't know how to read a {channel_info.dim}-dimensional dataset.")

    @property
    def channels(self):
        return self._channels

    def info_from_manifest(self, prefix):
        info = {'opdx_prefix': prefix}

        try:
            acquisition_time = datetime.datetime.strptime(
                self.manifest['/MetaData/Date'].data + ' ' + self.manifest['/MetaData/Time'].data,
                '%d/%m/%Y %I:%M:%S %p')
            info['acquisition_time'] = acquisition_time
        except KeyError:
            pass

        try:
            info['instrument'] = {'name': self.manifest['/MetaData/MeasurementSettings/InstrumentName'].data}
        except KeyError:
            pass

        return info

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

            self._channels += [ChannelInfo(self, channel_index,
                                           name=self.manifest[f'{prefix}/DataKind'].data,
                                           dim=1,
                                           nb_grid_pts=nb_grid_pts,
                                           physical_sizes=physical_size,
                                           unit=unit,
                                           height_scale_factor=height_scale_factor,
                                           info=self.info_from_manifest(prefix))]

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

        self._channels += [ChannelInfo(self, channel_index,
                                       name=self.manifest[f'{prefix}/DataKind'].data,
                                       dim=2,
                                       nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                                       physical_sizes=(physical_size_x, physical_size_y),
                                       unit=unit_x,
                                       height_scale_factor=height_scale_factor,
                                       info=self.info_from_manifest(prefix))]

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


def read_item(stream, manifest, prefix='', offset=0):
    """
    Reads in the next item out of the buffer and saves it in the manifest.
    May recursively call itself for containers.

    Parameters
    ----------
    stream : file-like stream object
        The input file/buffer.
    manifest : dict
        The manifest that will be updated by a call to this function.
    prefix : str, optional
        Prefix to the keys that are stored in the manifest. (Default: '')
    offset : int, optional
        Offset of f relative to the absolute start of the buffer holding the
        full OPDx.
        (Default: 0)

    Returns
    -------
    key : str
        Return the key for the item that was created by this read command.
        None indicates that the end of the buffer or file has been reached.
    """
    item = DektakItem()
    name = read_name(stream)

    path = f'{prefix}/{name}'

    item.typeid = read_uint8(stream)

    # simple types
    if item.typeid == DEKTAK_BOOLEAN:
        item.data = bool(read_uint8(stream))
    elif item.typeid == DEKTAK_SINT32:
        item.data = read_sint32(stream)
    elif item.typeid == DEKTAK_UINT32:
        item.data = read_uint32(stream)
    elif item.typeid == DEKTAK_SINT64:
        item.data = read_sint64(stream)
    elif item.typeid == DEKTAK_UINT64:
        item.data = read_uint64(stream)
    elif item.typeid == DEKTAK_FLOAT:
        item.data = read_float(stream)
    elif item.typeid == DEKTAK_DOUBLE:
        item.data = read_double(stream)
    elif item.typeid == DEKTAK_TIME_STAMP:
        item.data = stream.read(TIMESTAMP_SIZE)
    elif item.typeid == DEKTAK_STRING:
        item.data = read_string(stream)
    elif item.typeid == DEKTAK_QUANTITY:
        length = read_varlen(stream)
        start = stream.tell()
        item.data = _read_quantity(stream)
        stream.seek(start + length)
        #stream.read(20)
        #if start + length != stream.tell():
        #    raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
        #                       f'does not equal the expected position {start + length}.')
    elif item.typeid == DEKTAK_UNITS:
        length = read_varlen(stream)
        start = stream.tell()
        item.data = _read_unit(stream)
        if start + length != stream.tell():
            raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                               f'does not equal the expected position {start + length}.')
    elif item.typeid == DEKTAK_TERMINATOR:
        return None

    # Container types.
    elif item.typeid == DEKTAK_CONTAINER or item.typeid == DEKTAK_RAW_DATA or item.typeid == DEKTAK_RAW_DATA_2D:
        length = read_varlen(stream)
        start = stream.tell()
        while read_item(stream, manifest, prefix=path):
            pass
        # There are two bytes at the end of every container with unknown purpose...
        read_uint16(stream)
        if start + length != stream.tell():
            raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                               f'does not equal the expected position {start + length}.')
        stream.seek(start + length)
        return path
    # Types with string type name
    elif item.typeid == DEKTAK_DOUBLE_ARRAY:
        item.typename = read_name(stream)
        length = read_varlen(stream)
        item.data = (stream.tell(), length)
        # Skip over data
        stream.seek(length, 1)
    elif item.typeid == DEKTAK_STRING_LIST:
        item.typename = read_name(stream)
        length = read_varlen(stream)
        start = stream.tell()
        item.data = []
        while stream.tell() < start + length:
            s = read_name(stream)
            item.data += [s]
        if start + length != stream.tell():
            raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                               f'does not equal the expected position {start + length}.')
    elif item.typeid == DEKTAK_TYPE_ID:
        item.typename = read_name(stream)
        length = read_varlen(stream)
        item.data = stream.read(length)
    elif item.typeid == DEKTAK_POS_RAW_DATA:
        if path.startswith('/2D_Data'):
            item.typename = read_name(stream)
            length = read_varlen(stream)
            start = stream.tell()
            unitx, divisorx = read_dimension2d_content(stream)
            unity, divisory = read_dimension2d_content(stream)
            item.data = (unitx, divisorx, unity, divisory)
            if start + length != stream.tell():
                raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                                   f'does not equal the expected position {start + length}.')
        elif path.startswith('/1D_Data'):
            item.typename = read_name(stream)
            length = read_varlen(stream)
            start = stream.tell()
            unit = _read_unit(stream)
            count = read_uint64(stream)
            # Skip over data
            stream.seek(start + length)
            item.data = (unit, count)
        else:
            # TODO check if should assume 1D here like Gwyddion
            raise ValueError
    elif item.typeid == DEKTAK_MATRIX:
        item.typename = read_name(stream)
        some_int = read_uint32(stream)
        another_name = read_name(stream)
        length = read_varlen(stream)
        yres = read_uint32(stream)
        xres = read_uint32(stream)
        if length < 8:  # 2 * sizeof int32
            raise ValueError
        length -= 8
        item.data = (some_int, another_name, offset + stream.tell(), length, xres, yres)
        # Skip over data
        stream.seek(length, 1)
    else:
        raise ValueError(f"Don't know how to read type with id {item.typeid}.")
    if item.data is not None:
        manifest[path] = item
        return path
    else:
        return None


def _read_unit(f):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    f : bytes
        The input buffer

    Returns
    -------
    quantunit : DektakQuantUnit
        A quantunit item, filled with value, name and symbol
    """
    quantunit = DektakQuantUnit()
    quantunit.extra = []

    quantunit.name = read_name(f)
    quantunit.symbol = read_name(f)
    quantunit.symbol = mangle_length_unit_utf8(quantunit.symbol)

    quantunit.value = read_double(f)
    quantunit.extra = f.read(UNIT_EXTRA)

    return quantunit


def _read_quantity(f):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    f : bytes
        The input buffer

    Returns
    -------
    quantunit : DektakQuantUnit
        A quantunit item, filled with value, name and symbol
    """
    quantunit = DektakQuantUnit()
    quantunit.extra = []

    quantunit.value = read_double(f)

    quantunit.name = read_name(f)
    quantunit.symbol = read_name(f)
    quantunit.symbol = mangle_length_unit_utf8(quantunit.symbol)

    return quantunit


def read_dimension2d_content(stream):
    """
    Reads in information about a 2d dimension.

    Parameters
    ----------
    stream : bytes
        The input buffer

    Returns
    -------
    unit : DektakQuantUnit
        The unit
    divisor : float
        Divisor
    """
    unit = DektakQuantUnit()
    unit.value = read_double(stream)
    unit.name = read_name(stream)
    unit.symbol = read_name(stream)
    divisor = read_double(stream)
    unit.extra = stream.read(UNIT_EXTRA)
    return unit, divisor


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


def read_name(f):
    # Names always have a size of 4 bytes
    length = read_uint32(f)
    return f.read(length).decode('raw_unicode_escape')


def read_string(stream):
    # String have variable lengths
    string_length = read_varlen(stream)
    return stream.read(string_length).decode('raw_unicode_escape')


def read_varlen(f):
    lenlen = read_uint8(f)
    if lenlen == 1:
        return read_uint8(f)
    elif lenlen == 2:
        return read_uint16(f)
    elif lenlen == 4:
        return read_uint32(f)
    else:
        raise ValueError(f"Don't know how to read a variable length of size {lenlen}.")


def read_sint64(f):
    return unpack('<q', f.read(8))[0]


def read_uint64(f):
    return unpack('<Q', f.read(8))[0]


def read_sint32(f):
    buffer = f.read(4)
    if buffer:
        return unpack('<l', buffer)[0]
    else:
        return None

def read_uint32(f):
    buffer = f.read(4)
    if buffer:
        return unpack('<L', buffer)[0]
    else:
        return None


def read_sint16(f):
    return unpack('<h', f.read(2))[0]


def read_uint16(f):
    return unpack('<H', f.read(2))[0]


def read_uint8(f):
    return unpack('B', f.read(1))[0]

def read_double(f):
    return unpack('<d', f.read(8))[0]


def read_float(f):
    return unpack('<f', f.read(4))[0]
