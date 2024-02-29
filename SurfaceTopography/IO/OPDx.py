#
# Copyright 2019-2021, 2023 Lars Pastewka
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

# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/lextfile.c

from collections import namedtuple

import dateutil.parser
import numpy as np

from ..Exceptions import FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo

MAGIC = b'VCA DATA\x01\x00\x00\x55'

DEKTAK_MATRIX = 0x00  # Too lazy to assign an actual type id?
DEKTAK_BOOLEAN = 0x01  # Takes value 0 and 1
DEKTAK_SINT16 = 0x04
DEKTAK_UINT16 = 0x05
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
    _mime_types = ['application/x-dektak-opdx']
    _file_extensions = ['opdx']

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
                raise FileFormatMismatch('File magic does not match. This is not a Dektak OPDx file.')

            # Read OPDx file manifest (without reading arrays and matrices)
            self.manifest = {}
            while _read_item(f, self.manifest):
                pass

        # Populate channel information
        self._channels = []
        data_kind = self.manifest['/MetaData/DataKind']
        if data_kind == 'Surface Profile':
            # This file contains a line scan
            self.read_linescan_channel_infos(0)
        elif data_kind == 'Surface Height':
            # This file contains a topography scan
            self.read_topography_channel_infos(0)
        else:
            raise UnsupportedFormatFeature(f"Don't know how to read data of kind '{data_kind}'.")

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

        info = info.copy()
        info.update(channel_info.info)

        if channel_info.dim == 1:
            position, length = self.manifest[f'{prefix}/Array']
            assert length == 8 * channel_info.nb_grid_pts[0] + DOUBLE_ARRAY_EXTRA
            with OpenFromAny(self.file_path, 'rb') as f:
                f.seek(position + DOUBLE_ARRAY_EXTRA)
                data = np.frombuffer(f.read(length - DOUBLE_ARRAY_EXTRA), np.dtype('<f8'))

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return UniformLineScan(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                                   info=info).scale(channel_info.height_scale_factor)
        elif channel_info.dim == 2:
            some_int, another_name, position, length, xres, yres = self.manifest[f'{prefix}/Matrix']
            assert channel_info.nb_grid_pts == (xres, yres)
            assert length == 4 * xres * yres

            with OpenFromAny(self.file_path, 'rb') as f:
                f.seek(position)
                data = np.frombuffer(f.read(length), np.dtype('<f4')).reshape((yres, xres)).T

            physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

            return Topography(heights=data, physical_sizes=physical_sizes, unit=unit, periodic=periodic,
                              info=info).scale(channel_info.height_scale_factor)
        else:
            raise RuntimeError(f"Don't know how to read a {channel_info.dim}-dimensional dataset.")

    @property
    def channels(self):
        return self._channels

    def info_from_manifest(self, prefix):
        info = {'opdx_prefix': prefix}

        try:
            acquisition_time = str(dateutil.parser.parse(
                self.manifest['/MetaData/Date'] + ' ' + self.manifest['/MetaData/Time']))
            info['acquisition_time'] = acquisition_time
        except KeyError:
            pass

        try:
            info['instrument'] = {'name': self.manifest['/MetaData/MeasurementSettings/InstrumentName']}
        except KeyError:
            pass

        return info

    def read_linescan_channel_infos(self, channel_index):
        """
        Read line scan (profile) information
        """
        channel_names = self.manifest['/MetaData/1D_Channels/Height']

        for channel_name in channel_names:
            prefix = f'/1D_Data/{channel_name}'

            nb_grid_pts = self.manifest[f'{prefix}/NumPoints']
            physical_size = self.manifest[f'{prefix}/Extent'].value
            unit = self.manifest[f'{prefix}/Extent'].symbol

            height_scale_factor = self.manifest[f'{prefix}/DataScale'].value
            height_unit = self.manifest[f'{prefix}/DataScale'].symbol

            height_scale_factor *= get_unit_conversion_factor(height_unit, unit)

            self._channels += [ChannelInfo(self, channel_index,
                                           name=self.manifest[f'{prefix}/DataKind'],
                                           dim=1,
                                           nb_grid_pts=nb_grid_pts,
                                           physical_sizes=physical_size,
                                           uniform=True,
                                           unit=unit,
                                           height_scale_factor=height_scale_factor,
                                           info=self.info_from_manifest(prefix))]

            channel_index += 1

    def read_topography_channel_infos(self, channel_index):
        """
        Read topography (2D map) information
        """
        channel_name = self.manifest['/MetaData/PrimaryData2D']
        prefix = f'/2D_Data/{channel_name}'

        nb_grid_pts_y = self.manifest[f'{prefix}/Dimension1Points']
        nb_grid_pts_x = self.manifest[f'{prefix}/Dimension2Points']

        physical_size_y = self.manifest[f'{prefix}/Dimension1Extent'].value
        unit_y = self.manifest[f'{prefix}/Dimension1Extent'].symbol

        physical_size_x = self.manifest[f'{prefix}/Dimension2Extent'].value
        unit_x = self.manifest[f'{prefix}/Dimension2Extent'].symbol

        physical_size_y *= get_unit_conversion_factor(unit_y, unit_x)

        height_scale_factor = self.manifest[f'{prefix}/DataScale'].value
        height_unit = self.manifest[f'{prefix}/DataScale'].symbol

        height_scale_factor *= get_unit_conversion_factor(height_unit, unit_x)

        self._channels += [ChannelInfo(self, channel_index,
                                       name=self.manifest[f'{prefix}/DataKind'],
                                       dim=2,
                                       nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                                       physical_sizes=(physical_size_x, physical_size_y),
                                       uniform=True,
                                       unit=unit_x,
                                       height_scale_factor=height_scale_factor,
                                       info=self.info_from_manifest(prefix))]

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


DektakUnit = namedtuple('DektakUnit', ['name', 'symbol', 'value', 'extra'])


def _read_item(stream, manifest, prefix='', offset=0):
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
    name = _read_name(stream)
    path = f'{prefix}/{name}'
    data = None

    typeid = _read_scalar(stream, 'B')

    # Container types
    if typeid == DEKTAK_CONTAINER or typeid == DEKTAK_RAW_DATA or typeid == DEKTAK_RAW_DATA_2D:
        length = _read_varlen(stream)
        start = stream.tell()
        while _read_item(stream, manifest, prefix=path):
            pass
        # There are two bytes at the end of every container with unknown purpose...
        _read_scalar(stream, '<u2')
        if start + length != stream.tell():
            raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                               f'does not equal the expected position {start + length}.')
        stream.seek(start + length)
        return path
    elif typeid == DEKTAK_POS_RAW_DATA:
        if path.startswith('/2D_Data'):
            _read_name(stream)  # typename, we discard this information
            length = _read_varlen(stream)
            start = stream.tell()
            unitx, divisorx = read_dimension2d_content(stream)
            unity, divisory = read_dimension2d_content(stream)
            data = (unitx, divisorx, unity, divisory)
            if start + length != stream.tell():
                raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                                   f'does not equal the expected position {start + length}.')
        elif path.startswith('/1D_Data'):
            _read_name(stream)  # typename, we discard this information
            length = _read_varlen(stream)
            start = stream.tell()
            unit = _read_unit_data(stream)
            count = _read_scalar(stream, '<u8')
            # Skip over data
            stream.seek(start + length)
            data = (unit, count)
        else:
            # TODO check if should assume 1D here like Gwyddion
            raise ValueError
    elif typeid == DEKTAK_TERMINATOR:
        # Terminator
        return None
    else:
        try:
            t = _item_readers[typeid]
        except KeyError:
            raise ValueError(f"Don't know how to read type with id {typeid}. This occured at stream position "
                             f"{stream.tell()}.")
        if isinstance(t, str):
            data = _read_scalar(stream, t)
        else:
            data = t(stream)

    if data is not None:
        manifest[path] = data
        return path
    else:
        return None


def _read_time_stamp(stream):
    return stream.read(TIMESTAMP_SIZE)


def _read_unit(stream):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    f : bytes
        The input buffer

    Returns
    -------
    unit : DektakUnit
        A unit item, filled with value, name and symbol
    """
    length = _read_varlen(stream)
    start = stream.tell()
    unit = _read_unit_data(stream)
    if start + length != stream.tell():
        raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                           f'does not equal the expected position {start + length}.')
    return unit


def _read_unit_data(stream):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    stream : bytes
        The input buffer

    Returns
    -------
    unit : DektakUnit
        A unit item, filled with value, name and symbol
    """
    name = _read_name(stream)
    symbol = mangle_length_unit_utf8(_read_name(stream))

    value = _read_scalar(stream, '<f8')
    extra = stream.read(UNIT_EXTRA)

    return DektakUnit(name, symbol, value, extra)


def _read_quantity(stream):
    """
    Reads in a quantity unit: Value, name and symbol.

    Parameters
    ----------
    stream : bytes
        The input buffer

    Returns
    -------
    unit : DektakUnit
        A unit item, filled with value, name and symbol
    """
    length = _read_varlen(stream)
    start = stream.tell()

    value = _read_scalar(stream, '<f8')
    name = _read_name(stream)
    symbol = mangle_length_unit_utf8(_read_name(stream))

    stream.seek(start + length)

    return DektakUnit(name, symbol, value, [])


def read_dimension2d_content(stream):
    """
    Reads in information about a 2d dimension.

    Parameters
    ----------
    stream : bytes
        The input buffer

    Returns
    -------
    unit : DektakUnit
        The unit
    divisor : float
        Divisor
    """
    value = _read_scalar(stream, '<f8')
    name = _read_name(stream)
    symbol = _read_name(stream)
    divisor = _read_scalar(stream, '<f8')
    extra = stream.read(UNIT_EXTRA)
    return DektakUnit(name, symbol, value, extra), divisor


def _read_name(stream):
    # Names always have a size of 4 bytes
    length = _read_scalar(stream, '<u4')
    return stream.read(length).decode('UTF-8')


def _read_string(stream):
    # String have variable lengths
    string_length = _read_varlen(stream)
    return stream.read(string_length).decode('UTF-8')


def _read_scalar(stream, dtype):
    dtype = np.dtype(dtype)
    buffer = stream.read(dtype.itemsize)
    if buffer:
        return np.frombuffer(buffer, dtype=dtype, count=1)[0]
    else:
        return None


def _read_varlen(stream):
    lenlen = _read_scalar(stream, 'B')
    if lenlen == 1:
        return _read_scalar(stream, 'B')
    elif lenlen == 2:
        return _read_scalar(stream, '<u2')
    elif lenlen == 4:
        return _read_scalar(stream, '<u4')
    else:
        raise ValueError(f"Don't know how to read a variable length of size {lenlen}.")


def _read_string_list(stream):
    _read_name(stream)  # typename, we discard this information
    length = _read_varlen(stream)
    start = stream.tell()
    data = []
    while stream.tell() < start + length:
        s = _read_name(stream)
        data += [s]
    if start + length != stream.tell():
        raise RuntimeError(f'The OPDx reader has missed some data entries. The stream position {stream.tell()} '
                           f'does not equal the expected position {start + length}.')
    return data


def _read_double_array(stream):
    _read_name(stream)  # typename, we discard this information
    length = _read_varlen(stream)
    data = (stream.tell(), length)
    # Skip over data
    stream.seek(length, 1)
    return data  # This is start position and length of the buffer


def _read_type_id(stream):
    _read_name(stream)  # typename, we discard this information
    length = _read_varlen(stream)
    return stream.read(length)


def _read_matrix(stream):
    _read_name(stream)  # typename, we discard this information
    some_int = _read_scalar(stream, '<u4')
    another_name = _read_name(stream)
    length = _read_varlen(stream)
    yres = _read_scalar(stream, '<u4')
    xres = _read_scalar(stream, '<u4')
    if length < 8:  # 2 * sizeof int32
        raise ValueError
    length -= 8  # Remove xres and yres
    data = (some_int, another_name, stream.tell(), length, xres, yres)
    # Skip over data
    stream.seek(length, 1)
    return data


_item_readers = {
    DEKTAK_MATRIX: _read_matrix,
    DEKTAK_BOOLEAN: '?',
    DEKTAK_SINT16: '<i2',
    DEKTAK_UINT16: '<u2',
    DEKTAK_SINT32: '<i4',
    DEKTAK_UINT32: '<u4',
    DEKTAK_SINT64: '<i8',
    DEKTAK_UINT64: '<u8',
    DEKTAK_FLOAT: '<f4',
    DEKTAK_DOUBLE: '<f8',
    DEKTAK_TYPE_ID: _read_type_id,
    DEKTAK_STRING: _read_string,
    DEKTAK_QUANTITY: _read_quantity,
    DEKTAK_TIME_STAMP: _read_time_stamp,
    DEKTAK_UNITS: _read_unit,
    DEKTAK_DOUBLE_ARRAY: _read_double_array,
    DEKTAK_STRING_LIST: _read_string_list,
}
