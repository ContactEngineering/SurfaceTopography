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

from collections import OrderedDict
import numpy as np

from ..UniformLineScanAndTopography import Topography
from ..UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo, MetadataAlreadyFixedByFile

MAGIC = "VCA DATA\x01\x00\x00\x55"
MAGIC_SIZE = 12

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

MEAS_SETTINGS = "/MetaData/MeasurementSettings"
RAW_1D_DATA = "/1D_Data/Raw"
ANY_2D_DATA = "/2D_Data/"


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
            if physical_sizes < MAGIC_SIZE or ''.join(
                    self.buffer[:MAGIC_SIZE]) != MAGIC:
                raise ValueError('Invalid file format for Dektak OPDx.')

            pos = MAGIC_SIZE
            hash_table = OrderedDict()
            while pos < physical_sizes:
                buf, pos, hash_table, path = read_item(buf=self.buffer,
                                                       pos=pos,
                                                       hash_table=hash_table,
                                                       path="")

            # Make a list of channels containing metadata about the
            # topographies but not reading them in directly yet
            all_channels_data = find_2d_data(hash_table, self.buffer)

            # Reformat the resulting dicts
            for channel_name in all_channels_data.keys():
                all_channels_data[channel_name][
                    -1], default_channel_name = reformat_dict(
                    channel_name, all_channels_data[channel_name][-1])

            self._default_channel_index = list(all_channels_data.keys()).index(
                default_channel_name)

            #
            # Build channel info objects and additional metadata for extracting
            # data
            #
            self._channels = []
            self._channels_xres_yres_start_stop_q = []
            for channel_index, channel_name in enumerate(all_channels_data):

                channel_data = all_channels_data[channel_name]
                *xres_yres_start_stop_q, metadata = channel_data

                #
                # find out physical sizes in a  common unit
                # without touching the meta data, scale heights accordingly
                #
                unit_x = mangle_length_unit_utf8(metadata['Width_unit'])
                size_x = metadata['Width_value']

                unit_y = mangle_length_unit_utf8(metadata['Height_unit'])
                size_y = metadata['Height_value']

                unit_z = mangle_length_unit_utf8(metadata['z_unit'])

                # we want value in unit_x units
                unit_factor_y = get_unit_conversion_factor(unit_y,
                                                           unit_x)
                if unit_factor_y is None:
                    raise ValueError(
                        'Units for size in x ("{}") and y ("{}") direction '
                        'are incompatible.'.format(unit_x, unit_y))
                size_y *= unit_factor_y

                if unit_z is None:
                    # No unit given for heights. Since we can only return one
                    # common unit, no unit should be returned
                    try:
                        del metadata['unit']
                    except KeyError:
                        pass
                    unit = None
                else:
                    # we want value in unit_x units
                    unit_factor_z = get_unit_conversion_factor(unit_z,
                                                               unit_x)
                    if unit_factor_z is None:
                        raise ValueError(
                            'Units for width ("{}") and data units ("{}") '
                            'are incompatible.'.format(unit_x, unit_y))

                    # we have converted everything to this unit
                    unit = unit_x

                ch_info = ChannelInfo(self, channel_index,
                                      name=metadata['Name'], dim=2,
                                      nb_grid_pts=(metadata['ImageWidth'],
                                                   metadata['ImageHeight']),
                                      physical_sizes=(size_x, size_y),
                                      unit=unit,
                                      height_scale_factor=1,  # means that this factor is fixed by file contents
                                      info=metadata)

                self._channels.append(ch_info)
                self._channels_xres_yres_start_stop_q.append(
                    xres_yres_start_stop_q)  # needed for building heights

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
        res_x, res_y, start, end, q = self._channels_xres_yres_start_stop_q[
            channel_index]

        data = build_matrix(res_x, res_y, self.buffer[start:end], q).T

        unit_z = channel_info.info['z_unit']

        if channel_info.unit is not None:
            common_unit = channel_info.unit

            if unit is not None:
                raise MetadataAlreadyFixedByFile('unit')

            if (common_unit is not None) and (unit_z != common_unit):
                # There is a common unit, but the z-unit in the file differs
                # from that common unit. So we need to scale the data
                # accordingly, such that also the height are represented in
                # common units.
                unit_factor_z = get_unit_conversion_factor(unit_z, common_unit)
                if unit_factor_z is None:
                    raise ValueError(
                        'Common unit ("{}") derived from lateral units '
                        'and data units ("{}") are incompatible.'.format(
                            common_unit, unit_z))
                data *= unit_factor_z
        else:
            common_unit = unit

        physical_sizes = self._check_physical_sizes(
            physical_sizes, channel_info.physical_sizes)

        info = info.copy()
        info.update(channel_info.info)

        topography = Topography(heights=data, physical_sizes=physical_sizes,
                                unit=common_unit, info=info, periodic=periodic)
        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        return topography

    @property
    def channels(self):
        return self._channels

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__


def reformat_dict(name, metadata):
    """
    Reformat the metadata dict from c convention to a more readable format and
    remove artefacts. Also gets and returns the default channel.
    :param name: The name of the current channel.
    :param metadata: The metadata dict
    :return:
    new dict: The nicer dict
    primary_channel_name: The name of the primary channel.
    """
    new_dict = OrderedDict()

    primary_channel_name = None
    for key in metadata.keys():

        if key == '::MetaData::PrimaryData2D':
            primary_channel_name = metadata[key]

        if key.startswith('::MetaData::'):
            # These are in here for no reason and the value is None
            if not key[12:].endswith('::'):
                new_dict[key[12:].replace('::', '_').replace(' ', '_')] = \
                    metadata[key]

        if key.startswith(str(name) + '::'):
            name_len = 2 + len(name)
            new_dict[key[name_len:].replace('::', '_').replace(' ', '_')] = \
                metadata[key]

    if '' in new_dict.keys():
        new_dict.pop('')

    if 'xres' in new_dict.keys():
        new_dict.pop('xres')

    if 'yres' in new_dict.keys():
        new_dict.pop('yres')

    # Correct weird characters in units
    for unit_key in ['z_unit', 'Height_unit', 'Width_unit']:
        if unit_key in new_dict.keys():
            unit = new_dict.pop(unit_key)
            if unit.startswith("Â"):
                unit = unit[1:]
            new_dict[unit_key] = unit

    new_dict['Name'] = name

    return new_dict, primary_channel_name


class DektakItemData:
    def __init__(self):
        self.b = None
        self.ui = None
        self.si = None
        self.uq = None
        self.sq = None
        self.d = None
        self.timestamp = []
        self.buf = None
        self.qun = None
        self.rawpos1d = DektakRawPos1D()
        self.rawpos2d = DektakRawPos2D()
        self.matrix = DektakMatrix()
        self.strlist = None


class DektakItem:
    def __init__(self):
        self.typename = None
        self.typeid = None
        self.data = DektakItemData()


class DektakRawPos1D:
    def __init__(self):
        self.unit = DektakQuantUnit()
        self.divisor = None
        self.count = None
        self.buf = DektakBuf


class DektakRawPos2D:
    def __init__(self):
        self.unitx = DektakQuantUnit()
        self.unity = DektakQuantUnit()
        self.divisorx = None
        self.divisory = None


class DektakQuantUnit:
    def __init__(self):
        self.name = None
        self.symbol = None
        self.value = None
        self.extra = None


class DektakMatrix:
    def __init__(self):
        self.another_name = None
        self.some_int = None
        self.xres = None
        self.yres = None
        self.buf = DektakBuf()


class DektakBuf:
    def __init__(self, position=None, length=None):
        self.position = position
        self.length = length


def find_1d_data(hash_table, buf):
    """ THIS HAS NOT BEEN TESTED DUE TO NO FILES WITH 1D DATA AVAILABLE."""

    item = hash_table.pop("/MetaData/MeasurementSettings/SamplesToLog", None)

    if item is None:
        return None
    else:
        raise NotImplementedError


def find_2d_data(hash_table, buf):
    """ Get all the 2d data channels out of the previously filled hash table.

    :param hash_table: The filled hash table
    :param buf: The raw hex data
    :return: Dictionary with all names, data and metadata of the different
    channels
    """
    output = OrderedDict()
    channels = []

    # Get a list of all channels containing 2d data matrices
    for key in hash_table.keys():
        found = find_2d_data_matrix(key, hash_table[key])
        if found is not None:
            channels.append(found)

    for channel in channels:
        meta_data = create_meta(hash_table)

        string = ANY_2D_DATA
        string += channel

        length = len(string)

        # Get position and res of data matrix
        string += "/Matrix"
        item = hash_table[string]
        start = item.data.matrix.buf.p

        end = start + item.data.matrix.buf.length
        xres = item.data.matrix.xres
        yres = item.data.matrix.yres

        meta_data[channel + "::xres"] = xres
        meta_data[channel + "::yres"] = yres

        # TODO: multiply value by interpret(item.data.qun.symbol)
        string = string[:length]
        string += "/Dimension1Extent"
        item = hash_table[string]
        yreal = item.data.qun.value
        yunit = item.data.qun.symbol

        meta_data[channel + "::Height value"] = yreal
        meta_data[channel + "::Height unit"] = yunit

        string = string[:length]
        string += "/Dimension2Extent"
        item = hash_table[string]
        xreal = item.data.qun.value
        xunit = item.data.qun.symbol

        meta_data[channel + "::Width value"] = xreal
        meta_data[channel + "::Width unit"] = xunit

        string = string[:length]
        string += "/DataScale"
        item = hash_table[string]
        q = item.data.qun.value
        zunit = item.data.qun.symbol

        meta_data[channel + "::z scale"] = q
        meta_data[channel + "::z unit"] = zunit

        output[channel] = [xres, yres, start, end, q, meta_data]
    return output


def find_2d_data_matrix(name, item):
    """ Checks if an item is a matrix and if it is, returns it's channel name.
    :param name: The name (key) of a found item
    :param item: The item itself
    :return: The name of the matrix data channel
    """
    if item.typeid != DEKTAK_MATRIX:
        return
    if name[:9] != ANY_2D_DATA:
        return
    s = 9 + name[9:].find('/')
    if s == -1:
        return
    if not name[s + 1:] == "Matrix":
        return
    return name[9:s]


def create_meta(hash_table):
    """
    Gets all the metadata out of a hash table.
    :param hash_table: The hash table
    :return: Hash table with all metadata names and values
    """
    container = OrderedDict()
    for key in hash_table.keys():
        if not key.startswith('/MetaData/'):
            continue
        item = hash_table[key]

        if item.typeid == DEKTAK_BOOLEAN:
            metavalue = item.data.b
        elif item.typeid == DEKTAK_SINT32:
            metavalue = item.data.si
        elif item.typeid == DEKTAK_UINT32:
            metavalue = item.data.ui
        elif item.typeid == DEKTAK_SINT64:
            metavalue = item.data.sq
        elif item.typeid == DEKTAK_UINT64:
            metavalue = item.data.uq
        elif item.typeid == DEKTAK_DOUBLE or item.typeid == DEKTAK_FLOAT:
            metavalue = item.data.d
        elif item.typeid == DEKTAK_STRING:
            metavalue = "".join(item.data.buf)
        elif item.typeid == DEKTAK_QUANTITY:
            metavalue = str(item.data.qun.value) + item.data.qun.symbol
        elif item.typeid == DEKTAK_STRING_LIST:
            metavalue = "; ".join(item.data.strlist)
        elif item.typeid == DEKTAK_TERMINATOR:
            metavalue = None
        else:
            # Not really meta data
            continue
        metakey = key.replace("/", "::")
        container[metakey] = metavalue
    return container


def read_item(buf, pos, hash_table, path, abspos=0):
    """
    Reads in the next item out of the buffer and saves it in the hash table.
    May recursively call itself for containers.
    :param buf: The raw data buffer
    :param pos: Current position in the buffer
    :param hash_table: The output hash table
    :param path: Current name to save
    :param abspos: Absolute position in buffer to keep track when calling
    itself
    :return:
    Buffer, new position, hash table with new item in it, new path
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
            item.data.b = True
        elif b8 == '\x00':
            item.data.b = False
        else:
            raise ValueError("Something went wrong.")

    elif item.typeid == DEKTAK_SINT32:
        item.data.si, pos = read_int32(buf, pos, signed=True)

    elif item.typeid == DEKTAK_UINT32:
        item.data.ui, pos = read_int32(buf, pos, signed=False)

    elif item.typeid == DEKTAK_SINT64:
        item.data.sq, pos = read_int64(buf, pos, signed=True)

    elif item.typeid == DEKTAK_UINT64:
        item.data.uq, pos = read_int64(buf, pos, signed=False)

    elif item.typeid == DEKTAK_FLOAT:
        item.data.d, pos = read_float(buf, pos)

    elif item.typeid == DEKTAK_DOUBLE:
        item.data.d, pos = read_double(buf, pos)

    elif item.typeid == DEKTAK_TIME_STAMP:
        time, pos = read_with_check(buf, pos, TIMESTAMP_SIZE)
        item.data.timestamp.append(time)

    elif item.typeid == DEKTAK_STRING:
        item.data.buf, _, pos = read_structured(buf, pos)

    elif item.typeid == DEKTAK_QUANTITY:
        content, _, pos = read_structured(buf, pos)
        item.data.qun, itempos = read_quantunit_content(content, itempos,
                                                        False)

    elif item.typeid == DEKTAK_UNITS:
        content, _, pos = read_structured(buf, pos)
        item.data.qun, itempos = read_quantunit_content(content, itempos, True)

    elif item.typeid == DEKTAK_TERMINATOR:
        pos = len(buf)

    # Container types.
    elif item.typeid == DEKTAK_CONTAINER or \
            item.typeid == DEKTAK_RAW_DATA or \
            item.typeid == DEKTAK_RAW_DATA_2D:
        # TODO find out if maybe better place somewhere else
        content, start, pos = read_structured(buf, pos)
        abspos += start
        while itempos < len(content):
            content, itempos, hash_table, path = read_item(
                buf=content, pos=itempos, hash_table=hash_table,
                path=path, abspos=abspos)

    # Types with string type name
    elif item.typeid == DEKTAK_DOUBLE_ARRAY:
        item.typename, item.data.buf, _, pos = read_named_struct(buf, pos)

    elif item.typeid == DEKTAK_STRING_LIST:
        item.typename, content, start, pos = read_named_struct(buf, pos)
        item.data.strlist = []
        while itempos < len(content):
            s, itempos = read_name(content, itempos)
            item.data.strlist.append(s)

    elif item.typeid == DEKTAK_TYPE_ID:
        item.typename, item.data.buf, _, pos = read_named_struct(buf, pos)

    elif item.typeid == DEKTAK_POS_RAW_DATA:
        if path.startswith('/2D_Data'):
            item.typename, content, _, pos = read_named_struct(buf, pos)

            item.data.rawpos2d.unitx, item.data.rawpos2d.divisorx, itempos = \
                read_dimension2d_content(content, itempos,
                                         item.data.rawpos2d.unitx)
            item.data.rawpos2d.unity, item.data.rawpos2d.divisory, itempos = \
                read_dimension2d_content(content, itempos,
                                         item.data.rawpos2d.unity)

        elif path.startswith('/1_Data'):
            item.typename, content, _, pos = read_named_struct(buf, pos)
            content.position += buf.position

            item.data.rawpos1d.unit, itempos = read_quantunit_content(
                content, itempos, True)
            item.data.rawpos1d.count, itempos = read_int64(content, itempos)

            item.data.rawpos1d.buf = content
            item.data.rawpos1d.buf.position += itempos
            item.data.rawpos1d.buf.length -= itempos

        else:
            # TODO check if should assume 1D here like gwyddion
            raise ValueError

    elif item.typeid == DEKTAK_MATRIX:
        item.typename, pos = read_name(buf, pos)
        item.data.matrix.some_int, pos = read_int32(buf, pos)
        item.data.matrix.another_name, pos = read_name(buf, pos)
        item.data.matrix.buf.length, pos = read_varlen(buf, pos)
        item.data.matrix.yres, pos = read_int32(buf, pos)
        item.data.matrix.xres, pos = read_int32(buf, pos)

        if item.data.matrix.buf.length < 8:  # 2 * sizeof int32
            raise ValueError
        item.data.matrix.buf.length -= 8
        item.data.matrix.buf.p = pos + abspos

        if len(buf) - pos < item.data.matrix.buf.length:
            raise ValueError
        pos += item.data.matrix.buf.length

    else:
        raise ValueError
    hash_table[path] = item
    path = path[:orig_path_len]
    return buf, pos, hash_table, path


def read_quantunit_content(buf, pos, is_unit):
    """
    Reads in a quantity unit: Value, name and symbol.
    :param buf: The buffer
    :param pos: The position in the buffer
    :param is_unit: Whether or not it is a unit
    :return: A quantunit item, filled with value, name and symbol
    """
    quantunit = DektakQuantUnit()
    quantunit.extra = []

    if not is_unit:
        quantunit.value, pos = read_double(buf, pos)

    quantunit.name, pos = read_name(buf, pos)
    quantunit.symbol, pos = read_name(buf, pos)

    if is_unit:
        quantunit.value, pos = read_double(buf, pos)
        res, pos = read_with_check(buf, pos, UNIT_EXTRA)
        quantunit.extra += res

    return quantunit, pos


def read_dimension2d_content(buf, pos, unit):
    """
    Reads in information about a 2d dimension.
    :param buf: The buffer
    :param pos: The position in the buffer
    :param unit: The unit
    :return: The open_topography unit, divisor and new position in the buffer
    """
    unit.value, pos = read_double(buf, pos)
    unit.name, pos = read_name(buf, pos)
    unit.symbol, pos = read_name(buf, pos)
    divisor, pos = read_double(buf, pos)
    unit.extra, pos = read_with_check(buf, pos, UNIT_EXTRA)
    return unit, divisor, pos


def build_matrix(xres, yres, data, q=1):
    """
    Reads a float matrix of given dimensions and multiplies with a scale.
    :param xres: Resolution along x-axis
    :param yres: Resolution along y-axis
    :param data: The raw hex data
    :param q: The scale of the data, a double
    :return: A numpy array, now doubles aswell
    """
    data = ''.join(data)

    # build correct type: 4byte flat, little endian
    dt = np.dtype('f4')  # double
    dt = dt.newbyteorder('<')  # little-endian

    data = np.frombuffer(str.encode(data, "raw_unicode_escape"), dt)
    data = data.copy().reshape((yres, xres))
    data *= q
    return data


def read_name(buf, pos):
    """
    Reads a name.
    :param buf: The buffer
    :param pos: Position in buffer
    :return:
    name, new position in buffer
    """

    # Names always have a physical_sizes of 4 bytes
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
    :param buf: The buffer
    :param pos: Position in buffer
    :return:
    The slice of buffer, where it starts and the new position in the buffer
    """
    length, pos = read_varlen(buf, pos)
    if len(buf) < length or pos > len(buf) - length:
        raise ValueError("Some sizes went wrong.")
    start = pos
    pos += length
    return buf[start:start + length], start, pos


def read_named_struct(buf, pos):
    """
    Same as read_structured but there is a name to it.
    :param buf: The buffer
    :param pos: Position in buffer
    :return:
    Name of the buffer, that buffer, its start and the new position in the
    buffer
    """
    typename, pos = read_name(buf, pos)
    content, start, pos = read_structured(buf, pos)
    return typename, content, start, pos


def read_varlen(buf, pos):
    """
    Reads a length of variable length itself
    :param buf: The buffer
    :param pos: Position in the buffer
    :return:
    The open_topography length and new position in the buffer
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
    :param buf: The buffer
    :param pos: Position in the buffer
    :param signed: Whether of not the int is signed
    :return:
    The int and the new position in the buffer
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
    :param buf: The buffer
    :param pos: Position in the buffer
    :param signed: Whether of not the int is signed
    :return:
    The int and the new position in the buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = "<i4" if signed else "<u4"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_int16(buf, pos, signed=False):
    """
    Reads a 16bit int.
    :param buf: The buffer
    :param pos: Position in the buffer
    :param signed: Whether of not the int is signed
    :return:
    The int and the new position in the buffer
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
    :param buf: The buffer
    :param pos: Position in the buffer
    :return:
    The double and the new position in the buffer
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
    :param buf: The buffer
    :param pos: Position in the buffer
    :return:
    The float and the new position in the buffer
    """
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = np.dtype('f')  # double
    dt = dt.newbyteorder('<')  # little-endian
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[
        0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_with_check(buf, pos, nbytes):
    """
    Reads and returns n bytes.
    :param buf: The input buffer
    :param pos: The current position
    :param nbytes: number of bytes to open_topography in
    :return: The bytes and the new position in the buffer
    """

    if len(buf) < nbytes or len(buf) - nbytes < pos:
        raise ValueError("Some sizes went wrong.")

    out = buf[pos:pos + nbytes]
    pos += int(nbytes)

    out = out[0] if nbytes == 1 else out
    return out, pos
