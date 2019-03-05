import numpy as np

from PyCo.Topography.ParallelFromFile import TopographyLoader
from PyCo.Topography import Topography



MAGIC = "VCA DATA\x01\x00\x00\x55"
# MAGIC = "0x560x430x410x200x440x410x540x410x10x00x00x55"
# MAGIC = "564341204441544101000055"
# MAGIC = "0x560x430x410x200x440x410x540x410x010x000x000x55"
MAGIC_SIZE = 12

DEKTAK_MATRIX = 0x00          # Too lazy to assign an actual type id?
DEKTAK_BOOLEAN = 0x01         # Takes value 0 and 1
DEKTAK_SINT32 = 0x06
DEKTAK_UINT32 = 0x07
DEKTAK_SINT64 = 0x0a
DEKTAK_UINT64 = 0x0b
DEKTAK_FLOAT = 0x0c           # Single precision float
DEKTAK_DOUBLE = 0x0d          # Double precision float
DEKTAK_TYPE_ID = 0x0e         # Compound type holding some kind of type id
DEKTAK_STRING = 0x12          # Free-form string value
DEKTAK_QUANTITY = 0x13        # Value with units (compound type)
DEKTAK_TIME_STAMP = 0x15      # Datetime (string/9-byte binary)
DEKTAK_UNITS = 0x18           # Units (compound type)
DEKTAK_DOUBLE_ARRAY = 0x40    # Raw data array, in XML Base64-encoded
DEKTAK_STRING_LIST = 0x42     # List of Str
DEKTAK_RAW_DATA = 0x46        # Parent/wrapper tag of raw data
DEKTAK_RAW_DATA_2D = 0x47     # Parent/wrapper tag of raw data
DEKTAK_POS_RAW_DATA = 0x7c    # Base64-encoded positions, not sure how it differs from 64
DEKTAK_CONTAINER = 0x7d       # General nested data structure
DEKTAK_TERMINATOR = 0x7f      # Always the last item. Usually a couple of 0xff bytes inside.

TIMESTAMP_SIZE = 9
UNIT_EXTRA = 12
DOUBLE_ARRAY_EXTRA = 5

MEAS_SETTINGS = "/MetaData/MeasurementSettings"
RAW_1D_DATA = "/1D_Data/Raw"
ANY_2D_DATA = "/2D_Data/"


class TopographyLoaderOPDx(TopographyLoader):

    # Reads in the positions of all the data and metadata, runs quickly.
    def __init__(self, file_path, size=None, unit=None, info=None):
        super().__init__(size, unit, info)

        with open(file_path, "rb") as f:
            # read in file as hexadecimal
            # buffer = [format(byte, '02x') for byte in f.read()]
            self.buffer = [chr(byte) for byte in f.read()]
            # length of file
            size = len(self.buffer)

            # check if correct header
            if size < MAGIC_SIZE or ''.join(self.buffer[:MAGIC_SIZE]) != MAGIC:
                raise ValueError('Invalid file format for Dektak OPDx.')

            self.hash_table = dict()

            pos = MAGIC_SIZE

            while pos < size:
                buf, pos, self.hash_table, path = read_item(buf=self.buffer, pos=pos, hash_table=self.hash_table, path="")

    # Gets the actual data and metadata from the positions. Slower, but maybe able to speed this up in the future.
    def topography(self):
        channels = find_2d_data(self.hash_table, self.buffer)

        data, metadata = channels['Raw']
        unit = metadata.pop('z unit', None)
        info = {'unit': unit,
                'metadata': metadata}

        topo = Topography(
            heights=data,
            size=data.shape,
            info=info)

        return topo


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
    """
    Quantities have a name, symbol and value.
    """
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
    """
    Stores a position and a length.
    """
    def __init__(self, position=None, length=None):
        self.position = position
        self.length = length


def find_2d_data_matrix(name, item):
    if item.typeid != DEKTAK_MATRIX:
        return
    if name[:9] != ANY_2D_DATA:
        return
    s = 9 + name[9:].find('/')
    if s == -1:
        return
    if not name[s+1:] == "Matrix":
        return
    return name[9:s]


def find_2d_data(hash_table, buf):
    output = dict()
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

        meta_data["xres"] = xres
        meta_data["yres"] = yres

        # TODO: multiply value by interpret(item.data.qun.symbol)
        string = string[:length]
        string += "/Dimension1Extent"
        item = hash_table[string]
        yreal = item.data.qun.value
        yunit = item.data.qun.symbol

        meta_data["Height value"] = yreal
        meta_data["Height unit"] = yunit

        string = string[:length]
        string += "/Dimension2Extent"
        item = hash_table[string]
        xreal = item.data.qun.value
        xunit = item.data.qun.symbol

        meta_data["Width value"] = xreal
        meta_data["Width unit"] = xunit

        string = string[:length]
        string += "/DataScale"
        item = hash_table[string]
        q = item.data.qun.value
        zunit = item.data.qun.symbol

        meta_data["z scale"] = q
        meta_data["z unit"] = zunit

        # TODO: maybe speed up if possible
        data = buf[start:end]
        pos = 0
        rawdata = []
        for line in range(yres):
            row = []
            for point in range(xres):
                value, pos = read_float(data, pos)
                row.append(value*q)
            rawdata.append(np.array(row))

        rawdata = np.array(rawdata)
        output[channel] = (rawdata, meta_data)
    return output


def create_meta(hash_table):
    container = dict()
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


def load_opdx(file_path):
    with open(file_path, "rb") as f:
        # read in file as hexadecimal
        # buffer = [format(byte, '02x') for byte in f.read()]
        buffer = [chr(byte) for byte in f.read()]
        # length of file
        size = len(buffer)

        # check if correct header
        if size < MAGIC_SIZE or ''.join(buffer[:MAGIC_SIZE]) != MAGIC:
            raise ValueError('Invalid file format for Dektak OPDx.')

        hash_table = dict()

        pos = MAGIC_SIZE

        while pos < size:
            buf, pos, hash_table, path = read_item(buf=buffer, pos=pos, hash_table=hash_table, path="")

        # TODO: find_1d_data

        data_2d = find_2d_data(hash_table, buffer)

        """
        rawdata, metadata = data_2d["Image"]
        print(rawdata.shape)
        for key in metadata.keys():
            print(key + ": " + str(metadata[key]))
        """
        return data_2d


def find_1d_data(hash_table, buf):
    raise NotImplementedError


def read_item(buf, pos, hash_table, path, abspos=0):
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
        item.data.qun, itempos = read_quantunit_content(content, itempos, False)

    elif item.typeid == DEKTAK_UNITS:
        content, _, pos = read_structured(buf, pos)
        item.data.qun, itempos = read_quantunit_content(content, itempos, True)

    elif item.typeid == DEKTAK_TERMINATOR:
        # There are usually some 0xff bytes.  Not sure what to think about them.
        pos = len(buf)

    # Container types. Cannot tell any difference between these two.  Raw data purpose
    # seems to be wrapping actual raw data in something container-like.
    elif item.typeid == DEKTAK_CONTAINER or item.typeid == DEKTAK_RAW_DATA or item.typeid == DEKTAK_RAW_DATA_2D:
        content, start, pos = read_structured(buf, pos)  # TODO find out if maybe better place somewhere else
        abspos += start
        while itempos < len(content):
            content, itempos, hash_table, path = read_item(buf=content, pos=itempos, hash_table=hash_table, path=path, abspos= abspos)

    # Types with string type name (i.e.untyped serialised junk we have to know how to read).
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
                read_dimension2d_content(content, itempos, item.data.rawpos2d.unitx)
            item.data.rawpos2d.unity, item.data.rawpos2d.divisory, itempos = \
                read_dimension2d_content(content, itempos, item.data.rawpos2d.unity)

        elif path.startswith('/1_Data'):
            item.typename, content, _, pos = read_named_struct(buf, pos)
            content.position += buf.position

            item.data.rawpos1d.unit, itempos = read_quantunit_content(content, itempos, True)
            item.data.rawpos1d.count, itempos = read_int64(content, itempos)

            item.data.rawpos1d.buf = content
            item.data.rawpos1d.buf.position += itempos
            item.data.rawpos1d.buf.length -= itempos

        else:
            raise ValueError  # TODO check if should assume 1D here like gwyddion

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

        # TODO Here might be some error if one appears. If not, this is exactly as planned.
        if len(buf) - pos < item.data.matrix.buf.length:
            raise ValueError
        pos += item.data.matrix.buf.length

    else:
        raise ValueError
    hash_table[path] = item  # TODO: Might not be correct like this
    path = path[:orig_path_len]
    return buf, pos, hash_table, path


def read_named_struct(buf, pos):
    typename, pos = read_name(buf, pos)
    content, start, pos = read_structured(buf, pos)
    return typename, content, start, pos


def read_quantunit_content(buf, pos, is_unit):
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
    unit.value, pos = read_double(buf, pos)
    unit.name, pos = read_name(buf, pos)
    unit.symbol, pos = read_name(buf, pos)
    divisor, pos = read_double(buf, pos)
    unit.extra, pos = read_with_check(buf, pos, UNIT_EXTRA)
    return unit, divisor, pos


def read_name(buf, pos):
    """
    Returns a name after reading it's size.
    """
    length, pos = read_int32(buf, pos)  # Names always have a size of 4 bytes
    if len(buf) < length or pos > len(buf) - length:
        raise ValueError("Some sizes went wrong.")
    position = pos

    name = buf[position:position+length]
    name = "".join(s for s in name)
    pos += length
    return name, pos


def read_structured(buf, pos):
    length, pos = read_varlen(buf, pos)
    if len(buf) < length or pos > len(buf) - length:
        raise ValueError("Some sizes went wrong.")
    start = pos
    pos += length
    return buf[start:start+length], start, pos


def read_varlen(buf, pos):
    lenlen, pos = read_with_check(buf, pos, 1)
    lenlen = np.frombuffer(str.encode(lenlen, "raw_unicode_escape"), "<u1")[0]
    if lenlen == 1:
        length, pos = read_with_check(buf, pos, 1)
        length = np.frombuffer(str.encode(length, "raw_unicode_escape"), "<u1")[0]
    elif lenlen == 2:
        length, pos = read_int16(buf, pos)
    elif lenlen == 4:
        length, pos = read_int32(buf, pos)
    else:
        raise ValueError
    return length, pos


def read_int64(buf, pos, signed=False):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=8)
    out = ''.join(out)
    dt = "<i8" if signed else "<u8"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_int32(buf, pos, signed=False):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = "<i4" if signed else "<u4"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_int16(buf, pos, signed=False):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=2)
    out = ''.join(out)
    dt = "<i2" if signed else "<u2"
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_double(buf, pos):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=8)
    out = ''.join(out)
    dt = np.dtype('d')  # double
    dt = dt.newbyteorder('<')  # little-endian
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_float(buf, pos):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    out = ''.join(out)
    dt = np.dtype('f')  # double
    dt = dt.newbyteorder('<')  # little-endian
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), dt)[0]  # interpret hexadecimal -> int (little-endian)
    return out, pos


def read_with_check(buf, pos, nbytes):
    """
    :param buf: The input buffer
    :param pos: The current position
    :param nbytes: number of bytes to read in
    :return: (read content, new position)
    """

    if len(buf) < nbytes or len(buf) - nbytes < pos:
        raise ValueError("Some sizes went wrong.")

    out = buf[pos:pos+nbytes]
    pos += int(nbytes)

    out = out[0] if nbytes == 1 else out
    return out, pos