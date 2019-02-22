import struct
import numpy as np


filePath = "20181205_500um_polished_mounted_100x_vsi.OPDx"


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

buffer = []  # The actual contents of the file



class DektakItem:
    def __init__(self, typename, typeid, data):
        self.typename = typename
        self.typeid = typeid
        self.data = datas


class DektakBuf:
    """
    Stores a position and a length.
    """
    def __init__(self, position=None, length=None):
        self.position = position
        self.length = length


def load_opdx(file_path):
    global buffer
    with open(file_path, "rb") as f:
        # read in file as hexadecimal
        # buffer = [format(byte, '02x') for byte in f.read()]
        buffer = [chr(byte) for byte in f.read()]

        # length of file
        size = len(buffer)
        print(size)

        # check if correct header
        if size < MAGIC_SIZE or ''.join(buffer[:MAGIC_SIZE]) != MAGIC:
            raise ValueError('Invalid file format for Dektak OPDx.')

        hash_table = dict()  # not sure what this is
        buf = DektakBuf(position=0, length=size)
        pos = MAGIC_SIZE

        while pos < size:
            buf, pos, hash_table, path = read_item(buf=buf, pos=pos, hash_table=hash_table, path="")

        # container = find_1d_data(hash, container, str)

        # container = find_2d_data(hash, container, str)

        # container = gwy_container_get_n_items(container)

        # return container


def read_item(buf, pos, hash_table, path):
    global buffer
    print("-------------------------")
    print("started reading new item!")
    print("reading name")
    itempos = 0
    name, pos = read_name(buf, pos)

    path += '/'
    name_start = buf.position + name.position
    name_end = name_start + name.length
    print(buffer[name_start:name_end])
    path += ''.join(buffer[name_start:name_end])
    print("Resulting path: " + path)

    type_id, pos = read_with_check(buf, pos, 1)
    type_id = ord(type_id[0])

    # simple types
    if type_id == DEKTAK_BOOLEAN:
        raise NotImplementedError
    elif type_id == DEKTAK_SINT32 or type_id == DEKTAK_UINT32:
        raise NotImplementedError
    elif type_id == DEKTAK_SINT64 or type_id == DEKTAK_UINT64:
        raise NotImplementedError
    elif type_id == DEKTAK_FLOAT:
        raise NotImplementedError
    elif type_id == DEKTAK_DOUBLE:
        raise NotImplementedError
    elif type_id == DEKTAK_TIME_STAMP:
        raise NotImplementedError
    elif type_id == DEKTAK_STRING:
        raise NotImplementedError
    elif type_id == DEKTAK_QUANTITY:
        raise NotImplementedError
    elif type_id == DEKTAK_UNITS:
        print("Detected DEKTAK units.")
        content, pos = read_structured(buf, pos)
        read_quantunit_content(buf, pos, True)
        raise NotImplementedError
    elif type_id == DEKTAK_TERMINATOR:
        # There are usually some 0xff bytes.  Not sure what to think about them.
        print("Detected terminator.")
        pos = buf.length

    # Container types. Cannot tell any difference between these two.  Raw data purpose
    # seems to be wrapping actual raw data in something container-like.
    elif type_id == DEKTAK_CONTAINER or type_id == DEKTAK_RAW_DATA or type_id == DEKTAK_RAW_DATA_2D:
        print("detected type as container")
        content, pos = read_structured(buf, pos)
        while itempos < content.length:
            print("done reading structured!")
            print("content pos: "+ str(content.position))
            print("content len: " + str(content.length))
            print("itempos: " + str(itempos))
            print("pos: " + str(pos))
            print("RECURSION COMING UP!!!! --------------------------------------------------------")
            content.position += buf.position  # TODO find out if maybe better place somewhere else
            content, itempos, hash_table, path = read_item(buf=content, pos=itempos, hash_table=hash_table, path=path)
            print("DONE WITH RECURSION ------------------------------------------------------------")
            print(content.position)
            print(content.length)
            print(itempos)


    # Types with string type name (i.e.untyped serialised junk we have to know how to read).
    elif type_id == DEKTAK_DOUBLE_ARRAY:
        raise NotImplementedError

    elif type_id == DEKTAK_STRING_LIST:
        raise NotImplementedError

    elif type_id == DEKTAK_TYPE_ID:
        raise NotImplementedError

    elif type_id == DEKTAK_POS_RAW_DATA:
        raise NotImplementedError

    elif type_id == DEKTAK_MATRIX:
        raise NotImplementedError

    else:
        raise ValueError

    return buf, pos, hash_table, path


def read_quantunit_content(buf, pos, is_unit):
    if not is_unit:
        read_double(buf, pos)

    raise NotImplementedError


def read_name(buf, pos):
    name = DektakBuf()
    name.length, pos = read_int32(buf, pos)  # Names always have a size of 4 bytes
    print("In read name.")
    print("Name length: " + str(name.length))
    print("New pos: " + str(pos))
    print("buf pos: " + str(buf.position))
    if buf.length < name.length or pos > buf.length - name.length:
        raise ValueError("TODO")
    name.position = pos
    pos += name.length
    return name, pos


def read_structured(buf, pos):
    print("in read_structured")
    content = DektakBuf()
    content.length, pos = read_varlen(buf, pos)
    print("got length of content: " + str(content.length))
    if buf.length < content.length or pos > buf.length - content.length:
        raise ValueError("TODO")
    content.position = pos
    pos += content.length
    return content, pos


def read_int32(buf, pos):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=4)
    print("Converting the following received hexas to in32: " + str(out))

    out = ''.join(out)
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), "<u4")[0]  # interpret hexadecimal -> int (little-endian)
    print("Resulting int: " + str(out))
    return out, pos


def read_int16(buf, pos):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=2)
    print("Converting the following received hexas to in32: " + str(out))

    out = ''.join(out)
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), "<u2")[0]  # interpret hexadecimal -> int (little-endian)
    print("Resulting int: " + str(out))
    return out, pos


def read_double(buf, pos):
    out, pos = read_with_check(buf=buf, pos=pos, nbytes=8)
    print("Converting the following received hexas to double: " + str(out))

    out = ''.join(out)
    out = np.frombuffer(str.encode(out, "raw_unicode_escape"), "<u8")[0]  # interpret hexadecimal -> int (little-endian)
    print("Resulting double: " + str(out))
    return out, pos


def read_varlen(buf, pos):
    print("in read_varlen")
    lenlen, pos = read_with_check(buf, pos, 1)
    lenlen = np.frombuffer(str.encode(lenlen, "raw_unicode_escape"), "<u1")[0]
    print("read lenlen: " + str(lenlen))
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


def read_with_check(buf, pos, nbytes):
    """
    :param buf: The input buffer
    :param pos: The current position
    :param nbytes: number of bytes to read in
    :return: (read content, new position)
    """
    global buffer
    if buf.length < nbytes or buf.length - nbytes < pos:
        raise ValueError("Some sizes went wrong.")

    print("buffer: " + str(buffer[:30]))
    print("buf.position: " + str(buf.position))
    print("pos: " + str(pos))
    print("nbytes: " + str(nbytes))

    start = buf.position + pos
    end = start + nbytes
    out = buffer[start:end]
    pos += int(nbytes)

    print("out: " + str(out))
    print("new pos: " + str(pos))
    out = out[0] if nbytes == 1 else out
    return out, pos


load_opdx(filePath)