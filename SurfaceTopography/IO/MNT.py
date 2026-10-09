#
# Copyright 2023-2026 Lars Pastewka
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
Reader for Digital Surf Mountains MNT files.

An MNT file is a Microsoft Compound Document (OLE) file with the streams

- ScopedContents: the document, a tree of TLV entries (see below)
- XmlHeader: XML with the version of the Mountains software that wrote the
  file, its serial number and the operators used in the document, stored as
  an MFC `CArchive` string (UTF-16)
- ScopedResults: results of the studies (e.g. parameter tables)
- ImagePreview: JPEG preview of the document

ScopedContents starts with the size of the remaining stream (uint64 LE),
followed by a tree of TLV entries: tag (uint16 LE), size (uint64 LE) and
either data or nested entries. Text is stored as a marker byte (0x04)
followed by UTF-16 LE characters. Entries of the list of document objects
are preceded by an additional size field (uint64 LE).

A Mountains document is a list of objects, each identified by a class
number. Only the original measurement (class 9012) stores height data;
derived layers (levelled, form removed, ...) store the operators and a
reference to their parent and are recomputed by Mountains when the document
is opened. Studies (views, parameter tables, ...) are objects as well.

The measured surface contains

- one axis record per lateral direction with the grid spacing (and its
  unit), the origin (in the display unit) and the number of grid points,
- an axis record for the heights with the height step per count (and its
  unit), the origin (in the display unit) and the minimum and maximum count,
- the heights as int32 counts, stored row by row (x fastest) in
  zlib-compressed blocks, each prefixed with its element offset, number of
  elements and compressed size,
- optionally a mask image of (nx + 2) x (ny + 2) bytes (with a border of
  one pixel), in which 9 marks measured and 22 non-measured points.

The height at a grid point is `count * step + origin`.

Entries of the main container that describe the report page (page size in
pixels, margins in mm, zoom, DPI) are not needed to read the measurement and
are skipped.
"""

import struct
import zlib
from io import BytesIO

import defusedxml.ElementTree as ElementTree
import numpy as np
import olefile

from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..Support.UnitConversion import (
    get_unit_conversion_factor,
    is_length_unit,
    mangle_length_unit_utf8,
)
from ..UniformLineScanAndTopography import Topography
from .binary import RawBuffer, TLVContainer
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase, Skip

# Class numbers of document objects
_CLASS_SURFACE = 9012  # Original (measured) surface
_CLASS_SURFACE_DATA = 9062  # Measurement inside a surface object
_CLASS_MASK = 9056  # Mask of non-measured points
_CLASS_ARRAY = 9048  # Data array of the mask
_CLASS_HEIGHTS = 9049  # Height counts

# Values of the mask image. Only these two values have been observed; any
# value other than `_MASK_MEASURED` is treated as a non-measured point.
_MASK_MEASURED = 9
_MASK_NOT_MEASURED = 22

# Text entries start with this marker byte
_TEXT_MARKER = 0x04

_SIZE_FORMAT = "<Q"


class _Value:
    """
    Fixed-format value of a TLV entry.

    Reads the whole entry, so a value of unexpected size cannot shift the
    position of subsequent entries; such a value is stored as None.
    """

    def __init__(self, name, fmt):
        self._name = name
        self._fmt = fmt

    def name(self, context):
        return self._name

    def from_stream(self, stream_obj, context):
        data = stream_obj.read(context.get("_block_size", 0))
        if len(data) != struct.calcsize(self._fmt):
            return {self._name: None}
        return {self._name: struct.unpack(self._fmt, data)[0]}


class _Text:
    """Text of a TLV entry: a marker byte followed by UTF-16 LE characters."""

    def __init__(self, name):
        self._name = name

    def name(self, context):
        return self._name

    def from_stream(self, stream_obj, context):
        return {self._name: _decode_text(stream_obj.read(context.get("_block_size", 0)))}


def _decode_text(data):
    """
    Decode a text entry.

    Parameters
    ----------
    data : bytes
        Raw data of the entry.

    Returns
    -------
    text : str or None
        Decoded text, or None if the entry is not a text entry.
    """
    if len(data) < 1 or data[0] != _TEXT_MARKER or (len(data) - 1) % 2 != 0:
        return None
    try:
        return data[1:].decode("utf-16-le").rstrip("\x00")
    except UnicodeDecodeError:
        return None


def _decode_cstring(data):
    """
    Decode a Unicode string serialized by MFC's `CArchive`.

    The string starts with 0xFF 0xFE 0xFF (marker for a Unicode string)
    followed by the number of characters as uint8; the values 0xFF and
    0xFFFF indicate that the number follows as uint16 or uint32.

    Parameters
    ----------
    data : bytes
        Serialized string.

    Returns
    -------
    text : str or None
        Decoded string, or None if the data is not a Unicode string.
    """
    if data[:3] != b"\xff\xfe\xff":
        return None
    pos = 3
    length = data[pos]
    pos += 1
    if length == 0xFF:
        (length,) = struct.unpack_from("<H", data, pos)
        pos += 2
        if length == 0xFFFF:
            (length,) = struct.unpack_from("<I", data, pos)
            pos += 4
    return data[pos:pos + 2 * length].decode("utf-16-le", errors="replace")


def _decode_block_array(data, count):
    """
    Decode an array stored in zlib-compressed blocks.

    The array data starts with its total size in bytes (uint64 LE). Each
    block consists of its offset into the array and its length (both in
    elements, uint64 LE and uint32 LE), its compressed size in bytes
    (uint32 LE) and the zlib stream. Blocks are not necessarily in order.

    Parameters
    ----------
    data : bytes
        Raw data of the array.
    count : int
        Number of elements.

    Returns
    -------
    buffer : bytearray
        Decompressed array data.
    element_size : int
        Size of an element in bytes.
    """
    if len(data) < 8:
        raise CorruptFile("Truncated MNT data array.")
    (nb_bytes,) = struct.unpack_from("<Q", data, 0)
    if count == 0 or nb_bytes % count != 0:
        raise CorruptFile(
            f"Size of MNT data array ({nb_bytes} bytes) does not match "
            f"its number of elements ({count})."
        )
    element_size = nb_bytes // count
    buffer = bytearray(nb_bytes)
    nb_decompressed = 0
    pos = 8
    while pos < len(data):
        if pos + 16 > len(data):
            raise CorruptFile("Truncated block header in MNT data array.")
        offset, nb_elements, compressed_size = struct.unpack_from("<QII", data, pos)
        pos += 16
        block = zlib.decompress(data[pos:pos + compressed_size])
        pos += compressed_size
        start = offset * element_size
        if len(block) != nb_elements * element_size or start + len(block) > nb_bytes:
            raise CorruptFile("Inconsistent block in MNT data array.")
        buffer[start:start + len(block)] = block
        nb_decompressed += len(block)
    if nb_decompressed != nb_bytes:
        raise CorruptFile(
            f"MNT data array has {nb_decompressed} bytes, expected {nb_bytes}."
        )
    return buffer, element_size


def _container(children, name=None, **kwargs):
    """TLV container that skips entries not listed in `children`."""
    return TLVContainer(
        children, name=name, size_format=_SIZE_FORMAT, default=Skip(), **kwargs
    )


def _class_number(entries):
    """Class number of a document object, or None."""
    return (entries.get("class") or {}).get("number")


def _guid(entries):
    """Format a GUID entry as string, or return None."""
    if not entries or entries.get("data4") is None:
        return None
    return "{{{:08X}-{:04X}-{:04X}-{}-{}}}".format(
        entries["data1"],
        entries["data2"],
        entries["data3"],
        entries["data4"][:2].hex().upper(),
        entries["data4"][2:].hex().upper(),
    )


#
# Layout of the ScopedContents stream
#

# Class of an object
_class = _container({0x0001: _Value("number", "<I")}, "class")

# GUID of an object
_guid_layout = _container(
    {
        0x0001: _Value("data1", "<I"),
        0x0002: _Value("data2", "<H"),
        0x0003: _Value("data3", "<H"),
        0x0004: _Value("data4", "8s"),
    },
    "guid",
)

# Generic object: class, body (parsed depending on the class) and GUID
_object = _container(
    {
        0x0001: _class,
        0x0002: RawBuffer("body", lazy=False),
        0x0003: _guid_layout,
    }
)

# Scale of an axis
_axis_scale = _container(
    {
        0xFFFF: _container({0x0001: _Value("index", "<I"), 0x0002: _Text("name")}, "label"),
        0x0001: _Value("step", "<d"),  # Grid spacing or height per count
        0x0002: _Text("unit"),  # Unit of the step
        0x0003: _Text("display_unit"),  # Unit used for display and origin
        0x0004: _Value("unit_ratio", "<d"),  # Display unit in units of `unit`
        0x0005: _Value("origin", "<d"),  # In display unit
    },
    "scale",
)

# Lateral axis: scale and number of grid points
_lateral_axis = _container(
    {
        0xFFFF: _container(
            {
                0xFFFF: _axis_scale,
                0x0001: _Value("nb_grid_pts", "<I"),
                0x0002: _Value("first", "<I"),
                0x0003: _Value("last", "<I"),
            },
            "axis",
        )
    }
)

# Height axis: scale and range of counts
_height_axis = _container(
    {
        0xFFFF: _axis_scale,
        0x0001: _Value("min", "<i"),
        0x0002: _Value("max", "<i"),
    }
)

# Data array: number of elements and compressed blocks
_array = _container(
    {
        0x0001: _class,
        0x0002: _container(
            {0x0001: _Value("count", "<I"), 0x0002: RawBuffer("data", lazy=False)},
            "array",
        ),
    }
)

# Mask of non-measured points
_mask = _container(
    {
        0x0001: _class,
        0x0002: _container(
            {
                0x0001: _Value("width", "<I"),  # Without border
                0x0002: _Value("height", "<I"),  # Without border
                0x0005: _array,
            },
            "mask",
        ),
    }
)

# Measurement inside a surface object (class 9062)
_surface_data = _container(
    {
        0xFFFF: _container(
            {
                0xFFFF: _container(
                    {
                        0xFFFF: _container(
                            {0xFFFF: _container({0x0001: _Text("name")}, "description")},
                            "header",
                        ),
                        0x0001: _lateral_axis,
                        0x0002: _lateral_axis,
                        0x0003: _mask,
                    },
                    "grid",
                ),
                0x0001: _array,
                0x0002: _height_axis,
            },
            "measurement",
        )
    }
)

# Body of a surface object (class 9012)
_surface = _container(
    {
        0xFFFF: _container(
            {
                # GUID referenced as `StudiableGUID` in the XML header
                0xFFFF: _container({0x0001: _guid_layout}, "identity"),
                0x0004: _Text("title"),
            },
            "header",
        ),
        0x0001: _object,
    }
)

# Top level of the ScopedContents stream
_scoped_contents = _container(
    {
        0x0001: _container(
            {
                # Version of the Mountains software that wrote the file
                0x00C8: _container(
                    {
                        0x0001: _Value("major", "<I"),
                        0x0002: _Value("minor", "<I"),
                        0x0003: _Value("patch", "<I"),
                        0x0004: _Value("build", "<I"),
                    },
                    "software_version",
                ),
                # Serial number of the Mountains software installation (not of
                # the instrument)
                0x00CB: _Text("software_serial_number"),
                0x02BD: _container(
                    {
                        0x0001: _Value("nb_objects", "B"),
                        0x0002: _container(
                            {
                                0x0001: TLVContainer(
                                    {0x0001: RawBuffer("object", lazy=False)},
                                    name="objects",
                                    size_format=_SIZE_FORMAT,
                                    entry_prefix_format=_SIZE_FORMAT,
                                    default=Skip(),
                                )
                            },
                            "document",
                        ),
                    },
                    "contents",
                ),
            },
            "main",
        )
    }
)


def _parse(layout, data):
    """Parse raw data with a TLV layout."""
    return layout.from_stream(BytesIO(data), {"_block_size": len(data)})


def _parse_object(data):
    """Parse a document object into its class number and body."""
    entries = _parse(_object, data)
    return _class_number(entries), entries.get("body", {}).get("_raw")


def _as_list(value):
    """Repeated TLV entries are stored as a list, single ones not."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class MNTReader(ReaderBase):
    _format = "mnt"
    _mime_types = ["application/x-digitalsurf-mnt"]
    _file_extensions = ["mnt"]

    _name = "Digital Surf Mountains"
    _description = """
File format of the Digital Surf Mountains software. This format is a
Microsoft Compound Document (OLE) file containing a TLV-encoded document with
zlib-compressed height data. The reader returns the original measurements
stored in the document; derived layers are not stored in the file but
recomputed by Mountains.
"""

    def __init__(self, fobj):
        """
        Load Digital Surf Mountains data files.

        Arguments
        ---------
        fobj : filename or file object
            File or data stream to open.
        """
        self._fobj = fobj

        with OpenFromAny(fobj, "rb") as f:
            # Check the signature before reading the whole file, since format
            # detection tries this reader on arbitrary (possibly large) files
            if f.read(8) != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                raise FileFormatMismatch("Not an OLE compound document.")
            f.seek(0)
            file_data = f.read()

        try:
            ole = olefile.OleFileIO(file_data)
        except Exception as e:
            raise FileFormatMismatch(f"Failed to parse OLE file: {e}")

        try:
            if not ole.exists("ScopedContents"):
                raise FileFormatMismatch("Missing ScopedContents stream.")
            scoped_contents = ole.openstream("ScopedContents").read()
            xml_header = (
                ole.openstream("XmlHeader").read() if ole.exists("XmlHeader") else None
            )
        finally:
            ole.close()

        if len(scoped_contents) < 8:
            raise CorruptFile("ScopedContents stream is too short.")
        main = _parse(_scoped_contents, scoped_contents[8:]).get("main")
        if main is None:
            raise CorruptFile("Missing main container in ScopedContents stream.")

        software = self._software_metadata(main, xml_header)

        self._channels = []
        self._arrays = []
        objects = _as_list(
            main.get("contents", {}).get("document", {}).get("objects", {}).get("object")
        )
        for object_data in objects:
            class_number, body = _parse_object(object_data["_raw"])
            if class_number != _CLASS_SURFACE or body is None:
                continue
            channel = self._surface_channel(body, software)
            if channel is not None:
                self._channels.append(channel)

        if len(self._channels) == 0:
            raise CorruptFile("MNT file does not contain a measured surface.")

    @staticmethod
    def _software_metadata(main, xml_header):
        """Metadata on the Mountains software that wrote the file."""
        software = {}
        version = main.get("software_version")
        if version is not None and None not in (
            version.get("major"),
            version.get("minor"),
            version.get("patch"),
            version.get("build"),
        ):
            software["version"] = (
                f"{version['major']}.{version['minor']}.{version['patch']}."
                f"{version['build']}"
            )
        serial_number = main.get("software_serial_number")
        if serial_number:
            software["serial_number"] = serial_number

        if xml_header is not None:
            text = _decode_cstring(xml_header)
            if text is not None:
                try:
                    root = ElementTree.fromstring(text)
                except ElementTree.ParseError:
                    root = None
                if root is not None:
                    product_name = root.findtext("ProductName")
                    if product_name:
                        software["name"] = product_name
                    build_date = root.findtext("BuildDate")
                    if build_date:
                        software["build_date"] = build_date
                    operators = [
                        e.text for e in root.iterfind("OperatorsInUse/Operator") if e.text
                    ]
                    if operators:
                        software["operators"] = operators
        return software

    def _surface_channel(self, body, software):
        """
        Create channel information for a surface object.

        Returns None if the object does not contain a height map.
        """
        surface = _parse(_surface, body)
        guid = _guid(surface.get("header", {}).get("identity", {}).get("guid"))
        measurement = surface.get(0x0001)
        if (
            not isinstance(measurement, dict)
            or _class_number(measurement) != _CLASS_SURFACE_DATA
        ):
            return None
        surface_data = _parse(_surface_data, measurement["body"]["_raw"]).get(
            "measurement", {}
        )

        grid = surface_data.get("grid", {})
        x_axis = grid.get(0x0001, {}).get("axis")
        y_axis = grid.get(0x0002, {}).get("axis")
        heights = surface_data.get(0x0001)
        height_axis = surface_data.get(0x0002)
        if (
            x_axis is None
            or y_axis is None
            or heights is None
            or _class_number(heights) != _CLASS_HEIGHTS
            or height_axis is None
        ):
            return None
        x_scale = x_axis.get("scale", {})
        y_scale = y_axis.get("scale", {})
        z_scale = height_axis.get("scale", {})
        for scale in (x_scale, y_scale, z_scale):
            if scale.get("step") is None or not scale.get("unit"):
                raise CorruptFile("Incomplete axis definition in MNT file.")

        # Heights are reported in the display unit of the height axis
        unit = mangle_length_unit_utf8(z_scale.get("display_unit") or z_scale["unit"])
        if not is_length_unit(unit):
            # Not a height map (e.g. intensity)
            return None

        def length(value, from_unit):
            return value * get_unit_conversion_factor(
                mangle_length_unit_utf8(from_unit), unit
            )

        nb_grid_pts = (x_axis["nb_grid_pts"], y_axis["nb_grid_pts"])
        physical_sizes = (
            nb_grid_pts[0] * length(x_scale["step"], x_scale["unit"]),
            nb_grid_pts[1] * length(y_scale["step"], y_scale["unit"]),
        )
        height_scale_factor = length(z_scale["step"], z_scale["unit"])
        height_offset = (
            0.0 if z_scale.get("origin") is None else z_scale["origin"]
        ) * get_unit_conversion_factor(
            mangle_length_unit_utf8(z_scale.get("display_unit") or z_scale["unit"]),
            unit,
        )

        array = heights.get("array", {})
        if array.get("count") != nb_grid_pts[0] * nb_grid_pts[1]:
            raise CorruptFile(
                f"Number of heights ({array.get('count')}) does not match the "
                f"grid ({nb_grid_pts[0]} x {nb_grid_pts[1]})."
            )

        # Mask of non-measured points, if present
        mask = grid.get(0x0003, {})
        mask_array = None
        if _class_number(mask) == _CLASS_MASK:
            mask_array = mask.get("mask", {}).get(0x0005)
            if (
                mask_array is None
                or _class_number(mask_array) != _CLASS_ARRAY
                or (mask["mask"].get("width"), mask["mask"].get("height"))
                != nb_grid_pts
            ):
                raise CorruptFile("Inconsistent mask in MNT file.")
            mask_array = mask_array.get("array", {})

        def axis_metadata(scale):
            return {
                key: scale[key]
                for key in ("step", "unit", "origin", "display_unit")
                if scale.get(key) is not None
            }

        name = grid.get("header", {}).get("description", {}).get("name")
        title = surface.get("header", {}).get("title")
        raw_metadata = {
            "name": name,
            "title": title,
            "guid": guid,
            "axes": {
                "x": axis_metadata(x_scale),
                "y": axis_metadata(y_scale),
                "z": {
                    **axis_metadata(z_scale),
                    "min_count": height_axis.get("min"),
                    "max_count": height_axis.get("max"),
                },
            },
            "software": software,
        }
        raw_metadata = {k: v for k, v in raw_metadata.items() if v not in (None, {})}

        self._arrays.append(
            {
                "heights": array,
                "mask": mask_array,
                "height_offset": height_offset,
            }
        )
        return ChannelInfo(
            self,
            len(self._channels),
            name=name or title or "Default",
            dim=2,
            nb_grid_pts=nb_grid_pts,
            physical_sizes=physical_sizes,
            height_scale_factor=height_scale_factor,
            uniform=True,
            unit=unit,
            info={"raw_metadata": raw_metadata},
        )

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        channel = self._channels[channel_index]
        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")
        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")
        physical_sizes = self._check_physical_sizes(
            physical_sizes, channel.physical_sizes
        )

        nx, ny = channel.nb_grid_pts
        arrays = self._arrays[channel_index]

        # Heights are stored row by row (x fastest)
        buffer, element_size = _decode_block_array(
            arrays["heights"]["data"]["_raw"], arrays["heights"]["count"]
        )
        if element_size not in (2, 4):
            raise CorruptFile(
                f"Unsupported size of height values ({element_size} bytes)."
            )
        counts = np.frombuffer(buffer, dtype=f"<i{element_size}").reshape(ny, nx).T
        heights = counts * channel.height_scale_factor + arrays["height_offset"]

        if arrays["mask"] is not None:
            mask_buffer, _ = _decode_block_array(
                arrays["mask"]["data"]["_raw"], arrays["mask"]["count"]
            )
            # The mask image has a border of one pixel
            mask = np.frombuffer(mask_buffer, dtype=np.uint8)
            if mask.size != (nx + 2) * (ny + 2):
                raise CorruptFile("Size of mask does not match the grid.")
            mask = mask.reshape(ny + 2, nx + 2)[1:-1, 1:-1].T
            undefined = mask != _MASK_MEASURED
            if undefined.any():
                heights = np.ma.masked_array(heights, mask=undefined)

        _info = channel.info.copy()
        _info.update(info)

        return Topography(
            heights,
            physical_sizes=physical_sizes,
            unit=channel.unit,
            info=_info,
            periodic=periodic,
        )

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
