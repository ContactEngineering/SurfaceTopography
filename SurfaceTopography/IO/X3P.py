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
#
# Reference information and implementations:
# https://sourceforge.net/p/open-gps/mwiki
#

import hashlib
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from zipfile import ZipFile

import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch
from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import get_unit_conversion_factor
from .binary import BinaryArray, XMLStructure, ZipContainer
from .expr import C, Cond, F, Lit, Tup, V
from .Reader import Check, DeclarativeReaderBase, MagicMatch

_REVISION = "ISO5436 - 2000"
_FEATURE_TYPE = "SUR"

# Convenience accessors into the parsed main.xml
_axes = C.main.Record1.Axes
_dims = C.main.Record3.MatrixDimension
_data_link = C.main.Record3.DataLink

# The z axis may carry an increment (the height scale factor; typically
# missing or unity) and an offset: h = raw * Increment + Offset. The
# offset is typically present for integer data types and is folded into
# the raw data so that the height scale factor can still be applied
# through the pipeline.
_z_increment = F.get(_axes.CZ, "Increment", None)
_z_offset = F.get(_axes.CZ, "Offset", 0.0)
_fold_offset_fac = F.get(_axes.CZ, "Increment", 1.0)

# Optional Record2 metadata
_record2 = F.get(C.main, "Record2", {})
_instrument = F.get(_record2, "Instrument", {})
_model = F.get(_instrument, "Model", "unknown")
_manufacturer = F.get(_instrument, "Manufacturer", "unknown")
_instrument_version = F.get(_instrument, "Version", "unknown")
_serial = F.get(_instrument, "Serial", None)
_date = F.get(_record2, "Date", None)

# Assemble a human-readable instrument name from model, manufacturer and
# version, skipping fields reported as unknown
_instrument_name = Cond(
    _model == "unknown",
    Cond(
        _manufacturer != "unknown",
        Cond(
            _instrument_version != "unknown",
            _manufacturer + " (version " + _instrument_version + ")",
            _manufacturer,
        ),
        None,
    ),
    Cond(
        _manufacturer != "unknown",
        Cond(
            _instrument_version != "unknown",
            _model + " (" + _manufacturer + ", version "
            + _instrument_version + ")",
            _model + " (" + _manufacturer + ")",
        ),
        Cond(
            _instrument_version != "unknown",
            _model + " (version " + _instrument_version + ")",
            _model,
        ),
    ),
)


class X3PReader(DeclarativeReaderBase):
    _format = "x3p"
    _mime_types = ["application/x-iso5436-2-spm"]
    _file_extensions = ["x3p"]

    _name = "XML 3D surface profile (X3P)"
    _description = """
X3P is an XML-based container format for the exchange of surface topography
data, standardized in ISO 25178-72 (originally ISO 5436-2, Geometrical
Product Specifications — Surface texture). The full specification of the
format can be found [here](http://www.opengps.eu/).
"""

    # ZIP magic bytes (PK\x03\x04)
    _MAGIC_ZIP = b"PK\x03\x04"

    # A ZIP magic can only tell that the file *may* be an X3P; detection
    # needs to try parsing, so magic-byte detection stays a custom method
    # and no `_magic` is declared
    _magic = None

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC_ZIP):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC_ZIP):
            # ZIP file, could be X3P - need full parsing to confirm
            return MagicMatch.MAYBE
        return MagicMatch.NO

    _file_layout = ZipContainer(
        [
            (
                "main.xml",
                XMLStructure(
                    name="main",
                    converters={
                        "SizeX": F.int(V),
                        "SizeY": F.int(V),
                        "SizeZ": F.int(V),
                        "Increment": F.float(V),
                        "Offset": F.float(V),
                    },
                ),
            ),
            Check(
                F.get(C.main, "Record1", None) != None,  # noqa: E711
                FileFormatMismatch,
                "'Record1' not found in 'main.xml'.",
            ),
            Check(
                F.get(C.main, "Record3", None) != None,  # noqa: E711
                FileFormatMismatch,
                "'Record3' not found in 'main.xml'.",
            ),
            Check(
                C.main.Record1.Revision == _REVISION,
                CorruptFile,
                f"Revision should be '{_REVISION}'.",
            ),
            Check(
                C.main.Record1.FeatureType == _FEATURE_TYPE,
                CorruptFile,
                f"FeatureType should be '{_FEATURE_TYPE}'.",
            ),
            Check(
                _axes.CX.AxisType == "I",
                CorruptFile,
                "CX AxisType is not 'I'. Don't know how to handle this.",
            ),
            Check(
                _axes.CY.AxisType == "I",
                CorruptFile,
                "CY AxisType is not 'I'. Don't know how to handle this.",
            ),
            Check(
                _axes.CZ.AxisType == "A",
                CorruptFile,
                "CZ AxisType is not 'A'. Don't know how to handle this.",
            ),
            Check(
                _dims.SizeZ == 1,
                CorruptFile,
                "Z dimension has extend != 1. Volumetric data is not "
                "supported.",
            ),
            # The height data member is named within the XML index. Data
            # types are per ISO 5436-2: I = uint16, L = uint32,
            # F = float32, D = float64.
            (
                _data_link.PointDataLink,
                BinaryArray(
                    "data",
                    Tup(_dims.SizeY, _dims.SizeX),
                    F.dtype(
                        Lit({"I": "<u2", "L": "<u4", "F": "<f4", "D": "<f8"})[
                            _axes.CZ.DataType
                        ]
                    ),
                    # Transpose to (nx, ny) order and fold the z offset
                    # into the raw data (dtype-preserving when there is
                    # no offset)
                    conversion_fun=Cond(
                        _z_offset != 0,
                        F.transpose(V) + _z_offset / _fold_offset_fac,
                        F.transpose(V),
                    ),
                ),
            ),
            # Optional member marking invalid (undefined) data points,
            # one bit per point, least-significant bit first, in the same
            # point order as the height data (ISO 5436-2). This is the
            # only way invalid points can be encoded for integer data
            # types. The conversion turns the packed bits into a boolean
            # undefined-data mask.
            (
                F.get(_data_link, "ValidPointsLink", None),
                BinaryArray(
                    "undefined_mask",
                    Tup((_dims.SizeX * _dims.SizeY + 7) // 8),
                    F.dtype("u1"),
                    conversion_fun=F.logical_not(
                        F.transpose(
                            F.reshape(
                                F.unpackbits(V, _dims.SizeX * _dims.SizeY),
                                Tup(_dims.SizeY, _dims.SizeX),
                            )
                        )
                    ),
                ),
                True,
            ),
        ],
        # `main.xml` is what identifies the format
        mismatch_error=FileFormatMismatch,
    )

    _channel_bindings = [
        {
            "name": "Default",
            "dim": 2,
            "nb_grid_pts": Tup(_dims.SizeX, _dims.SizeY),
            "physical_sizes": Tup(
                _axes.CX.Increment * _dims.SizeX,
                _axes.CY.Increment * _dims.SizeY,
            ),
            "height_scale_factor": _z_increment,
            "uniform": True,
            "periodic": False,
            "unit": "m",  # Unit is always meters
            "info": {
                "acquisition_time": Cond(
                    _date != None,  # noqa: E711
                    F.parse_datetime(_date),
                    None,
                ),
                "instrument": {
                    "name": _instrument_name,
                    "vendor": Cond(
                        _manufacturer != "unknown", _manufacturer, None
                    ),
                    "serial": Cond(
                        _serial.isin("not available", "unknown"),
                        None,
                        _serial,
                    ),
                },
                "raw_metadata": C.main,
            },
            "data": C.data,
            "mask": {"source": C.undefined_mask, "rule": V},
        }
    ]


def write_x3p(
    self,
    fobj,
    dtype="D",
    manufacturer="SurfaceTopography",
    model="Python Library",
    version=None,
):
    """
    Write topography to an X3P file (ISO 5436-2 / ISO 25178-72 format).

    X3P is a container format conforming to the ISO 5436-2 standard for
    surface texture data exchange.

    Parameters
    ----------
    self : :obj:`Topography`
        The topography to write.
    fobj : str or file-like object
        File path or file-like object to write to.
    dtype : str, optional
        Data type for height values. Options are:
        - 'D' : 64-bit float (default)
        - 'F' : 32-bit float
        - 'L' : 32-bit unsigned integer (requires height_scale_factor)
        - 'I' : 16-bit unsigned integer (requires height_scale_factor)
    manufacturer : str, optional
        Manufacturer name to include in metadata. (Default: 'SurfaceTopography')
    model : str, optional
        Model/software name. (Default: 'Python Library')
    version : str, optional
        Version string. If None, uses the SurfaceTopography version.
    """
    if self.dim != 2:
        raise ValueError("X3P format only supports 2D topographies.")

    if self.communicator is not None and self.communicator.size > 1:
        raise RuntimeError("X3P writer does not support MPI parallelization.")

    # Data type mapping (reverse of reader)
    dtype_map = {
        "I": np.dtype("<u2"),
        "L": np.dtype("<u4"),
        "F": np.dtype("f4"),
        "D": np.dtype("f8"),
    }

    if dtype not in dtype_map:
        raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {list(dtype_map.keys())}")

    np_dtype = dtype_map[dtype]

    # Get version if not provided
    if version is None:
        try:
            from .. import __version__
            version = __version__
        except ImportError:
            version = "unknown"

    # Convert to meters (X3P uses meters as the base unit)
    unit = self.unit if self.unit is not None else "m"
    scale_to_meters = get_unit_conversion_factor(unit, "m")

    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes

    # Grid spacing in meters
    dx = sx * scale_to_meters / nx
    dy = sy * scale_to_meters / ny

    # Get height data and convert to meters. Undefined (masked) data points
    # are encoded as NaN for floating-point data types; for integer data
    # types they are stored as the minimum raw value and marked as invalid
    # in the validity file (written below).
    heights = np.ma.filled(self.heights(), np.nan) * scale_to_meters
    invalid_mask = np.isnan(heights)
    has_invalid = invalid_mask.any()

    # For integer types, we need to scale the data
    if dtype in ("I", "L"):
        height_min = np.nanmin(heights)
        height_max = np.nanmax(heights)
        height_range = height_max - height_min
        if height_range == 0:
            height_range = 1.0
        if has_invalid:
            # NaN cannot be cast to an integer; the actual value stored for
            # invalid points is arbitrary since they are flagged in the
            # validity file
            heights = np.where(invalid_mask, height_min, heights)
        if dtype == "I":
            scale_factor = height_range / 65535
            heights = ((heights - height_min) / scale_factor).astype(np_dtype)
        else:  # L
            scale_factor = height_range / 4294967295
            heights = ((heights - height_min) / scale_factor).astype(np_dtype)
        z_offset = height_min
        z_increment = scale_factor
    else:
        z_offset = None
        z_increment = None

    # Convert heights to the target dtype
    heights = heights.astype(np_dtype)

    # Build XML structure
    ns = "http://www.opengps.eu/2008/ISO5436_2"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"

    root = ElementTree.Element(
        f"{{{ns}}}ISO5436_2",
        attrib={
            f"{{{xsi}}}schemaLocation": f"{ns} {ns}/ISO5436_2.xsd",
        }
    )
    root.set("xmlns:p", ns)
    root.set("xmlns:xsi", xsi)

    # Record1: Axes information
    record1 = ElementTree.SubElement(root, "Record1")
    ElementTree.SubElement(record1, "Revision").text = "ISO5436 - 2000"
    ElementTree.SubElement(record1, "FeatureType").text = "SUR"

    axes = ElementTree.SubElement(record1, "Axes")

    # CX axis
    cx = ElementTree.SubElement(axes, "CX")
    ElementTree.SubElement(cx, "AxisType").text = "I"
    ElementTree.SubElement(cx, "DataType").text = "L"
    ElementTree.SubElement(cx, "Increment").text = f"{dx:.15e}"
    ElementTree.SubElement(cx, "Offset").text = "0"

    # CY axis
    cy = ElementTree.SubElement(axes, "CY")
    ElementTree.SubElement(cy, "AxisType").text = "I"
    ElementTree.SubElement(cy, "DataType").text = "L"
    ElementTree.SubElement(cy, "Increment").text = f"{dy:.15e}"
    ElementTree.SubElement(cy, "Offset").text = "0"

    # CZ axis
    cz = ElementTree.SubElement(axes, "CZ")
    ElementTree.SubElement(cz, "AxisType").text = "A"
    ElementTree.SubElement(cz, "DataType").text = dtype
    if z_increment is not None:
        ElementTree.SubElement(cz, "Increment").text = f"{z_increment:.15e}"
    if z_offset is not None:
        ElementTree.SubElement(cz, "Offset").text = f"{z_offset:.15e}"

    # Record2: Metadata
    record2 = ElementTree.SubElement(root, "Record2")

    # Date - use acquisition_time from info if available, otherwise current time
    info = self.info
    if "acquisition_time" in info and info["acquisition_time"] is not None:
        date_str = info["acquisition_time"].isoformat()
    else:
        date_str = datetime.now().isoformat()
    ElementTree.SubElement(record2, "Date").text = date_str

    # Instrument information
    instrument = ElementTree.SubElement(record2, "Instrument")
    ElementTree.SubElement(instrument, "Manufacturer").text = manufacturer
    ElementTree.SubElement(instrument, "Model").text = model
    ElementTree.SubElement(instrument, "Serial").text = "not available"
    ElementTree.SubElement(instrument, "Version").text = version

    ElementTree.SubElement(record2, "CalibrationDate").text = date_str

    probing = ElementTree.SubElement(record2, "ProbingSystem")
    ElementTree.SubElement(probing, "Type").text = "Software"
    ElementTree.SubElement(probing, "Identification").text = "SurfaceTopography Python Library"

    ElementTree.SubElement(record2, "Comment").text = ""

    # Record3: Matrix dimension and data link
    record3 = ElementTree.SubElement(root, "Record3")

    matrix_dim = ElementTree.SubElement(record3, "MatrixDimension")
    ElementTree.SubElement(matrix_dim, "SizeX").text = str(nx)
    ElementTree.SubElement(matrix_dim, "SizeY").text = str(ny)
    ElementTree.SubElement(matrix_dim, "SizeZ").text = "1"

    # Prepare binary data (X3P stores data in column-major order, transposed)
    binary_data = heights.T.tobytes()
    md5_hash = hashlib.md5(binary_data).hexdigest().upper()

    data_link = ElementTree.SubElement(record3, "DataLink")
    ElementTree.SubElement(data_link, "PointDataLink").text = "bindata/data.bin"
    ElementTree.SubElement(data_link, "MD5ChecksumPointData").text = md5_hash

    # Validity data: one bit per point, least-significant bit first, in the
    # same point order as the height data
    valid_data = None
    if has_invalid:
        valid_data = np.packbits(
            ~invalid_mask.T.reshape(-1), bitorder="little"
        ).tobytes()
        ElementTree.SubElement(data_link, "ValidPointsLink").text = "bindata/valid.bin"
        ElementTree.SubElement(data_link, "MD5ChecksumValidPoints").text = \
            hashlib.md5(valid_data).hexdigest().upper()

    # Record4: Checksum file reference (optional, but included for completeness)
    record4 = ElementTree.SubElement(root, "Record4")
    ElementTree.SubElement(record4, "ChecksumFile").text = "md5checksum.hex"

    # Generate XML string
    xml_declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'

    # Manual XML serialization to match X3P format expectations
    def serialize_element(elem, indent=0):
        """Serialize element with proper formatting."""
        tag = elem.tag
        # Remove namespace prefix if present
        if tag.startswith("{"):
            tag = "p:" + tag.split("}")[1]

        result = "  " * indent + f"<{tag}"
        for key, value in elem.attrib.items():
            if key.startswith("{"):
                # Handle namespace attributes
                ns_url, attr_name = key[1:].split("}")
                if "XMLSchema-instance" in ns_url:
                    result += f' xsi:{attr_name}="{value}"'
                else:
                    result += f' {attr_name}="{value}"'
            else:
                result += f' {key}="{value}"'

        if len(elem) == 0 and elem.text is None:
            result += "/>\n"
        elif len(elem) == 0:
            result += f">{elem.text}</{tag}>\n"
        else:
            result += ">\n"
            for child in elem:
                result += serialize_element(child, indent + 1)
            result += "  " * indent + f"</{tag}>\n"

        return result

    xml_string = xml_declaration + serialize_element(root)

    # Create checksum file content
    checksum_content = f"{md5_hash} *bindata/data.bin\n"

    # Write ZIP file (`ZipFile` accepts both file names and file-like objects)
    with ZipFile(fobj, "w") as x3p:
        x3p.writestr("main.xml", xml_string.encode("utf-8"))
        x3p.writestr("bindata/data.bin", binary_data)
        if valid_data is not None:
            x3p.writestr("bindata/valid.bin", valid_data)
        x3p.writestr("md5checksum.hex", checksum_content.encode("utf-8"))


UniformTopographyInterface.register_function("to_x3p", write_x3p)
