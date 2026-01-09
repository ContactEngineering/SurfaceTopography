#
# Copyright 2022-2023 Lars Pastewka
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
from zipfile import BadZipFile, ZipFile

import dateutil.parser
import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import get_unit_conversion_factor
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase


class X3PReader(ReaderBase):
    _format = "x3p"
    _mime_types = ["application/x-iso5436-2-spm"]
    _file_extensions = ["x3p"]

    _name = "XML 3D surface profile (X3P)"
    _description = """
X3P is a container format conforming to the ISO 5436-2 (Geometrical Product
Specifications — Surface texture) standard. The format is defined in ISO
25178 and is a standardized format for the exchange of surface topography
data. The full specification of the format can be found
[here](http://www.opengps.eu/).
"""

    _REVISION = "ISO5436 - 2000"
    _FEATURE_TYPE = "SUR"

    # Data types of binary container
    # See: https://sourceforge.net/p/open-gps/mwiki/X3p/
    _DTYPE_MAP = {
        "I": np.dtype("<u2"),
        "L": np.dtype("<u4"),
        "F": np.dtype("f4"),
        "D": np.dtype("f8"),
    }

    # ZIP magic bytes (PK\x03\x04)
    _MAGIC_ZIP = b'PK\x03\x04'

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < len(cls._MAGIC_ZIP):
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC_ZIP):
            # ZIP file, could be X3P - need full parsing to confirm
            return MagicMatch.MAYBE
        return MagicMatch.NO

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, "rb") as f:
            try:
                with ZipFile(f, "r") as x3p:
                    try:
                        main_xml = x3p.open("main.xml")
                    except OSError:
                        # This appears not to be an X3P
                        raise FileFormatMismatch("ZIP file does not have 'main.xml'.")

                    xmlroot = ElementTree.parse(main_xml).getroot()
                    # Information on the measurement (grid points, size, etc.)
                    record1 = xmlroot.find("Record1")
                    # Information on the data file
                    record3 = xmlroot.find("Record3")

                    if record1 is None:
                        raise FileFormatMismatch("'Record1' not found in 'main.xml'.")
                    if record3 is None:
                        raise FileFormatMismatch("'Record3' not found in 'main.xml'.")

                    # Parse record1
                    revision = record1.find("Revision").text
                    if revision != self._REVISION:
                        raise CorruptFile(
                            f"Revision should be '{self._REVISION}' but is '{revision}."
                        )
                    feature_type = record1.find("FeatureType").text
                    if feature_type != self._FEATURE_TYPE:
                        raise CorruptFile(
                            f"FeatureType should be '{self._FEATURE_TYPE}' but is {feature_type}."
                        )
                    axes = record1.find("Axes")
                    cx = axes.find("CX")
                    cy = axes.find("CY")
                    cz = axes.find("CZ")

                    if cx.find("AxisType").text != "I":
                        raise CorruptFile(
                            "CX AxisType is not 'I'. Don't know how to handle this."
                        )
                    if cy.find("AxisType").text != "I":
                        raise CorruptFile(
                            "CY AxisType is not 'I'. Don't know how to handle this."
                        )
                    if cz.find("AxisType").text != "A":
                        raise CorruptFile(
                            "CZ AxisType is not 'A'. Don't know how to handle this."
                        )

                    grid_spacing_x = float(cx.find("Increment").text)
                    grid_spacing_y = float(cy.find("Increment").text)

                    datatype = cz.find("DataType").text
                    self._dtype = self._DTYPE_MAP[datatype]

                    increment = cz.find("Increment")
                    if increment is not None:
                        # We have no proper test for this, as in all files that
                        # we have, the z-increment is either missing or unity.
                        self._height_scale_factor = float(increment.text)
                    else:
                        self._height_scale_factor = None

                    # Parse record3
                    matrix_dimension = record3.find("MatrixDimension")
                    nb_grid_pts_x = int(matrix_dimension.find("SizeX").text)
                    nb_grid_pts_y = int(matrix_dimension.find("SizeY").text)
                    nb_grid_pts_z = int(matrix_dimension.find("SizeZ").text)

                    if nb_grid_pts_z != 1:
                        raise CorruptFile(
                            "Z dimension has extend != 1. Volumetric data is not supported."
                        )

                    self._nb_grid_pts = (nb_grid_pts_x, nb_grid_pts_y)
                    self._physical_sizes = (
                        grid_spacing_x * nb_grid_pts_x,
                        grid_spacing_y * nb_grid_pts_y,
                    )

                    data_link = record3.find("DataLink")
                    self._name_of_binary_file = data_link.find("PointDataLink").text

                    # Check if binary file exists and has a reasonable size
                    binary_info = x3p.getinfo(self._name_of_binary_file)
                    expected_file_size = (
                        np.prod(self._nb_grid_pts) * self._dtype.itemsize
                    )
                    if binary_info.file_size < expected_file_size:
                        raise CorruptFile(
                            f"Binary file is too small. It has size if {binary_info.file_size} bytes, "
                            f"but I expected a size of {expected_file_size} bytes."
                        )

                    # Unit is always meters
                    self._unit = "m"
                    self._info = {}

                    # Metadata; if this record is missing, we just don't extract metadata
                    record2 = xmlroot.find("Record2")
                    if record2 is not None:
                        raw_metadata = {}

                        date = record2.find("Date")
                        if date is not None:
                            raw_metadata["Date"] = date.text
                            self._info["acquisition_time"] = dateutil.parser.parse(
                                date.text
                            )

                        calibration_date = record2.find("CalibrationDate")
                        if calibration_date is not None:
                            raw_metadata["CalibrationDate"] = calibration_date.text

                        instrument = record2.find("Instrument")
                        if instrument is not None:
                            instrument_metadata = {
                                child.tag: child.text for child in instrument
                            }
                            raw_metadata["Instrument"] = instrument_metadata
                            model = manufacturer = version = None
                            if "Model" in instrument_metadata:
                                model = instrument_metadata["Model"]
                            if "Manufacturer" in instrument_metadata:
                                manufacturer = instrument_metadata["Manufacturer"]
                            if "Version" in instrument_metadata:
                                version = instrument_metadata["Version"]

                            instrument_info = {}
                            if model == "unknown":
                                if manufacturer != "unknown":
                                    if version != "unknown":
                                        instrument_info["name"] = (
                                            f"{manufacturer} (version {version})"
                                        )
                                    else:
                                        instrument_info["name"] = manufacturer
                            else:
                                if manufacturer != "unknown":
                                    if version != "unknown":
                                        instrument_info["name"] = (
                                            f"{model} ({manufacturer}, version {version})"
                                        )
                                    else:
                                        instrument_info["name"] = (
                                            f"{model} ({manufacturer})"
                                        )
                                else:
                                    if version != "unknown":
                                        instrument_info["name"] = (
                                            f"{model} (version {version})"
                                        )
                                    else:
                                        instrument_info["name"] = model
                            if instrument_info != {}:
                                self._info["instrument"] = instrument_info

            except BadZipFile:
                # This is not an X3P
                raise FileFormatMismatch("This is not a ZIP file.")

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                0,  # channel index
                name="Default",
                dim=2,
                nb_grid_pts=self._nb_grid_pts,
                physical_sizes=self._physical_sizes,
                height_scale_factor=self._height_scale_factor,
                uniform=True,
                periodic=False,
                unit=self._unit,
                info=self._info,
            )
        ]

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=None,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(
                f"There is only a single channel. Channel index must be {self._default_channel_index}."
            )

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None and self._height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        _info = self._info.copy()
        _info.update(info)

        with OpenFromAny(self.file_path, "rb") as f:
            with ZipFile(f, "r") as x3p:
                nx, ny = self._nb_grid_pts
                rawdata = x3p.open(self._name_of_binary_file).read(
                    np.prod(self._nb_grid_pts) * self._dtype.itemsize
                )
                height_data = (
                    np.frombuffer(
                        rawdata, count=np.prod(self._nb_grid_pts), dtype=self._dtype
                    )
                    .reshape(ny, nx)
                    .T
                )

        topo = Topography(
            height_data,
            self._physical_sizes,
            unit=self._unit,
            info=_info,
            periodic=False if periodic is None else periodic,
        )
        if self._height_scale_factor is not None:
            return topo.scale(self._height_scale_factor)
        elif height_scale_factor is not None:
            return topo.scale(height_scale_factor)
        else:
            return topo


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

    # Get height data and convert to meters
    heights = self.heights()
    if np.ma.isMaskedArray(heights):
        # X3P doesn't directly support masked data - use NaN for undefined
        heights = np.ma.filled(heights, np.nan)
    heights = heights * scale_to_meters

    # For integer types, we need to scale the data
    if dtype in ("I", "L"):
        height_min = np.nanmin(heights)
        height_max = np.nanmax(heights)
        height_range = height_max - height_min
        if height_range == 0:
            height_range = 1.0
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

    # Write ZIP file
    if isinstance(fobj, str):
        with ZipFile(fobj, "w") as x3p:
            x3p.writestr("main.xml", xml_string.encode("utf-8"))
            x3p.writestr("bindata/data.bin", binary_data)
            x3p.writestr("md5checksum.hex", checksum_content.encode("utf-8"))
    else:
        # File-like object
        with ZipFile(fobj, "w") as x3p:
            x3p.writestr("main.xml", xml_string.encode("utf-8"))
            x3p.writestr("bindata/data.bin", binary_data)
            x3p.writestr("md5checksum.hex", checksum_content.encode("utf-8"))


UniformTopographyInterface.register_function("to_x3p", write_x3p)
