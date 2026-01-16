#
# Copyright 2023-2025 Lars Pastewka
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

The MNT format is a Microsoft Compound Document File (OLE) containing:
- ImagePreview: JPEG preview image
- ScopedContents: Binary data with TLV-encoded metadata and zlib-compressed
  height data
- ScopedResults: Parameter data
- XmlHeader: UTF-16 encoded XML metadata

The ScopedContents stream uses a hierarchical TLV (Tag-Length-Value) structure:
- First 8 bytes: uint64 LE size field (= stream size - 8)
- Remaining bytes: TLV entries with tag (uint16 LE) + size (uint64 LE) + data

Height data container (tag 0x02BD) structure:
- compressed_blocks contains multiple sections separated by uint64 size fields:
  - Section 1: Compressed height data blocks (zlib-compressed)
  - Section 2: Additional metadata
  - Section 3: Image parameters (nested ~6 levels deep in tag 0x0002 containers)

Image parameters (in Section 3, innermost tag 0x0002 container):
- 0x0007: Width in pixels (uint32)
- 0x0008: Height in pixels (uint32)
- 0x0009: Physical size X in mm (double)
- 0x000a: Physical size Y in mm (double)

Height data format:
- Stored as int32 values
- Data type tag (tag 0x0001 inside height_data container) indicates masking:
  - Tag 12: zeros indicate undefined/masked pixels (common at image corners)
  - Tag 39: pure int32 data, all values including zeros are valid heights
- Height scale factor stored in pixel_scales container (typically 10 nm/count)

Note: 0xFFFF tags are used as section markers/delimiters and must be skipped
when parsing nested containers.
"""

import struct
import zlib
from io import BytesIO

import numpy as np
import olefile

from ..Exceptions import CorruptFile, FileFormatMismatch
from ..Support.UnitConversion import get_unit_conversion_factor, mangle_length_unit_utf8
from ..UniformLineScanAndTopography import Topography
from .binary import BinaryStructure, RawBuffer, TextBuffer, TLVContainer
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase, Skip


class MNTReader(ReaderBase):
    _format = "mnt"
    _mime_types = ["application/x-digitalsurf-mnt"]
    _file_extensions = ["mnt"]

    _name = "Digital Surf Mountains"
    _description = """
File format of the Digital Surf Mountains software. This format is a
Microsoft Compound Document (OLE) file containing TLV-encoded metadata
and compressed height data.
"""

    # Block structures and TLV parser are initialized lazily to avoid circular imports
    _block_structures = None
    _tlv_parser = None
    _section3_parser = None

    # TLV format constants
    _TLV_TAG_FORMAT = "<H"
    _TLV_SIZE_FORMAT = "<Q"

    # TLV tags for image parameters (found in Section 3 of compressed_blocks)
    _TAG_PHYSICAL_SIZE_X = 0x0009
    _TAG_PHYSICAL_SIZE_Y = 0x000A

    # Parser for sections inside compressed_blocks - initialized lazily
    _compressed_blocks_sections_parser = None

    @classmethod
    def _find_image_params_container(cls, section_data):
        """
        Navigate through nested TLV containers to find the image parameters.

        Uses the declarative _section3_parser to parse the nested structure:
        Section 3 root -> 0x0002 -> level2 -> 0x0001 -> level3 -> 0x0002 ->
        level4 -> 0x0002 -> level5 -> 0x0002 -> level6 (contains dimension info)

        Parameters
        ----------
        section_data:bytes
            Section 3 data.

        Returns
        -------
        params:dict or None
            Dictionary with parsed level6 data, or None if not found.
        """
        # Initialize parser if needed
        if cls._section3_parser is None:
            cls._init_block_structures()

        try:
            # Parse section 3 using declarative structure
            stream = BytesIO(section_data)
            parsed = cls._section3_parser.from_stream(stream, {})

            # Navigate to level6 which contains the dimension info
            # Structure: section3 -> 0x0002 -> level2 -> 0x0001 -> level3 ->
            #            0x0002 -> level4 -> 0x0002 -> level5 -> 0x0002 -> level6
            section3 = parsed.get("section3", parsed)
            level1 = section3.get(0x0002) or section3.get("level2")
            if level1 is None:
                return None

            level2 = level1.get("level2", level1)
            level2_inner = level2.get(0x0001) or level2.get("level3")
            if level2_inner is None:
                return None

            level3 = level2_inner.get("level3", level2_inner)
            level3_inner = level3.get(0x0002) or level3.get("level4")
            if level3_inner is None:
                return None

            level4 = level3_inner.get("level4", level3_inner)
            level4_inner = level4.get(0x0002) or level4.get("level5")
            if level4_inner is None:
                return None

            level5 = level4_inner.get("level5", level4_inner)
            level5_inner = level5.get(0x0002) or level5.get("level6")
            if level5_inner is None:
                return None

            level6 = level5_inner.get("level6", level5_inner)
            return level6

        except Exception:
            return None

    @classmethod
    def _extract_image_params(cls, data):
        """
        Extract image parameters from compressed_blocks data.

        Navigates the hierarchical TLV structure to find:
        - Width (tag 0x0007, uint32)
        - Height (tag 0x0008, uint32)
        - Physical size X (tag 0x0009, double, in mm)
        - Physical size Y (tag 0x000a, double, in mm)

        Parameters
        ----------
        data:bytes
            Raw compressed_blocks data.

        Returns
        -------
        params:dict
            Dictionary with keys: 'width', 'height', 'physical_size_x',
            'physical_size_y'. Values are None if not found.
        """
        result = {
            "width": None,
            "height": None,
            "physical_size_x": None,
            "physical_size_y": None,
        }

        # Initialize parsers if needed
        if cls._compressed_blocks_sections_parser is None:
            cls._init_block_structures()

        # compressed_blocks structure:
        # [outer_tag 0x0001][outer_size] = 10 bytes header
        # [prefix][tag][size][data]...   = children with 8-byte prefixes
        if len(data) < 28:
            return result

        # Skip outer container header (10 bytes), parse children with prefixes
        stream = BytesIO(data[10:])

        # Parse sections using declarative parser with entry_prefix_format
        parsed = cls._compressed_blocks_sections_parser.from_stream(stream, {})
        sections_data = parsed.get("sections", parsed)

        # All sections have tag 0x0001, so they're stored as a list
        sections = sections_data.get(0x0001, [])
        if not isinstance(sections, list):
            sections = [sections]

        # Section 3 contains image parameters (index 2, 0-based)
        if len(sections) < 3:
            return result

        # Get the raw data of section 2
        section3_entry = sections[2]
        section3_raw = section3_entry.get("_raw") if isinstance(section3_entry, dict) else None
        if section3_raw is None:
            return result

        # Find the innermost container with image parameters
        params = cls._find_image_params_container(section3_raw)
        if params is None:
            return result

        # Extract values using declarative names or tag IDs
        result["width"] = params["width"]["value"]
        result["height"] = params["height"]["value"]
        result["physical_size_x"] = params["physical_size_x"]["value"]
        result["physical_size_y"] = params["physical_size_y"]["value"]

        return result

    @classmethod
    def _extract_metadata(cls, main_container):
        """
        Extract human-readable metadata from the parsed main container.

        Parameters
        ----------
        main_container : dict
            Parsed main container from TLV structure.

        Returns
        -------
        metadata : dict
            Dictionary containing extracted metadata fields.
        """
        metadata = {}

        # Extract pixel scales
        pixel_scales = main_container.get("pixel_scales", {})
        if pixel_scales:
            scales = {}
            for key, name in [
                ("scale_x", "x"),
                ("scale_y", "y"),
                ("scale_z", "z"),
                ("scale_4", "unknown"),
            ]:
                val = pixel_scales.get(key) or pixel_scales.get(
                    {"scale_x": 1, "scale_y": 2, "scale_z": 3, "scale_4": 4}.get(key)
                )
                if isinstance(val, dict) and "value" in val:
                    scales[name] = val["value"]
            if scales:
                metadata["pixel_scales_nm"] = scales

        # Extract block params
        block_params = main_container.get("block_params", {})
        if block_params:
            params = {}
            for key in ["factor_a", "factor_b", "param_3", "param_4", "param_5"]:
                val = block_params.get(key)
                if isinstance(val, dict) and "value" in val:
                    params[key] = val["value"]
            if params:
                metadata["block_params"] = params

        # Extract dimension params
        dim_params = main_container.get("dimension_params", {})
        if dim_params:
            dims = {}
            for key in ["nx", "ny"]:
                val = dim_params.get(key)
                if isinstance(val, dict) and "value" in val:
                    dims[key] = val["value"]
            if dims:
                metadata["dimension_params"] = dims

        # Extract serial number
        serial = main_container.get("serial_number")
        if serial and isinstance(serial, str):
            metadata["serial_number"] = serial.strip("\x00")

        # Extract measurement params
        meas_params = main_container.get("measurement_params", {})
        if meas_params:
            params = {}
            for key in ["instrument_name", "objective"]:
                val = meas_params.get(key)
                if val and isinstance(val, str):
                    params[key] = val.strip("\x00")
            if params:
                metadata["measurement_params"] = params

        # Extract extended metadata
        ext_meta = main_container.get("extended_metadata", {})
        if ext_meta:
            ext = {}
            desc = ext_meta.get("extended_description")
            if desc and isinstance(desc, str):
                ext["description"] = desc.strip("\x00")
            if ext:
                metadata["extended_metadata"] = ext

        return metadata

    @classmethod
    def _init_block_structures(cls):
        """Initialize TLV block structure definitions.

        MNT File Structure Overview
        ===========================
        The ScopedContents stream contains a hierarchical TLV structure:

        [8 bytes: uint64 stream size]
        [TLV entries...]

        Top-level tags:
        - 0x0001: Main data container (contains all measurement data)
        - 0x0002: Format flags (uint32)
        - 0x0003: Format version (uint32)

        Main container (0x0001) children:
        - 0x00c8: File metadata container (timestamps, software info)
        - 0x00c9: Unknown (small, possibly flags)
        - 0x00ca: Measurement parameters container
        - 0x00cb: Serial number (UTF-16 encoded string)
        - 0x012d: Extended metadata container
        - 0x0003: Dimension parameters (nb_blocks, rows_per_block factors)
        - 0x0006: Pixel scale factors (physical sizes, units)
        - 0x02bd: Height data container (zlib-compressed blocks)
        - 0x0258: Color palette/visualization settings
        - 0xffff: Section delimiter/marker

        Compressed Data Block Prefix (16 bytes before zlib stream):
        - Bytes 0-7:  uint64 LE - Element offset (for block ordering)
        - Bytes 8-11: uint32 LE - Elements per block (nx * rows_per_block)
        - Bytes 12-15: uint32 LE - Compressed size

        Height Data Format:
        - Stored as int32 values
        - Data type tag indicates masking behavior:
          - Tag 12: zeros indicate undefined/masked pixels
          - Tag 39: pure int32, all values including zeros are valid
        """

        # Helper to create BinaryStructure with single uint32 field
        def uint32(name):
            return BinaryStructure([("value", "I")], name=name)

        # =====================================================================
        # Nested container definitions (deepest first)
        #
        # Legend:
        #   [CONFIRMED] - Verified through hex analysis of multiple files
        #   [LIKELY]    - Strong evidence but not fully confirmed
        #   [GUESS]     - Speculative based on tag position or similar formats
        #   [UNKNOWN]   - Purpose unknown, included for completeness
        # =====================================================================

        SIZE_FMT = "<Q"  # MNT uses uint64 LE size fields

        # Container 0x00c8 children - [CONFIRMED] block dimension parameters
        # Evidence: Tags 0x0001 and 0x0002 contain factor_a and factor_b
        # which multiply to give rows_per_block
        block_params_children = {
            0x0001: uint32("factor_a"),  # [CONFIRMED] uint32
            0x0002: uint32("factor_b"),  # [CONFIRMED] uint32
            0x0003: uint32("param_3"),  # [UNKNOWN] uint32
            0x0004: uint32("param_4"),  # [UNKNOWN] uint32
            0x0005: uint32("param_5"),  # [UNKNOWN] uint32
            0x0006: uint32("param_6"),  # [UNKNOWN] uint32
            0x0007: uint32("param_7"),  # [UNKNOWN] uint32
            0x0008: Skip(comment="variable size"),
            0x0009: Skip(comment="variable size"),
            0x000A: Skip(comment="variable size"),
        }

        # Container 0x0003 children (inside main) - dimension parameters
        dimension_params_children = {
            0x0001: uint32("nx"),  # [LIKELY] X dimension as uint32
            0x0002: uint32("ny"),  # [LIKELY] Y dimension as uint32
        }

        # Container 0x0006 children - [CONFIRMED] pixel scale factors
        # Evidence: Tags 0x0001-0x0004 contain double values (all 10.0 in test file)
        # These are likely scale factors in nm/count for X, Y, Z axes
        pixel_scale_children = {
            0x0001: BinaryStructure([("value", "d")], name="scale_x"),  # [CONFIRMED] X scale (nm/count)
            0x0002: BinaryStructure([("value", "d")], name="scale_y"),  # [CONFIRMED] Y scale (nm/count)
            0x0003: BinaryStructure([("value", "d")], name="scale_z"),  # [CONFIRMED] Z scale (nm/count)
            0x0004: BinaryStructure([("value", "d")], name="scale_4"),  # [CONFIRMED] Unknown scale
            0x0005: TextBuffer("x_unit"),  # [LIKELY] Contains ASCII text (unit?)
            0x0006: TextBuffer("y_unit"),  # [LIKELY] Contains ASCII text (unit?)
            0x0007: TextBuffer("z_unit"),  # [LIKELY] Contains ASCII text (unit?)
        }

        # Container 0x00ca children - [GUESS] possibly measurement parameters
        measurement_params_children = {
            0x0001: Skip(comment="1 byte, measurement type"),
            0x0002: Skip(comment="scan params"),
            0x0003: Skip(
                comment="container with unknown structure, instrument settings"
            ),
            0x0004: TextBuffer("instrument_name"),  # [LIKELY] Contains ASCII text
            0x0005: TextBuffer("objective"),  # [LIKELY] Contains ASCII text
        }

        # =====================================================================
        # Nested structure for image parameters in Section 3 of compressed_blocks
        #
        # Section 3 structure (from outermost to innermost):
        # Section 3 root (tags: 0x0001, 0x0002, 0x0003)
        # └── 0x0002 (level 1)
        #     ├── [0xFFFF markers]
        #     └── 0x0001 (level 2, tags: 0x0001, 0x0002, 0x0003)
        #         └── 0x0002 (level 3)
        #             ├── [0xFFFF markers]
        #             └── 0x0002 (level 4, tags: 0x0001, 0x0002, 0x0003)
        #                 └── 0x0002 (level 5)
        #                     ├── [0xFFFF markers]
        #                     └── 0x0002 (level 6 - innermost)
        #                         ├── 0x0007 (width, uint32)
        #                         ├── 0x0008 (height, uint32)
        #                         ├── 0x0009 (physical_size_x, double in mm)
        #                         └── 0x000a (physical_size_y, double in mm)
        # =====================================================================

        # Level 6 (innermost): Contains actual dimension info
        level6_children = {
            0xFFFF: Skip(comment="section delimiter/marker"),
            0x0007: uint32("width"),  # [CONFIRMED] Image width in pixels
            0x0008: uint32("height"),  # [CONFIRMED] Image height in pixels
            0x0009: BinaryStructure([("value", "d")], name="physical_size_x"),
            0x000A: BinaryStructure([("value", "d")], name="physical_size_y"),
        }

        # Level 5: Container wrapping level 6
        level5_children = {
            0xFFFF: Skip(comment="section delimiter/marker"),
            0x0002: TLVContainer(level6_children, name="level6", size_format=SIZE_FMT),
        }

        # Level 4: Container wrapping level 5
        level4_children = {
            0xFFFF: Skip(comment="section delimiter/marker"),
            0x0002: TLVContainer(level5_children, name="level5", size_format=SIZE_FMT),
        }

        # Level 3: Container wrapping level 4
        level3_children = {
            0x0002: TLVContainer(level4_children, name="level4", size_format=SIZE_FMT),
        }

        # Level 2: Container wrapping level 3
        level2_children = {
            0xFFFF: Skip(comment="section delimiter/marker"),
            0x0001: TLVContainer(level3_children, name="level3", size_format=SIZE_FMT),
        }

        # Level 1 (Section 3 root): Container wrapping level 2
        section3_children = {
            0x0002: TLVContainer(level2_children, name="level2", size_format=SIZE_FMT),
        }

        # Store the section 3 parser for use in dimension extraction
        cls._section3_parser = TLVContainer(
            section3_children, name="section3", size_format=SIZE_FMT
        )

        # Parser for sections inside compressed_blocks
        # Structure: outer container header (10 bytes) + children with 8-byte prefixes
        # Each section has tag 0x0001, so they'll be stored as a list
        # We store sections as raw data since we only need section 2 for dimensions
        compressed_blocks_sections_children = {
            0x0001: RawBuffer("section", size=None, lazy=False),
        }
        cls._compressed_blocks_sections_parser = TLVContainer(
            compressed_blocks_sections_children,
            name="sections",
            size_format=SIZE_FMT,
            entry_prefix_format=SIZE_FMT,  # 8-byte look-ahead size before each entry
        )

        # Container 0x02bd children - [CONFIRMED] height data
        # Evidence: Contains TLV metadata followed by zlib-compressed blocks
        # Note: compressed_blocks is stored as raw because it contains both
        # TLV metadata (with dimensions) and binary zlib-compressed data
        height_data_children = {
            # [CONFIRMED] Data type tag: 12 = zeros are undefined, 39 = pure int32
            0x0001: BinaryStructure([("value", "B")], name="data_type"),
            0x0002: RawBuffer("compressed_blocks", size=None, lazy=False),
        }

        # Container 0x012d children - [UNKNOWN]
        # Unit info container (0x000A inside extended_metadata)
        # Contains Z display unit
        unit_info_children = {
            0x0001: Skip(comment="unknown uint32"),
            0x0002: Skip(comment="unknown double"),
            0x0003: Skip(comment="unknown double"),
            0x0004: RawBuffer("z_unit_raw", size=None, lazy=False),  # Z unit as length-prefixed UTF-16
            0x0005: Skip(comment="unknown uint32"),
        }

        # Axis info container (0x0009 inside extended_metadata)
        # Contains X, Y, Z storage units and scale factors
        axis_info_children = {
            0x0010: BinaryStructure([("value", "d")], name="scale_x"),  # X scale factor
            0x0011: BinaryStructure([("value", "d")], name="scale_y"),  # Y scale factor
            0x0012: BinaryStructure([("value", "d")], name="scale_z"),  # Z scale factor
            0x0013: RawBuffer("x_unit_raw", size=None, lazy=False),  # X unit (length-prefixed UTF-16)
            0x0014: RawBuffer("y_unit_raw", size=None, lazy=False),  # Y unit (length-prefixed UTF-16)
            0x0015: RawBuffer("z_unit_raw", size=None, lazy=False),  # Z storage unit (length-prefixed UTF-16)
        }

        extended_metadata_children = {
            0x0001: TextBuffer("extended_description"),  # [LIKELY] Contains ASCII text
            0x0002: Skip(comment="container with unknown structure, extended params"),
            0x0003: Skip(comment="extended flags"),
            0x0009: TLVContainer(
                axis_info_children, name="axis_info", size_format=SIZE_FMT
            ),  # [CONFIRMED] Contains X, Y, Z storage units
            0x000A: TLVContainer(
                unit_info_children, name="unit_info", size_format=SIZE_FMT
            ),  # [CONFIRMED] Contains Z display unit
        }

        # Container 0x0258 children - [GUESS] possibly visualization/palette
        # Evidence: Only present in some files, similar tag range to other formats
        color_palette_children = {
            0x0001: Skip(comment="palette info"),
            0x0002: Skip(comment="palette data"),
        }

        # =====================================================================
        # Main container children (tag 0x0001)
        # [CONFIRMED] Tag 0x0001 is the main container
        # =====================================================================
        main_container_children = {
            # [CONFIRMED] These tags exist and are containers or leaf nodes as marked
            0x00C8: TLVContainer(
                block_params_children, name="block_params", size_format=SIZE_FMT
            ),
            0x00C9: Skip(comment="file flags"),
            0x00CB: TextBuffer(
                "serial_number"
            ),  # [LIKELY] Contains text (serial number?)
            0x00CA: TLVContainer(
                measurement_params_children,
                name="measurement_params",
                size_format=SIZE_FMT,
            ),
            0x012D: TLVContainer(
                extended_metadata_children,
                name="extended_metadata",
                size_format=SIZE_FMT,
            ),
            0x0190: Skip(comment="acquisition mode"),
            # Grid structure - [CONFIRMED] tag 0x0003 contains dimension info
            0x0001: Skip(comment="data type"),
            0x0002: Skip(comment="format flags"),
            0x0003: TLVContainer(
                dimension_params_children, name="dimension_params", size_format=SIZE_FMT
            ),
            0x0004: Skip(comment="grid type"),
            0x0005: Skip(comment="data encoding"),
            # [GUESS] Tags 0x0006-0x000d might be physical dimensions
            0x0006: TLVContainer(
                pixel_scale_children, name="pixel_scales", size_format=SIZE_FMT
            ),
            0x0007: Skip(comment="origin X"),
            0x0008: Skip(comment="origin Y"),
            0x0009: Skip(comment="origin Z"),
            0x000A: Skip(comment="container with unknown structure, coordinate system"),
            0x000B: Skip(comment="aspect ratio"),
            0x000C: Skip(comment="rotation"),
            0x000D: Skip(comment="tilt"),
            # [GUESS] Tags 0x0064-0x0066 grouped together, maybe statistics
            0x0064: Skip(comment="height stats"),
            0x0065: Skip(comment="RMS stats"),
            0x0066: Skip(comment="higher moments"),
            0x0258: TLVContainer(
                color_palette_children, name="color_palette", size_format=SIZE_FMT
            ),
            # [CONFIRMED] Height data container
            0x02BD: TLVContainer(
                height_data_children, name="height_data", size_format=SIZE_FMT
            ),
            # [GUESS] Tags 0x0014-0x001b might be processing/ROI related
            0x0014: Skip(comment="container with unknown structure, filter history"),
            0x0015: Skip(comment="container with unknown structure, leveling history"),
            0x0016: Skip(comment="container with unknown structure, form removal"),
            0x0017: Skip(comment="processing flags"),
            0x0018: Skip(comment="quality score"),
            0x0019: Skip(comment="container with unknown structure, ROI definitions"),
            0x001A: Skip(comment="container with unknown structure, mask regions"),
            0x001B: Skip(comment="container with unknown structure, annotations"),
            0xFFFF: Skip(comment="section delimiter/marker"),
        }

        # =====================================================================
        # Top-level block structures (after 8-byte size header)
        # [CONFIRMED] TLV structure: uint16 tag + uint64 size + data
        # =====================================================================
        cls._block_structures = {
            0x0001: TLVContainer(
                main_container_children, name="main", size_format=SIZE_FMT
            ),
            0x0002: Skip(comment="format flags"),
            0x0003: Skip(comment="format version"),
            0x0004: Skip(comment="file type"),
            0x0005: BinaryStructure(
                [("value", "Q")], name="num_blocks"
            ),  # [CONFIRMED] uint64
        }

        # Create top-level TLV parser
        cls._tlv_parser = TLVContainer(cls._block_structures, size_format=SIZE_FMT)

    def __init__(self, fobj):
        """
        Load Digital Surf Mountains data files.

        Arguments
        ---------
        fobj:filename or file object
            File or data stream to open.
        """
        # Initialize block structures if not already done
        if self._block_structures is None:
            self._init_block_structures()

        self._fobj = fobj
        self._channels = []

        with OpenFromAny(fobj, "rb") as f:
            # Read enough bytes for OLE detection and initial parsing
            header = f.read(8)
            f.seek(0)

            # Check for OLE signature
            if header[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                raise FileFormatMismatch("Not an OLE compound document")

            # Read entire file for OLE parsing
            file_data = f.read()

        # Parse as OLE file
        try:
            ole = olefile.OleFileIO(file_data)
        except Exception as e:
            raise FileFormatMismatch(f"Failed to parse OLE file: {e}")

        # Verify this is an MNT file by checking for expected streams
        if not ole.exists("ScopedContents"):
            ole.close()
            raise FileFormatMismatch("Missing ScopedContents stream")

        # Read ScopedContents
        scoped_contents = ole.openstream("ScopedContents").read()

        # Verify minimum header length for dimension extraction
        if len(scoped_contents) < 0x72:
            ole.close()
            raise CorruptFile("ScopedContents header too short")

        # Parse TLV structure for metadata storage
        # First 8 bytes are stream size (skip), then TLV entries start
        stream = BytesIO(scoped_contents[8:])
        self._metadata = self._tlv_parser.from_stream(stream, {})

        # Extract compressed height data from parsed metadata
        try:
            main = self._metadata["main"]
            height_data = main["height_data"]
            compressed_blocks_raw = height_data["compressed_blocks"]["_raw"]
        except (KeyError, TypeError):
            ole.close()
            raise CorruptFile("Missing compressed height data in metadata")

        # Extract data type tag from height_data container
        # Tag 0x0001 inside 0x02BD contains a type indicator:
        #   - 12: int32 data with zero as undefined marker
        #   - 39: pure int32 data (zeros are valid heights)
        self._data_type_tag = None
        try:
            data_type = height_data.get("data_type")
            if data_type is not None and isinstance(data_type, dict):
                self._data_type_tag = data_type.get("value")
        except (KeyError, TypeError):
            pass

        # Find all zlib-compressed blocks within the compressed data
        # Each block has a 16-byte prefix:
        #   Bytes 0-7:  uint64 LE - Element offset (for ordering)
        #   Bytes 8-11: uint32 LE - Elements per block
        #   Bytes 12-15: uint32 LE - Compressed size
        # Note: MNT files may have multiple sections of zlib blocks separated
        # by TLV structure, so we scan all blocks rather than chaining.
        zlib_blocks = []
        i = 0
        while i < len(compressed_blocks_raw) - 2:
            # Check for zlib header bytes
            if compressed_blocks_raw[i] == 0x78 and compressed_blocks_raw[i + 1] in [
                0x01,
                0x5E,
                0x9C,
                0xDA,
            ]:
                try:
                    decompressed = zlib.decompress(compressed_blocks_raw[i:])
                    if len(decompressed) >= 1000:  # Only substantial blocks
                        # Extract block prefix information
                        element_offset = 0
                        elements_per_block = 0
                        if i >= 16:
                            prefix = compressed_blocks_raw[i - 16:i]
                            element_offset = struct.unpack("<Q", prefix[0:8])[0]
                            elements_per_block = struct.unpack("<I", prefix[8:12])[0]
                        # Filter out false positives (invalid prefix values)
                        if elements_per_block < 1000000:
                            zlib_blocks.append(
                                {
                                    "pos": i,
                                    "size": len(decompressed),
                                    "data": decompressed,
                                    "element_offset": element_offset,
                                    "elements_per_block": elements_per_block,
                                }
                            )
                except zlib.error:
                    pass
            i += 1

        if not zlib_blocks:
            ole.close()
            raise CorruptFile("No zlib-compressed data found in height data")

        # Group blocks by decompressed size to identify height data blocks
        # (MNT files may contain multiple data layers with different block sizes)
        from collections import Counter

        size_counts = Counter(b["size"] for b in zlib_blocks)
        most_common_size = size_counts.most_common(1)[0][0]

        # Filter to only blocks with the most common size (height data)
        height_blocks = [b for b in zlib_blocks if b["size"] == most_common_size]

        # Sort blocks by element_offset to get correct ordering
        height_blocks.sort(key=lambda b: b["element_offset"])

        # Extract image parameters from TLV structure
        # This includes width, height, and physical sizes
        image_params = self._extract_image_params(compressed_blocks_raw)
        nx = image_params["width"]
        ny = image_params["height"]

        if nx is None or ny is None:
            ole.close()
            raise CorruptFile("Could not extract image dimensions from MNT file")

        # Store metadata for later use
        self._ole_data = file_data
        self._zlib_blocks = [(b["pos"], b["data"]) for b in height_blocks]
        self._nx = nx
        self._ny = ny

        # Extract physical sizes from image parameters
        physical_size_x = image_params["physical_size_x"]
        physical_size_y = image_params["physical_size_y"]

        # Helper to parse length-prefixed UTF-16 LE string
        def parse_unit_string(raw_buffer):
            if raw_buffer and isinstance(raw_buffer, dict):
                raw_data = raw_buffer.get("_raw")
                if raw_data and len(raw_data) >= 3:
                    byte_len = raw_data[0]
                    if len(raw_data) >= 1 + byte_len:
                        return raw_data[1:1 + byte_len].decode("utf-16-le")
            return None

        # Extract units from extended_metadata
        ext_meta = main.get("extended_metadata", {})

        # Extract X and Y storage units from axis_info container (0x0009)
        # Tags 0x0013 and 0x0014 contain X and Y units
        x_storage_unit = None
        y_storage_unit = None
        try:
            axis_info = ext_meta.get("axis_info") or ext_meta.get(0x0009)
            if axis_info and isinstance(axis_info, dict):
                x_unit_raw = axis_info.get("x_unit_raw") or axis_info.get(0x0013)
                y_unit_raw = axis_info.get("y_unit_raw") or axis_info.get(0x0014)
                x_storage_unit = parse_unit_string(x_unit_raw)
                y_storage_unit = parse_unit_string(y_unit_raw)
        except (KeyError, TypeError, UnicodeDecodeError):
            pass

        # Extract Z display unit from unit_info container (0x000A)
        # Tag 0x0004 contains the Z unit for display
        z_unit = None
        try:
            unit_info = ext_meta.get("unit_info") or ext_meta.get(0x000A)
            if unit_info and isinstance(unit_info, dict):
                z_unit_raw = unit_info.get("z_unit_raw") or unit_info.get(0x0004)
                z_unit = parse_unit_string(z_unit_raw)
        except (KeyError, TypeError, UnicodeDecodeError):
            pass

        # Normalize unit strings (e.g., ensure µ is MICRO SIGN not GREEK MU)
        x_storage_unit = mangle_length_unit_utf8(x_storage_unit) if x_storage_unit else "mm"
        y_storage_unit = mangle_length_unit_utf8(y_storage_unit) if y_storage_unit else "mm"
        unit = mangle_length_unit_utf8(z_unit) if z_unit else "µm"

        # Convert physical sizes from storage units to the target Z unit
        if physical_size_x is not None and physical_size_y is not None:
            x_to_unit = get_unit_conversion_factor(x_storage_unit, unit)
            y_to_unit = get_unit_conversion_factor(y_storage_unit, unit)
            physical_size_x = physical_size_x * x_to_unit
            physical_size_y = physical_size_y * y_to_unit
        else:
            # Fallback: use pixel count as physical size
            physical_size_x = float(nx)
            physical_size_y = float(ny)

        # Extract height scale factor from pixel_scales container
        # The scale values are in nm/count
        height_scale_factor_nm = None
        try:
            pixel_scales = main.get("pixel_scales", {})
            # Tag 0x0003 contains Z scale, or try scale_z by name
            scale_z = pixel_scales.get("scale_z") or pixel_scales.get(0x0003)
            if scale_z is not None and isinstance(scale_z, dict):
                height_scale_factor_nm = scale_z.get("value")
        except (KeyError, TypeError):
            pass

        # Convert height scale factor from nm/count to unit/count
        if height_scale_factor_nm is not None:
            nm_to_unit = get_unit_conversion_factor("nm", unit)
            self._height_scale_factor = height_scale_factor_nm * nm_to_unit
        else:
            self._height_scale_factor = None

        # Build metadata dictionary from parsed TLV structure
        parsed_metadata = self._extract_metadata(main)
        parsed_metadata["image_params"] = {
            "nx": nx,
            "ny": ny,
            "physical_size_x_mm": image_params["physical_size_x"],
            "physical_size_y_mm": image_params["physical_size_y"],
        }
        if height_scale_factor_nm is not None:
            parsed_metadata["height_scale_factor_nm"] = height_scale_factor_nm

        # Create channel info
        self._channels = [
            ChannelInfo(
                self,
                0,
                name="Default",
                dim=2,
                nb_grid_pts=(nx, ny),
                physical_sizes=(physical_size_x, physical_size_y),
                uniform=True,
                unit=unit,
                info={"raw_metadata": parsed_metadata},
            )
        ]

        ole.close()

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info=None,
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        channel = self._channels[channel_index]
        nx, ny = channel.nb_grid_pts

        # Combine all blocks (already sorted by element_offset)
        combined_data = b"".join(data for pos, data in self._zlib_blocks)

        # Height data is stored as int32 values
        arr_i32 = np.frombuffer(combined_data, dtype="<i4")
        heights = arr_i32[: nx * ny].astype(float).reshape(ny, nx, order="C")

        # Transpose to get (nx, ny) shape expected by SurfaceTopography
        heights = heights.T

        # Determine if masking is needed based on data_type_tag
        # Tag 12: zeros indicate undefined/masked pixels
        # Tag 39: pure int32, all values including zeros are valid
        if self._data_type_tag == 12:
            invalid_mask = heights == 0
        else:
            invalid_mask = None

        # Apply height scale factor
        # If user provides explicit height_scale_factor, use that
        # Otherwise, use the internal scale factor from pixel_scales if available
        if height_scale_factor is not None:
            heights = heights * height_scale_factor
        elif self._height_scale_factor is not None:
            # Apply internal scale factor (converts counts to µm)
            heights = heights * self._height_scale_factor

        # Check physical sizes
        sx, sy = channel.physical_sizes
        if physical_sizes is not None:
            sx, sy = physical_sizes

        # Build info dict
        _info = channel.info.copy()
        if info is not None:
            _info.update(info)

        if invalid_mask is not None and invalid_mask.any():
            heights = np.ma.masked_array(heights, mask=invalid_mask)

        topography = Topography(
            heights,
            physical_sizes=(sx, sy),
            unit=unit if unit is not None else channel.unit,
            info=_info,
            periodic=periodic,
        )

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
