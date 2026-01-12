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

Height data formats:
- Int16 pairs: (height_value, validity_flag) - high bytes are 0 (valid) or -1 (invalid)
- Pure int32: Full int32 height values without validity channel

Note: 0xFFFF tags are used as section markers/delimiters and must be skipped
when parsing nested containers.
"""

import struct
import zlib
from io import BytesIO

import numpy as np
import olefile

from ..Exceptions import CorruptFile, FileFormatMismatch
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
    _TAG_WIDTH = 0x0007
    _TAG_HEIGHT = 0x0008
    _TLV_TAG_FORMAT = "<H"
    _TLV_SIZE_FORMAT = "<Q"

    # TLV tags for image parameters (found in Section 3 of compressed_blocks)
    _TAG_PHYSICAL_SIZE_X = 0x0009
    _TAG_PHYSICAL_SIZE_Y = 0x000A

    @classmethod
    def _parse_sections(cls, data):
        """
        Parse the section structure within compressed_blocks.

        The compressed_blocks data has this structure:
        - [0:10] Outer container (tag 0x0001, size uint64)
        - [10:18] uint64 value (unknown purpose)
        - [18:...] Sections as TLV containers, each followed by uint64 size field

        Sections are:
        - Section 1: Compressed height data blocks (large)
        - Section 2: Additional metadata
        - Section 3: Image parameters (contains width, height, physical sizes)

        Parameters
        ----------
        data:bytes
            Raw compressed_blocks data.

        Returns
        -------
        sections:list of bytes
            List of section data (excluding size headers).
        """
        sections = []

        if len(data) < 28:
            return sections

        # Skip outer container header (10 bytes) and first uint64 value (8 bytes)
        inner = data[18:]

        pos = 0
        while pos < len(inner) - 10:
            # Read TLV entry for this section
            tag = struct.unpack("<H", inner[pos:pos + 2])[0]
            size = struct.unpack("<Q", inner[pos + 2:pos + 10])[0]

            if tag > 0x2000 or size > len(inner) - pos - 10:
                break

            section_data = inner[pos + 10:pos + 10 + size]
            sections.append(section_data)

            # Move past this section
            section_end = pos + 10 + size

            # Check for uint64 size field after section (separator between sections)
            if section_end + 8 <= len(inner):
                next_size = struct.unpack("<Q", inner[section_end:section_end + 8])[0]
                if next_size < 100000:  # Reasonable size for next section
                    pos = section_end + 8
                else:
                    break
            else:
                break

        return sections

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

        # Parse sections
        sections = cls._parse_sections(data)

        # Section 3 contains image parameters (index 2, 0-based)
        if len(sections) < 3:
            return result

        section3 = sections[2]

        # Find the innermost container with image parameters
        params = cls._find_image_params_container(section3)
        if params is None:
            return result

        # Extract values using declarative names or tag IDs
        # Try by name first (more readable), then by tag ID
        result["width"] = params["width"]["value"]
        result["height"] = params["height"]["value"]
        result["physical_size_x"] = params["physical_size_x"]["value"]
        result["physical_size_y"] = params["physical_size_y"]["value"]

        return result

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
        - Stored as int16 pairs: (height_value, validity_flag)
        - Height values at even indices, validity at odd indices
        - Validity: 0 = valid, -1 = invalid/masked
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

        # Container 0x0006 children - [GUESS] possibly pixel scales/units
        # Evidence: Similar structure to SUR format scale blocks
        pixel_scale_children = {
            0x0001: Skip(comment="possibly X scale"),
            0x0002: Skip(comment="possibly Y scale"),
            0x0003: Skip(comment="possibly Z scale"),
            0x0004: TextBuffer("x_unit"),  # [LIKELY] Contains ASCII text (unit?)
            0x0005: TextBuffer("y_unit"),  # [LIKELY] Contains ASCII text (unit?)
            0x0006: TextBuffer("z_unit"),  # [LIKELY] Contains ASCII text (unit?)
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

        # Container 0x02bd children - [CONFIRMED] height data
        # Evidence: Contains TLV metadata followed by zlib-compressed blocks
        # Note: compressed_blocks is stored as raw because it contains both
        # TLV metadata (with dimensions) and binary zlib-compressed data
        height_data_children = {
            0x0001: Skip(comment="data header, appears before compressed data"),
            0x0002: RawBuffer("compressed_blocks", size=None, lazy=False),
        }

        # Container 0x012d children - [UNKNOWN]
        extended_metadata_children = {
            0x0001: TextBuffer("extended_description"),  # [LIKELY] Contains ASCII text
            0x0002: Skip(comment="container with unknown structure, extended params"),
            0x0003: Skip(comment="extended flags"),
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

        # Find all zlib-compressed blocks within the compressed data
        # Each block has a 16-byte prefix:
        #   Bytes 0-7:  uint64 LE - Element offset (for ordering)
        #   Bytes 8-11: uint32 LE - Elements per block
        #   Bytes 12-15: uint32 LE - Compressed size
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
        # Tags 0x0009 and 0x000a contain physical sizes in mm
        physical_size_x = image_params["physical_size_x"]
        physical_size_y = image_params["physical_size_y"]

        # Convert mm to µm if physical sizes were found
        if physical_size_x is not None and physical_size_y is not None:
            # Values are in mm, convert to µm
            physical_size_x = physical_size_x * 1000.0  # mm -> µm
            physical_size_y = physical_size_y * 1000.0  # mm -> µm
            unit = "µm"
        else:
            # Fallback: use pixel count as physical size
            physical_size_x = float(nx)
            physical_size_y = float(ny)
            unit = "µm"

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
                info={
                    "raw_metadata": {
                        "nx": nx,
                        "ny": ny,
                        "physical_size_x_mm": image_params["physical_size_x"],
                        "physical_size_y_mm": image_params["physical_size_y"],
                    }
                },
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
        nx, ny = channel.nb_grid_pts

        # Combine all blocks (already sorted by element_offset)
        combined_data = b"".join(data for pos, data in self._zlib_blocks)

        # Detect data format by checking the high int16 bytes
        # Int16 pairs format: high bytes are only 0 (valid) or -1 (invalid)
        # Pure int32 format: high bytes have varying values
        arr_i16 = np.frombuffer(combined_data, dtype="<i2")
        high_bytes = arr_i16[1::2]  # Odd indices = high bytes of int32s
        unique_high = np.unique(high_bytes)
        is_int16_pairs = len(unique_high) <= 2 and set(unique_high).issubset({-1, 0})

        if is_int16_pairs:
            # Int16 pairs format: (height_int16, validity_int16)
            heights_flat = arr_i16[::2].astype(float)  # Even indices = heights
            validity_flat = arr_i16[
                1::2
            ]  # Odd indices = validity (0=valid, -1=invalid)
            # Reshape using C-order (row-major storage)
            heights = heights_flat[: nx * ny].reshape(ny, nx, order="C")
            invalid_mask = (validity_flat[: nx * ny] == -1).reshape(ny, nx, order="C")
        else:
            # Pure int32 format
            arr_i32 = np.frombuffer(combined_data, dtype="<i4").astype(float)
            # Reshape using C-order (row-major storage)
            heights = arr_i32[: nx * ny].reshape(ny, nx, order="C")
            # No built-in validity mask
            invalid_mask = np.zeros((ny, nx), dtype=bool)

        # Transpose to get (nx, ny) shape expected by SurfaceTopography
        heights = heights.T
        invalid_mask = invalid_mask.T

        # Apply scale factor if provided
        if height_scale_factor is not None:
            heights = heights * height_scale_factor

        # Check physical sizes
        sx, sy = channel.physical_sizes
        if physical_sizes is not None:
            sx, sy = physical_sizes

        # Build info dict
        _info = channel.info.copy()
        _info.update(info)

        if invalid_mask.any():
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
