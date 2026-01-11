Digital Surf Mountains MNT Format
=================================

This document describes the internal structure of the Digital Surf Mountains MNT
file format, based on reverse engineering of example files.

Overview
--------

The MNT format is used by Digital Surf's Mountains software suite for storing
surface topography data from profilometers, interferometers, and other surface
measurement instruments. It is a Microsoft Compound Document File (OLE/COM
structured storage) containing multiple streams.

File Structure
--------------

The file uses the OLE compound document format, identifiable by the magic bytes:

.. code-block:: text

    D0 CF 11 E0 A1 B1 1A E1

The file contains the following streams:

ImagePreview
    A JPEG image showing a preview of the topography. Note that this preview
    may include user interface elements and its aspect ratio does not necessarily
    match the actual data dimensions.

ScopedContents
    Binary stream containing the height data. This is the main data stream with
    a global header followed by zlib-compressed data blocks.

ScopedResults
    Parameter data and analysis results. The exact structure is not fully
    documented.

XmlHeader
    UTF-16 encoded XML metadata. Contains general file information (software
    version, serial number, operators) but does not store the data dimensions.

ScopedContents Structure
------------------------

The ScopedContents stream uses a hierarchical TLV (Tag-Length-Value) encoding
throughout. The entire stream can be traversed as a sequence of TLV entries.

TLV Entry Format
++++++++++++++++

Each TLV entry has the following structure:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - Tag
     - uint16 LE
     - Identifies the entry type
   * - Size
     - uint64 LE
     - Number of bytes in the data section
   * - Data
     - variable
     - The payload (size bytes, may contain nested TLV entries)

Top-Level Structure
+++++++++++++++++++

The stream begins with a size field followed by top-level TLV entries:

.. code-block:: text

    Offset 0x0000: uint64 LE - Total stream size minus 8 (integrity check)
    Offset 0x0008: Tag 0x03, size=1 - Format version
    Offset 0x0013: Tag 0x02, size=4 - Format flags
    Offset 0x0021: Tag 0x04, size=8 - Timestamp or identifier
    Offset 0x0033: Tag 0x05, size=8 - Contains dimension parameters
    Offset 0x0045: Tag 0x01        - Main container (rest of stream)

The Tag 0x01 container holds all remaining data as nested TLV entries.

Known Tags
++++++++++

Tags observed in example files:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Tag
     - Description
   * - 0x0001
     - Container/nested structure
   * - 0x0002-0x0005
     - Top-level metadata
   * - 0x00c8
     - Metadata block (variable size)
   * - 0x00c9
     - 4-byte value
   * - 0x00ca
     - Metadata block (variable size)
   * - 0x00cb
     - Serial number string (UTF-16)
   * - 0x012d
     - Extended metadata
   * - 0x02bd
     - Height data container (contains compressed blocks)
   * - 0xffff
     - Marker/separator

UTF-16 String Encoding
++++++++++++++++++++++

String values (e.g., serial numbers) have a type byte prefix:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - Type byte
     - uint8
     - Value 0x04 indicates a UTF-16 string
   * - String
     - UTF-16 LE
     - The string data (size - 1 bytes)

Example: Serial number "DS-364280957" is encoded as:

.. code-block:: text

    cb 00                       Tag 0xcb (serial number)
    19 00 00 00 00 00 00 00     Size = 25 bytes
    04                          Type byte (string)
    44 00 53 00 2d 00 ...       UTF-16 LE "DS-364280957" (24 bytes)

Dimension Parameters
++++++++++++++++++++

Key fields for calculating image dimensions are found at fixed byte offsets
within the Tag 0x05 data block (offsets from start of ScopedContents):

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Offset
     - Type
     - Description
   * - 0x3d
     - uint8
     - Number of compressed data blocks
   * - 0x63
     - uint8
     - Rows factor A (first component of rows per block)
   * - 0x71
     - uint8
     - Rows factor B (second component of rows per block)

Height Data Container (Tag 0x02bd)
----------------------------------

The height data is stored in a nested TLV structure within Tag 0x02bd:

.. code-block:: text

    Tag 0x02bd (height data container)
    ├── Tag 0x0001, size=1: num_blocks (e.g., 32)
    └── Tag 0x0002 (compressed data container)
        └── Tag 0x0001 (packed block data)
            └── [16-byte prefix + zlib data] × num_blocks

**Important**: The innermost level does NOT use TLV encoding. The compressed
blocks are packed sequentially with a fixed 16-byte prefix before each zlib
stream.

Block Prefix Structure
++++++++++++++++++++++

Each compressed block is preceded by a 16-byte prefix with a **fixed structure**
(not TLV):

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Offset
     - Type
     - Description
   * - 0-7
     - uint64 LE
     - Element offset. Used to determine the correct ordering of blocks
       when reconstructing the height map. Blocks must be sorted by this
       value in ascending order.
   * - 8-11
     - uint32 LE
     - Elements per block. The number of height values (not bytes) contained
       in this block after decompression.
   * - 12-15
     - uint32 LE
     - Compressed size. The size in bytes of the following zlib-compressed
       data.

The packed block format is:

.. code-block:: text

    [element_offset (8)] [elements_per_block (4)] [compressed_size (4)] [zlib data]
    [element_offset (8)] [elements_per_block (4)] [compressed_size (4)] [zlib data]
    ... (repeated for each block)

Zlib Compression
++++++++++++++++

The compressed data uses standard zlib compression. Common zlib header bytes:

- ``78 01`` - No compression
- ``78 5E`` - Fast compression
- ``78 9C`` - Default compression
- ``78 DA`` - Best compression

Block Identification
++++++++++++++++++++

Height data blocks can be identified by:

1. Finding all zlib-compressed blocks with substantial decompressed size
   (typically ≥50000 bytes for height data)
2. Grouping blocks by their ``elements_per_block`` value
3. The largest group of blocks with the same ``elements_per_block`` contains
   the height data

Dimension Calculation
---------------------

The image dimensions are calculated from header parameters and block metadata:

.. code-block:: text

    rows_per_block = header[0x63] × header[0x71]
    width = elements_per_block / rows_per_block
    height = num_blocks × rows_per_block

Where:

- ``header[0x3d]`` is the number of blocks (``num_blocks``)
- ``header[0x63]`` is rows factor A
- ``header[0x71]`` is rows factor B
- ``elements_per_block`` is read from the block prefix (bytes 8-11)

Example calculations:

.. code-block:: text

    File mnt-1.mnt:
      num_blocks = 32
      factor_a = 9, factor_b = 2
      rows_per_block = 9 × 2 = 18
      elements_per_block = 18000
      width = 18000 / 18 = 1000
      height = 32 × 18 = 576
      Result: 1000 × 576

    File mnt-2.mnt:
      num_blocks = 32
      factor_a = 10, factor_b = 3
      rows_per_block = 10 × 3 = 30
      elements_per_block = 38400
      width = 38400 / 30 = 1280
      height = 32 × 30 = 960
      Result: 1280 × 960

Data Format
-----------

MNT files store height data as interleaved 16-bit signed integer pairs:

.. code-block:: text

    [height_0, secondary_0, height_1, secondary_1, height_2, secondary_2, ...]

- **Height values** (even indices): 16-bit signed integers representing height
  in raw units
- **Secondary values** (odd indices): May contain different types of data:

  - **Validity flag**: If only values 0 and -1 are present, this indicates
    validity (0 = valid, -1 = invalid/masked)
  - **Other data**: Some files contain small integer values (e.g., error
    estimates) in the secondary channel; these are ignored for height
    reconstruction

The data format is always 4 bytes per pixel (two int16 values), regardless of
the secondary channel content.

Block Ordering
--------------

**Important**: Blocks may not appear in the file in the correct order for
reconstruction. The ``element_offset`` value (bytes 0-7 of each block prefix)
must be used to sort blocks before combining them.

Example block ordering:

.. code-block:: text

    Block 0: element_offset = 0      → First block (top of image)
    Block 1: element_offset = 18000  → Second block
    Block 2: element_offset = 36000  → Third block
    ...

The element offset indicates the starting position of each block's data in
the final reconstructed array.

Data Reconstruction
-------------------

To reconstruct the height map:

1. Collect all height data blocks (identified by ``elements_per_block``)
2. Sort blocks by their ``element_offset`` value (ascending)
3. Concatenate the decompressed data in sorted order
4. Parse as int16 pairs: extract heights from even indices
5. Reshape to (height, width) and transpose to (width, height) for row-major
   storage
6. If the secondary channel contains only 0 and -1 values, create a mask
   where ``secondary != 0`` indicates invalid data. Otherwise, mask only
   extreme int16 values (near ±32767).

Physical Units
--------------

The MNT format does not reliably store physical sizes or unit information in
an easily accessible location. Users should:

- Provide physical sizes manually when loading
- Height values are in raw integer units; a scale factor may be needed

Example Files
-------------

**mnt-1.mnt**: Uses int16 pairs format. Dimensions: 1000 × 576. The secondary
channel contains small integer values (not a validity flag), so only extreme
int16 values (near ±32767) are masked. A small number of pixels (~316) have
invalid data.

**mnt-2.mnt**: Uses int16 pairs format with validity flag in the secondary
channel (0 = valid, -1 = invalid). Dimensions: 1280 × 960. Corner regions
contain masked (invalid) data.

Known Limitations
-----------------

- Physical sizes must be provided by the user
- Height scale factor not automatically extracted
- The meaning of storing rows_per_block as two factors (A × B) is not
  understood
- The complete TLV tag vocabulary is not documented; only tags observed in
  example files are described

Related Formats
---------------

The MNT TLV structure is similar to the MicroProf FRT format, also used by
Digital Surf software. Both use:

- 2-byte tags (uint16 LE)
- 8-byte sizes (uint64 LE in newer versions)

Key differences:

- FRT uses a flat block list with a count at the start
- MNT uses nested containers (Tag 0x01 wraps most content)
- Some tag IDs overlap (0x000b, 0x0065, 0x0066)

References
----------

The MNT format is proprietary to Digital Surf. This documentation is based on
reverse engineering and may be incomplete or contain errors.
