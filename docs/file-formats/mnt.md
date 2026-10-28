# Digital Surf Mountains MNT

```python
from SurfaceTopography import open_topography

reader = open_topography("mnt-2.mnt")
topography = reader.topography()
```

MNT files are the documents of the Digital Surf Mountains software
(MountainsMap). A document contains the original measurement together with
the operators applied to it (levelling, form removal, ...) and the studies
(views, parameter tables, ...). The reader (format identifier `mnt`, class
`SurfaceTopography.IO.MNT.MNTReader`) returns the original measurements. Derived
layers are not stored in the file; Mountains recomputes them from the
operators when it opens the document.

The format is proprietary and undocumented. This page describes the structure
as far as the reader uses it; it was determined from example files and may be
incomplete.

## What the reader returns

| | Source |
|-|-|
| channels | one per measured surface whose height axis has a length unit; other surfaces (e.g. intensity) are skipped |
| `name` | name of the measurement, else the title of the surface object |
| `nb_grid_pts` | number of grid points of the two lateral axes |
| `physical_sizes` | grid spacing times number of grid points, see [below](#heights-and-physical-sizes) |
| `unit` | display unit of the height axis; lateral sizes are converted to it |
| `height_scale_factor` | height step per count, in `unit` |
| undefined data | points marked as not measured in the mask |
| `info["raw_metadata"]` | measurement name, title, GUID, axis scales, range of counts, and the software that wrote the file (`name`, `version`, `build_date`, `serial_number`, `operators`) |

Physical sizes, unit and height scale are stored in the file, so passing
`physical_sizes`, `unit` or `height_scale_factor` to `topography` raises
`MetadataAlreadyFixedByFile`. The reader does not support MPI-parallel reading.

## File structure

An MNT file is a Microsoft Compound Document (OLE structured storage), with the
signature `D0 CF 11 E0 A1 B1 1A E1`. It contains the streams

| Stream | Content | Used |
|-|-|-|
| `ScopedContents` | the document, a tree of TLV entries | yes |
| `XmlHeader` | XML with name, version and build date of the Mountains software and the operators used in the document, stored as an MFC `CArchive` string (UTF-16) | metadata |
| `ScopedResults` | results of the studies (e.g. parameter tables) | no |
| `ImagePreview` | JPEG preview of the document | no |

The `XmlHeader` string starts with `FF FE FF` (Unicode string), followed by the
number of characters as uint8; the values `FF` and `FF FF` indicate that the
number follows as uint16 or uint32.

### TLV entries

`ScopedContents` starts with the size of the remaining stream (uint64 LE),
followed by a tree of entries

| Field | Type | Description |
|-|-|-|
| tag | uint16 LE | identifies the entry within its parent |
| size | uint64 LE | number of bytes of the data |
| data | `size` bytes | a value or nested entries |

Text is stored as the marker byte `0x04` followed by UTF-16 LE characters.
Entries of the list of document objects are preceded by an additional size
field (uint64 LE). All integers are little endian; values are uint32 (`u32`),
int32 (`i32`) or float64 (`f64`) unless noted otherwise. Entries not listed
below are skipped; among them are those describing the report page (page size
in pixels, margins in mm, zoom, DPI).

### Document

```text
0x0001  main container
├── 0x00C8  software version: 0x0001 major, 0x0002 minor, 0x0003 patch, 0x0004 build (u32)
├── 0x00CB  serial number of the Mountains installation (text; not the instrument)
└── 0x02BD  contents
    ├── 0x0001  number of objects (uint8)
    └── 0x0002  document
        └── 0x0001  list of objects (entries with tag 0x0001, each preceded by its size)
```

Every document object has the entries

| Tag | Content |
|-|-|
| `0x0001` | class: `0x0001` class number (u32) |
| `0x0002` | body, parsed according to the class |
| `0x0003` | GUID: `0x0001` data1 (u32), `0x0002` data2 (uint16), `0x0003` data3 (uint16), `0x0004` data4 (8 bytes) |

The class numbers used by the reader are

| Class | Object |
|-|-|
| 9012 | original (measured) surface |
| 9062 | measurement inside a surface object |
| 9049 | array of height counts |
| 9056 | mask of non-measured points |
| 9048 | data array of the mask |

Only class 9012 stores height data. Derived layers are objects of other
classes that store their operators and a reference to their parent.

### Measured surface

```text
surface (class 9012) body
├── 0xFFFF  header
│   ├── 0xFFFF  identity: 0x0001 GUID (the "StudiableGUID" of the XML header)
│   └── 0x0004  title (text)
└── 0x0001  object of class 9062, whose body is
    └── 0xFFFF
        └── 0xFFFF  measurement
            ├── 0xFFFF  grid
            │   ├── 0xFFFF  header: 0xFFFF description: 0x0001 name (text)
            │   ├── 0x0001  lateral axis x
            │   ├── 0x0002  lateral axis y
            │   └── 0x0003  mask (optional, class 9056)
            ├── 0x0001  heights (class 9049)
            └── 0x0002  height axis
```

An axis scale has the entries

| Tag | Type | Content |
|-|-|-|
| `0xFFFF` | | label: `0x0001` index (u32), `0x0002` name (text) |
| `0x0001` | f64 | step: grid spacing, or height per count |
| `0x0002` | text | unit of the step |
| `0x0003` | text | display unit |
| `0x0004` | f64 | display unit in units of the step unit |
| `0x0005` | f64 | origin, in the display unit |

A lateral axis is a container `0xFFFF` holding the scale (`0xFFFF`) and the
number of grid points (`0x0001`, u32), the first (`0x0002`) and the last
(`0x0003`) index. The height axis holds the scale (`0xFFFF`) and the minimum
(`0x0001`, i32) and maximum (`0x0002`, i32) count.

### Arrays

The heights (class 9049) and the data of the mask (class 9048) are array
objects with the body `0x0002`, which contains the number of elements
(`0x0001`, u32) and the data (`0x0002`). The data starts with its total size in
bytes (uint64 LE), followed by zlib-compressed blocks, each with a fixed
16-byte header (not a TLV entry):

| Offset | Type | Content |
|-|-|-|
| 0 | uint64 LE | offset of the block in the array, in elements |
| 8 | uint32 LE | number of elements in the block |
| 12 | uint32 LE | compressed size in bytes |
| 16 | | zlib stream |

The blocks are not necessarily stored in the order of their offsets; each
block is placed at its offset. The size of an element is the total size
divided by the number of elements. The reader accepts heights stored as
16-bit or 32-bit signed integers; both example files store 32-bit counts.

The mask (class 9056) contains its width (`0x0001`, u32) and height
(`0x0002`, u32), which equal the number of grid points, and the data array
(`0x0005`). The data is an image of $(N_x + 2)(N_y + 2)$ bytes with a border of
one pixel. The value 9 marks measured points and 22 non-measured points; the
reader treats every value other than 9 as not measured.

## Heights and physical sizes

The counts $c_{ij}$ are stored row by row with $x$ running fastest: element
$k = i + N_x j$ of the array is the count at grid point $(i, j)$. Let
$\delta_x$, $\delta_y$ and $\delta_z$ be the steps of the $x$, $y$ and height
axes, converted to the display unit of the height axis, and $z_0$ the origin
of the height axis (stored in that unit). The reader computes

$$
h_{ij} = z_0 + c_{ij}\,\delta_z
$$

and

$$
L_x = N_x\,\delta_x, \qquad L_y = N_y\,\delta_y .
$$

The origins of the lateral axes position the scan on the sample; they are
reported in `info["raw_metadata"]["axes"]` but not used.

## Example files

The files are in `test/file_format_examples` of the source repository.

| | `mnt-1.mnt` | `mnt-2.mnt` |
|-|-|-|
| Mountains version | 9.2.0.9994 | 10.3.3.10995 |
| $N_x \times N_y$ | 960 × 600 | 1280 × 960 |
| $\delta_x$, $\delta_y$ | 1.8534 µm, 1.8546 µm | 74.880 nm |
| $\delta_z$ | $10^{-4}$ nm | 0.23273 nm |
| display unit | µm | µm |
| blocks × elements per block | 32 × 18000 | 32 × 38400 |
| element size | 4 bytes | 4 bytes |
| mask | none | 45405 non-measured points |

`mnt-2.mnt` was created from `mnt-2.opd`
([Wyko OPD](../formats.md#topography-files)). Both files give the same physical
sizes and undefined points, and the heights agree to within the height step
$\delta_z$.

## Related formats

FRT MicroProf files use a similar TLV encoding with 2-byte tags and 4-byte or
8-byte sizes, but a flat list of entries instead of the nested containers of
MNT.
