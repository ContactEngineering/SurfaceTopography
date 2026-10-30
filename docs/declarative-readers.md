# Declarative readers

```python
from SurfaceTopography.IO.Reader import DeclarativeReaderBase, CompoundLayout, If, Switch, Seek
from SurfaceTopography.IO.binary import BinaryStructure, BinaryArray, Validate
from SurfaceTopography.IO.expr import C, V, F, Tup, Cond
from SurfaceTopography.IO.description import reader_description, reader_from_dict
```

Most binary file readers in SurfaceTopography are *declarative*: instead of
parsing code, a reader declares

- a **layout**, a tree of layout nodes (`CompoundLayout`, `BinaryStructure`,
  `BinaryArray`, `ZipContainer`, ...) that describes how the file is laid out,
- **channel bindings**, which map the parsed metadata to channels (grid size,
  physical size, units, height scale factor, undefined-data rule), and
- **registration metadata**: format id, name, file extensions, MIME types and
  magic bytes.

A generic engine, `DeclarativeReaderBase`, executes these declarations. All
data-dependent quantities (array shapes, validation conditions, conversions)
are written in a small expression language instead of Python code, so a
complete reader serializes to a JSON *format description document*. The
documents are the interface to a second engine written in C++ (`libsdsio` in
SDSAlgorithms) that executes the same descriptions. The normative definition of
the documents is the [format description contract](format-description-contract.md).

The format knowledge in a reader (byte offsets, masks, padding rules, magic
constants) is often the result of reverse engineering and documented nowhere
else. Keeping it in a single description, rather than in one parser per
language, means the two engines cannot diverge in what they know about a
format; differences can only arise in the engines themselves, and those are
caught by shared conformance tests (see [Conformance](#conformance)).

## Declarative formats

The following readers are fully declarative. They are exported to description
documents by `SurfaceTopography.IO.export` and pinned by conformance goldens;
the capabilities column lists what an engine must implement beyond `core`
(see [Capabilities](#capabilities)).

| Format id | Reader | Capabilities beyond `core` |
|-|-|-|
| `al3d` | `AL3DReader` | |
| `bcr` | `BCRReader` | `text` |
| `frt` | `FRTReader` | |
| `jpk` | `JPKReader` | `tiff` |
| `lext` | `LEXTReader` | `tiff` |
| `mdt` | `MDTReader` | |
| `metropro` | `MetroProReader` | |
| `mi` | `MIReader` | `text` |
| `nms` | `NMSReader` | |
| `opd` | `OPDReader` | |
| `os3d` | `OS3DReader` | |
| `plu` | `PLUReader` | |
| `plux` | `PLUXReader` | `xml`, `zip` |
| `ps` | `PSReader` | `tiff` |
| `sdf` | `SDFReader` | `text` |
| `sur` | `SURReader` | |
| `tmd` | `TMDReader` | |
| `vk` | `VKReader` | `zip` |
| `wsxm` | `WSXMReader` | `text` |
| `x3p` | `X3PReader` | `xml`, `zip` |
| `zmg` | `ZMGReader` | |
| `zon` | `ZONReader` | `xml`, `zip`, `zstd` |

`OIRReader` uses a declarative layout with Python callables in it and its own
`channels` implementation; it is therefore Python-only and not exported. The
remaining readers (e.g. MNT, DATX, Gwyddion, NetCDF, plain-text formats) are
native Python readers. The user-facing list of all formats is on
[Supported file formats](formats.md).

## Anatomy of a reader

A declarative reader is a subclass of `DeclarativeReaderBase` with class
attributes only:

```python
import struct

import numpy as np

from SurfaceTopography.Exceptions import FileFormatMismatch
from SurfaceTopography.IO.binary import BinaryArray, BinaryStructure, Validate
from SurfaceTopography.IO.expr import C, F, V, Tup
from SurfaceTopography.IO.Reader import CompoundLayout, DeclarativeReaderBase


class ToyReader(DeclarativeReaderBase):
    _format = "toy"
    _mime_types = ["application/x-toy"]
    _file_extensions = ["toy"]
    _name = "Toy format"
    _description = "Example of a declarative reader."
    _magic = [(0, b"TOY1")]  # (offset, bytes) alternatives

    _file_layout = CompoundLayout([
        BinaryStructure([
            ("magic", "4s", Validate(V == "TOY1", FileFormatMismatch)),
            ("nb_grid_pts_x", "I"),
            ("nb_grid_pts_y", "I"),
            ("pixel_size", "d"),  # in µm
        ], byte_order="<", name="header"),
        BinaryArray(
            "heights",
            Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),  # row-major
            F.dtype("<i2"),
            conversion_fun=F.transpose(V),  # to (x, y) indexing
        ),
    ])

    _channel_bindings = [{
        "name": "default",
        "dim": 2,
        "nb_grid_pts": Tup(C.header.nb_grid_pts_x, C.header.nb_grid_pts_y),
        "physical_sizes": Tup(C.header.nb_grid_pts_x * C.header.pixel_size,
                              C.header.nb_grid_pts_y * C.header.pixel_size),
        "unit": "µm",
        "height_scale_factor": 1e-3,
        "uniform": True,
        "info": {},
        "data": C.heights,
        "mask": {"source": C.heights, "rule": V == -32768},
    }]


# A 3 x 2 example file with one undefined pixel
heights = np.array([[1, 2, 3], [4, -32768, 6]], dtype="<i2")  # (ny, nx)
with open("example.toy", "wb") as f:
    f.write(struct.pack("<4sIId", b"TOY1", 3, 2, 0.5) + heights.tobytes())

topography = ToyReader("example.toy").topography()
print(topography.nb_grid_pts, topography.unit)  # (3, 2) µm
print(topography.heights())                      # masked array, one pixel masked
```

Opening the file executes the layout and stores the parsed metadata
(`reader.metadata`); `reader.channels` evaluates the channel bindings; the
height data is read only by `reader.topography()`.

The same reader as a document, and a generic reader created from it:

```python
import json

document = reader_description(ToyReader)        # JSON-compatible dict
print(document["capabilities"])                 # ['core']
GenericReader = reader_from_dict(json.loads(json.dumps(document)))
print(GenericReader("example.toy").topography().heights())  # identical
```

## Layout nodes

Each node parses part of the stream and returns a dictionary that is merged
into the parser context, under the node's `name` if it has one. Later nodes
see earlier results through the context; the enclosing context is available as
`__parent__`.

| Node | Module | Purpose |
|-|-|-|
| `CompoundLayout` | `Reader` | Sequence of nodes sharing one context |
| `BinaryStructure` | `binary` | Scalar fields in `struct` notation, with `Validate`/`Convert` hooks |
| `BinaryArray` | `binary` | Bulk array data (lazy, see [Laziness](#laziness)) |
| `RawBuffer`, `TextBuffer` | `binary` | Sized byte or text block |
| `Skip` | `Reader` | Advance the stream without storing |
| `Seek` | `Reader` | Move to an absolute stream position |
| `If` | `Reader` | Select a layout by conditions |
| `Switch` | `Reader` | Select a layout by the value of a key (tag-driven formats) |
| `For` | `Reader` | Repeat a layout a computed number of times |
| `ForEach` | `Reader` | Repeat a layout for each element of a parsed list |
| `While` | `Reader` | Repeat while a condition holds |
| `SizedChunk` | `Reader` | Execute a layout in a size-bounded window, with bounds checking |
| `Check` | `Reader` | Validate a condition between parsed values |
| `Let` | `Reader` | Store computed values without reading |
| `TLVContainer` | `binary` | Tag-length-value block sequences |
| `ZipContainer` | `binary` | Members of a ZIP archive, each with its own layout |
| `XMLStructure` | `binary` | XML document as nested mapping |
| `TIFFContainer` | `binary` | Pages and tags of a TIFF file |
| `TextLine`, `TextHeader`, `TextMatrix` | `binary` | Line-oriented text headers and number matrices |
| `ZlibBlockChain` | `binary` | Chained zlib-compressed blocks |

The parameters of each node are documented in its docstring (`help(Switch)`);
their serialized form and exact semantics are in the
[contract](format-description-contract.md#layout-nodes).

## Expressions

Expressions are built from four roots in `SurfaceTopography.IO.expr`:

| Root | Meaning | Example |
|-|-|-|
| `C` | the parser context | `C.header.nb_grid_pts_x` |
| `V` | the value being processed (field, array, stream) | `(V & 0x03) == 0` |
| `F` | the function registry | `F.dtype("<i2")`, `F.transpose(V)` |
| `Tup(...)` | a tuple, e.g. an array shape | `Tup(C.ny, C.nx)` |

`Cond(condition, then, otherwise)` is a conditional with short-circuit
evaluation, `DictExpr({...})` builds a mapping with static keys, and
`V.isin(a, b, ...)` tests membership. Arithmetic, bitwise and comparison
operators, indexing and slicing are overloaded. Python's `and`, `or` and `not`
cannot be overloaded; use `&` and `|` on parenthesized boolean subexpressions,
or `Cond`. An expression has no truth value; using one in an `if` raises
`TypeError`.

Expressions are callable with the signatures the layout classes use for
Python callables — `(context)` for shapes, sizes and conditions,
`(value, context)` for validators and converters — so layouts accept either.
Only expressions serialize: a Python callable in a layout becomes an *opaque*
marker in the document, and a document with opaque markers cannot be executed
by another engine.

The functions available through `F` form a **closed registry**, listed in the
[contract](format-description-contract.md#function-registry). The registry is
exactly the set of primitives a foreign engine has to implement, so keeping it
closed bounds the work of implementing and testing an engine, and keeps the
expression language from growing into a general-purpose language that every
engine would have to reproduce with identical semantics. A format that cannot
be described with the existing primitives is a candidate for a native reader
rather than for a new registry entry. (`register_function` adds functions for
Python-only use; documents that call them are not portable.)

## Channel bindings

`_channel_bindings` is a list of dictionaries whose values are literals or
expressions over the parsed metadata. The keys are

| Key | Meaning |
|-|-|
| `name`, `dim`, `unit`, `periodic`, `uniform` | channel properties; `dim` 1 yields a line scan |
| `nb_grid_pts`, `physical_sizes` | tuples, $(N_x, N_y)$ and $(L_x, L_y)$ |
| `height_scale_factor` | factor applied to the raw data; absent means unscaled |
| `data_kind`, `data_unit` | for channels that do not contain heights (e.g. `"phase"`) |
| `info` | (nested) metadata; `None` entries are omitted |
| `data` | context path to the lazy array of the channel |
| `mask` | `{"source": ..., "rule": ...}`; the rule sees the source array as `V` and is true for undefined pixels |
| `foreach` | list expression; one channel per element, available as `C.item` and `C.item_index` |
| `where` | filter; no channel is emitted where it evaluates falsy |
| `checks` | per-channel validations raising a named error |

Data arrays are indexed $(x, y)$, i.e. have shape $(N_x, N_y)$; descriptions
transpose or crop row-major file data via conversion expressions. Expressing
the mapping from metadata to channels declaratively, rather than in a
per-format `channels` method, puts the grid size, the scaling and the
undefined-data rule into the description, so that both engines derive them from
the same source.

## Laziness

Reading is split into two phases. Opening a file executes the layout:
scalar fields and small buffers are materialized, while `BinaryArray` (and
lazy `RawBuffer`) nodes record only the position, shape, data type and
conversion of their data and skip over it. The data of a channel is read when
`topography()` is called: the file is reopened, the recorded handle is read
and the conversion and mask rule are applied. Handles inside a
`ZipContainer` reopen the archive member (and reapply decompression);
handles inside a `TIFFContainer` reread the page or tag. Opening a file is
therefore cheap even for large height maps, and corruption of the metadata is
detected on opening. `TextMatrix` and `ZlibBlockChain` are the exceptions:
their values are parsed on opening, since a text region or a block chain
cannot be sized without scanning it.

## Capabilities

Some layout nodes and registry functions depend on container or compression
libraries: `zip` (`ZipContainer`), `zstd` and `zlib` (stream filters,
`ZlibBlockChain`), `xml` (`XMLStructure`), `text` (`TextLine`, `TextHeader`,
`TextMatrix`) and `tiff` (`TIFFContainer`). A description lists the
capabilities it requires (`document["capabilities"]`, computed by
`SurfaceTopography.IO.description.capabilities`). In C++ each capability pulls
in a dependency; an engine built without one reports such a format as known
but unsupported instead of failing while parsing. The Python engine implements
all capabilities.

## Serialization and export

`SurfaceTopography.IO.description` converts between layouts and documents:

| Function | Purpose |
|-|-|
| `reader_description(reader_class)` | complete description document of a reader class |
| `reader_from_dict(document)` | generic `DeclarativeReaderBase` subclass executing a document |
| `layout_to_dict(layout)`, `layout_from_dict(d)` | a single layout tree |
| `count_opaque(document)` | number of non-serializable hooks |
| `capabilities(document)` | sorted capability list |

`SurfaceTopography.IO.export` writes the description documents of all exported
readers and the conformance goldens for their example files in
`test/file_format_examples`:

```
python -m SurfaceTopography.IO.export --descriptions DIR
python -m SurfaceTopography.IO.export --goldens DIR [--corpus CORPUS_DIR]
```

The exporter refuses to write a document that contains opaque markers.

## Two engines

SurfaceTopography is the reference implementation: readers are authored here
as Python classes, and the documents and goldens are generated from them.
Authoring in Python keeps the reverse-engineering loop short (interactive
exploration against real files, tests against the example corpus) and the
descriptions readable (operator overloading, helper functions, comments). The
JSON document is generated, not edited by hand.

The C++ engine is `DeclarativeReader` in `libsdsio` (SDSAlgorithms): an
expression evaluator, a layout interpreter with lazy array handles, and a
format registry keyed by magic bytes, with the description documents embedded
at build time. The documents and goldens are vendored into SDSAlgorithms and
regenerated from SurfaceTopography; adding a declarative format there requires
no C++ code unless the format needs a new layout node, registry function or
capability.

## Conformance

The test suite enforces that the documents are complete and equivalent to
the authored readers:

| Test | Checks |
|-|-|
| `test/IO/test_description.py` | for every exported reader: zero opaque markers; contract-valid document structure; a generic reader rehydrated from the JSON document reproduces channel metadata, heights and undefined pixels of the authored reader on every fixture; magic detection from the document; deterministic export |
| `test/IO/test_expr.py` | expression semantics, JSON round trip, equivalence of expression-based layouts |
| `test/IO/test_declarative_reader_serialization.py` | serialized form of layout nodes, opaque markers, rehydration, capability computation |
| `test/IO/test_declarative_reader.py` | engine features (line scans, data kinds, loops) |

The list of exported readers and their fixtures (`_EXPORTED_READERS` in
`SurfaceTopography/IO/export.py`) parametrizes `test_description.py`, so every
exported reader is covered automatically. On the C++ side, a conformance runner
executes the same goldens; the goldens compare probe pixels and masked
statistics over the full arrays (see the
[contract](format-description-contract.md#conformance-goldens)), so
differences in byte order, division semantics or mask handling surface as test
failures.

## Adding a format

1. Write the reader as a `DeclarativeReaderBase` subclass: `_file_layout` and
   `_channel_bindings` built from expressions (no Python callables), `_magic`
   for detection. Add an example file to `test/file_format_examples` and
   per-format tests in `test/IO/`, and register the reader in
   `SurfaceTopography/IO/__init__.py`.
2. Add the format id and its fixtures to `_EXPORTED_READERS` and the class to
   `exported_reader_classes()` in `SurfaceTopography/IO/export.py`.
   `test_description.py` then checks that the description is complete and that
   the generic reader reproduces the authored one.
3. In SDSAlgorithms, regenerate the vendored descriptions and goldens; the C++
   conformance runner picks up the new goldens.
4. If a new registry function, layout node or capability is unavoidable, revise
   the contract (incrementing the schema version per its change policy) and
   implement the primitive in both engines.

## Design considerations

### Semantic divergence at the edges

Byte order and alignment, floor versus truncating division, string
encodings and NaN handling are where independent implementations
typically disagree. The contract specifies these normatively, and the
goldens include masked statistics over full arrays rather than only probe
pixels.

### Formats beyond the declarative model

Formats with nested tag-length-value structures and tag-driven layout
switching (e.g. OIR) test the limits of the layout nodes. Where a format
cannot be expressed with the existing primitives, it remains a native
reader rather than extending the registry.

### Related systems

[Kaitai Struct](https://kaitai.io/) (a YAML-based binary format language
compiled to parsers in many languages) is the closest existing system and
the design reference for the expression language. DFDL (Data Format
Description Language) is an example of a description language whose
expression language grew large. [Wuffs](https://github.com/google/wuffs)
generates memory-safe parsers for untrusted input; interpreting
descriptions likewise centralizes bounds checks in one engine instead of
in per-format parsing code.
