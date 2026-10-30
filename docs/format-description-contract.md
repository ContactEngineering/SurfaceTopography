# Format description contract

**Schema version: 1**

This page is the normative contract between producers of format description
documents (the SurfaceTopography exporter, `SurfaceTopography.IO.export`) and
the engines that execute them (SurfaceTopography's `DeclarativeReaderBase` and
any other implementation of this contract). Where behavior is not
specified here, engines must not rely on it agreeing across implementations.
The architecture around the contract is described in
[Declarative readers](declarative-readers.md).

The key words *must*, *must not*, *should* and *may* are to be interpreted as
in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

Sections marked **[draft]** may still change; everything else is stable within
schema version 1.

## Document structure

A format description is a single JSON document:

```json
{
  "schema_version": 1,
  "format": {
    "id": "zon",
    "name": "Keyence ZON",
    "description": "...",
    "file_extensions": ["zon"],
    "mime_types": ["application/x-keyence-zon"],
    "magic": [
      {"offset": 0, "bytes": "S1BLMA=="},
      {"offset": 0, "bytes": "S1BLMQ=="}
    ]
  },
  "capabilities": ["core", "xml", "zip", "zstd"],
  "layout": { ...layout node... },
  "channels": [ ...channel bindings... ]
}
```

- `schema_version`: Integer. Engines *must* refuse documents with a version greater than the
  one they implement, with a distinguishable "unsupported schema" error. Any
  change to this contract that an existing engine cannot safely ignore
  requires incrementing the version.

- `format.magic`: List of alternatives; each alternative matches if the file contains the
  given bytes (base64-encoded) at the given offset. Detection semantics:
  *yes* if any alternative matches; *no* if the probe buffer covers all
  alternatives and none match; *maybe* if the buffer is too short. An empty
  list means detection always answers *maybe* (trial parse required).

- `capabilities`: The complete, sorted list of capabilities the description requires (see
  [Capabilities](#capabilities)); it always contains `core`. Engines *must*
  be able to decide support from this list alone, without walking the
  layout.

## Value model

Values flowing through the parser context and expressions are:

- **null** — also the representation of scalar NaN read from a file (see
  below).
- **boolean**
- **integer** — signed 64-bit. Unsigned fields are widened; values outside the
  signed 64-bit range are outside this contract.
- **float** — IEEE-754 double precision. Fields of lower precision are widened
  on read.
- **string** — Unicode. String fields of binary structures (`s`, `u`, `U`) have
  leading and trailing NUL characters removed, then leading and trailing
  spaces (U+0020). Other nodes specify their own stripping rules below.
- **bytes** — serialized as base64 where they appear in documents.
- **array** — n-dimensional, typed; produced only by array layout primitives
  and by expression operations on arrays.
- **mapping / list** — nested context structures produced by layout nodes.

- **Scalar NaN policy**: A scalar float field whose value is NaN *must* be stored in the context as
  null. (Rationale: NaN compares unequal to itself, so NaN values would break
  equality of metadata dictionaries.) Arrays keep their NaN payloads;
  undefined *data points* are represented by masks, never by NaN sentinel
  comparisons hidden in engine code.

- **Datetimes**: The neutral representation of a point in time is an ISO-8601 string with
  offset (e.g. `"2021-03-24T13:49:17-04:00"`). The `parse_datetime` registry
  function normalizes vendor date strings to this form. Engines *may* expose
  a native datetime type to their host language (the Python engine returns
  `datetime.datetime` objects), but the serialized form is the ISO string.

## Expressions

Expressions are trees of nodes, each a JSON object with a `kind` key. Engines
*must* reject unknown `kind` values with a distinguishable error. The Python
authoring API lives in `SurfaceTopography.IO.expr`.

### Node catalog

- `{"kind": "lit", "value": <json>}`: Literal. `value` is any JSON value that survives a JSON round trip
  (mappings must have string keys).

- `{"kind": "bytes", "value": "<base64>"}`: Bytes literal.

- `{"kind": "val"}`: The value currently being processed (the field under validation or
  conversion, the array under conversion, the stream under filtering).

- `{"kind": "ctx", "path": ["header", "nb_grid_pts_x"]}`: Context reference; resolves the path segment by segment in the current
  parser context. The segment `__parent__` refers to the enclosing context.
  Resolution failure is an engine error (corrupt description or file).

- `{"kind": "binop", "op": "<op>", "args": [<node>, <node>]}`: Binary operation. Operators: `+ - * / // % & | ^ << >> == != < <= > >= in`.

- `{"kind": "unop", "op": "<op>", "arg": <node>}`: Unary operation. Operators: `neg`, `not`.

- `{"kind": "call", "name": "<fn>", "args": [<node>, ...]}`: Invocation of a registry function (see
  [Function registry](#function-registry)). Unknown names *must* produce a
  distinguishable "unsupported function" error.

- `{"kind": "tuple", "items": [<node>, ...]}`: Fixed-length sequence (e.g. an array shape).

- `{"kind": "cond", "condition": <node>, "then": <node>, "otherwise": <node>}`: Conditional with short-circuit evaluation: only the selected branch is
  evaluated.

- `{"kind": "getitem", "base": <node>, "index": [<part>, ...], "tuple": <bool>}`: Indexing and slicing. Each part is a node or
  `{"kind": "slice", "bounds": [<start>, <stop>, <step>]}` with each bound
  null or a node. `tuple` records whether the index was multi-axis.

- `{"kind": "dict", "items": {"<key>": <node>, ...}}`: Mapping with static string keys, for context restructuring. Computed keys
  are not part of schema version 1.

### Operator semantics

These follow Python semantics; C++ implementations *must* reproduce them:

- `/` is true division and always yields a float.
- `//` is **floor** division and `%` follows the sign of the divisor. C++ `/`
  on integers truncates toward zero and `%` follows the dividend — a C++
  engine *must not* map these operators directly.
- `==`/`!=` compare by value; comparing values of unrelated types is unequal
  rather than an error. Ordering comparisons are defined for number–number and
  string–string (Unicode code point order) operands only.
- `in` tests membership of a scalar in a list.
- Bitwise operators require integers (or booleans, treated as 0/1).
- Arithmetic, comparison and bitwise operators applied to arrays operate
  elementwise with numpy broadcasting semantics; `getitem` follows
  numpy/Python indexing including negative indices and slice clamping.
  Applied to a mapping, a string index performs key lookup (so context paths
  can continue past an index, e.g. `entries[0].prefix`).
- The condition of `cond` must evaluate to a boolean or integer (nonzero is
  true). Arrays as conditions are an error.

### Function registry

The registry is a **closed list**; extending it is a contract revision. Schema
version 1 defines, under the `core` capability:

| Name | Signature | Semantics |
|-|-|-|
| `dtype` | (string) → dtype token | See [Data types](#data-types). |
| `float` | (string \| number) → float | Decimal parse / cast. |
| `int` | (string \| number) → integer | Decimal integer parse / truncating cast. |
| `str` | (value) → string | Decimal formatting for numbers. |
| `len` | (string \| list \| array) → integer | Length / first-axis length. |
| `abs` | (number \| array) → same | Absolute value, elementwise on arrays. |
| `isnan` | (number \| null \| array) → bool \| array | True for NaN; null (scalar NaN) is NaN. |
| `isfinite` | (number \| array) → bool \| array | False for NaN and infinities. |
| `logical_not` | (bool \| array) → bool \| array | Elementwise negation. |
| `transpose` | (array) → array | Axis reversal. |
| `flip` | (array, integer) → array | Reverse along the given axis. |
| `reshape` | (array, tuple) → array | C-order reshape. |
| `unpackbits` | (uint8 array, integer) → array | Unpack into the given number of bits, least-significant bit first (ISO 5436-2 validity masks). |
| `strip` | (string) → string | Strip leading and trailing whitespace. |
| `split` | (string, string) → list | Split at every occurrence of the separator. |
| `to_map` | (list, string, string) → mapping | Build a mapping from a list of records: record[key-field] → record[value-field]. |
| `get` | (mapping, string, value) → value | Lookup with default. |
| `merge` | (mapping, mapping) → mapping | Right side wins on key collisions. |
| `omit` | (mapping, key) → mapping | Mapping without the given key. |
| `pluck` | (list, string) → list | Values of the key from those records that have it. |
| `values_with_prefix` | (mapping, string) → list | Values of all keys starting with the prefix, in mapping order. |
| `match_records` | (list, key, value) → list | Records whose value under the key (a field name or a path of names) equals the value. |
| `match_records_containing` | (list, key, string) → list | Records whose value under the key contains the string. |
| `index_by` | (list, string) → mapping | Map each record's value under the key to the record (records missing the key are skipped). |
| `parse_xml` | (string) → mapping | Parse an XML document into nested mappings (attributes as `@name` keys, text content as `#text`). |
| `unit_conversion_factor` | (string, string) → float \| null | Length-unit ratio from/to; null if either unit is unknown. Unit table: Gm Mm km m mm µm um nm Å A pm fm. |
| `mangle_length_unit` | (string[, bool]) → string | Normalize a vendor length-unit string to the unit table (e.g. `um` → `µm`). If the optional flag is false, `A` is the ampere and is kept. |
| `is_length_unit` | (string[, bool]) → bool | True if the string is in the length unit table. If the optional flag is false, `A` (the ampere) is not a length unit. |
| `parse_datetime` | (string) → datetime | Vendor date string → ISO-8601 (see [Value model](#value-model)). |
| `make_datetime` | (6–7 × integer) → datetime \| null | From year, month, day, hour, minute, second; the optional seventh argument is a UTC offset in minutes (result is timezone-aware); null if invalid (e.g. all-zero fields). |
| `from_timestamp` | (integer) → datetime | POSIX timestamp, as a naive datetime in the local time zone of the host. |

Datetime-producing functions are used only for values that end up in the
`info` section of channel bindings, either directly or as converters of
metadata fields that only `info` references. Engines that do not expose `info`
(see [Channel bindings](#channel-bindings)) *may* evaluate them to null.

Capability-gated functions (see [Capabilities](#capabilities)):

| Name | Capability | Semantics |
|-|-|-|
| `zstd_reader` | `zstd` | Wrap a stream in zstandard decompression. |
| `zlib_reader` | `zlib` | Wrap a stream in zlib decompression. |

### Data types

Array data types are denoted by strings of the form
`[<byte order>]<type><size>` where byte order is `<` (little-endian, default),
`>` (big-endian), `=` (host) or `|` (not applicable; single-byte types); type is
`i` (signed integer), `u` (unsigned integer) or `f` (IEEE-754 float); and size
is the item size in bytes (`i`/`u`: 1, 2, 4, 8; `f`: 4, 8). Examples: `<i4`,
`>f8`, `u1`, `|u1`.

Binary structure field formats use the Python `struct` mini-language with
standard sizes, restricted to the codes `? B h H i I l L q Q f d s`, with the
following SurfaceTopography-specific codes:

| Code | Meaning |
|-|-|
| `b` | Raw bytes; the repeat count is the number of bytes. (This is **not** `struct`'s signed char.) The value is a bytes object. |
| `s` | String, decoded as latin-1. |
| `u` | String, decoded as UTF-8. |
| `U` | String, decoded as UTF-16; the (mandatory) repeat count is the number of 16-bit code units. A byte-order mark selects the byte order; without one, the string is little-endian. |
| `t` / `T` | Pascal string (ASCII) with a 16-/32-bit length prefix in the structure's byte order; leading and trailing NUL characters are removed. Written without repeat count or byte-order prefix. |

Formats carry optional decimal repeat counts and an optional per-field `<`/`>`
byte-order prefix.

- **Byte order**: Structure-level byte order is `<`, `>` or `=` only. The native byte order
  `@` *must not* appear in exported documents; the exporter rewrites it to
  `=`. (Rationale: the Python engine unpacks fields individually, so `@`
  never introduces alignment padding and is equivalent to `=` — but an
  engine unpacking whole structures would insert padding and disagree.) The
  rewrite is only size-preserving for standard-size codes; the exporter
  refuses to serialize an `@` structure containing a native-size code
  (`l L n N P`, whose widths are platform-dependent) instead of silently
  changing the field width.

## Layout nodes

Layout nodes form the parse tree executed against the stream. Each is a JSON
object with a `type` key naming the node and an optional `name` under which its
result is stored in the enclosing context (an absent or null name merges the
result into the enclosing context). Engines *must* reject unknown `type`
values.

The per-node field schemas are those produced by the reference
implementation's serializer (`SurfaceTopography.IO.description`) and are
frozen with schema version 1; the reference serializer's output *is* the
normative schema. Several nodes accept an optional `context_mapper`
expression that is applied to their result (`CompoundLayout`, `SizedChunk`)
or to the context before the branch is selected (`If`). The catalog:

Core nodes (capability `core`):

- `CompoundLayout`: Ordered sequence of child nodes (`structures`) sharing one context; each
  child sees the results of the previous children and the enclosing context
  as `__parent__`.

- `BinaryStructure`: Sequence of scalar fields with a structure-level `byte_order`. Each field
  is `{"name": <string|null>, "format": <struct format>, "hooks": [<hook>,
  ...]}` where a hook is either
  `{"validate": {"value": <literal or boolean expression>, "error":
  <taxonomy name>}}` or `{"convert": <expression>}`, applied in order. A
  literal validate value is an equality check; an expression is evaluated
  with the field value bound to `val`. A null field name discards the value
  after hooks run.

- `BinaryArray`: Bulk data. Shape (tuple expression), dtype (expression), optional
  `conversion` expression and optional `mask` expression. Produces a *lazy
  array handle* (see [Two-phase reading](#two-phase-reading)), never inline
  data.

- `RawBuffer` / `TextBuffer`: Sized byte/text blocks; a null `size` means the payload size of the
  enclosing `TLVContainer` entry. A `RawBuffer` with `lazy` true produces a
  handle; with `lazy` false it stores `{"_raw": <bytes>, "_size":
  <integer>}`. A `TextBuffer` is decoded with its `encoding` (undecodable
  bytes replaced) and has leading and trailing NUL characters removed.

- `Skip`: Advance the stream by `size` bytes without storing; a null `size` means the
  payload size of the enclosing `TLVContainer` entry.

- `Seek`: Move the stream to an absolute position (`offset` expression), for formats
  whose header carries absolute offsets to data regions.

- `If`: Condition/layout pairs (`branches`) plus optional `default`; the layout of
  the first true condition is executed. Without a true condition and without
  a default, the result is empty.

- `For`: Repeat a child layout `count` times, collecting a list.

- `While` **[draft]**: Repeat a sequence of layouts and conditions (`body`) while the conditions
  hold, collecting a list.

- `Switch`: Select a child layout from a list of `cases` keyed by the value of the
  `key` expression (tag-driven formats); optional `default`. A key matching
  no case and no default is a corrupt-file error.

- `SizedChunk`: Execute a child layout inside a window of `size` bytes. With `mode`
  `read-once`, the child must consume exactly the window; with
  `skip-missing`, the stream is advanced to the window end if the child
  consumed less; with `loop`, the child is repeated until the window is
  consumed and the results form a list. Consuming more than the window is an
  error in all modes.

- `Check`: Validates a condition against the context without reading from the
  stream; a false result raises the error named by the `error` taxonomy
  entry (with `comment` as message). Use for consistency checks between
  previously parsed values.

- `ForEach`: Repeats a structure for each element of a previously parsed list
  (`items`, e.g. the payloads of a block directory). The structure's context
  contains the current element as `item` and its index as `item_index`; the
  per-element results form a list.

- `Let`: Stores expression `values` into the context without reading from the
  stream, e.g. to make the `item` of an enclosing `ForEach` part of a
  structure's result.

- `TLVContainer`: Tag-length-value sequences (`tag_format`, `size_format`) with a tag→layout
  mapping (`tags`). An entry's layout is a layout node, `"text"` (ASCII text
  with leading and trailing NUL characters removed) or null (raw bytes). The
  number of entries is bounded by `count` and/or `container_size`; an
  optional `entry_prefix_format` is skipped before each entry. The context
  passed to each entry's layout contains the previously parsed *named*
  entries of the container, so later entries can reference earlier ones
  (e.g. for data-dependent array shapes). An optional `default` layout
  handles tags missing from the mapping (e.g. `Skip` to skip unknown
  blocks); without it, unmapped tags store their raw bytes. Within an
  entry's layout, the entry's payload size is available as the context value
  `_block_size`. Repeated tags collect their entries in a list. With
  `store_by_name` true, entries are additionally stored under their layout's
  name. With `hex_tag_keys` true, entries are stored under hexadecimal
  string keys (`"0x66"`) instead of integer tags — required when the
  entries end up in reported metadata, which must survive a JSON round trip.

Capability-gated nodes:

| Node | Capability | Semantics |
|-|-|-|
| `ZipContainer` | `zip` | Parse named members of a ZIP archive, each with its own layout; optional `stream_filter` expression (e.g. zstd) applied per member; members may be `optional`. Member names may be expressions over the previously parsed members (e.g. data files named within an XML index); a member entry may loop over a list (`foreach`, one member per element, with `item`/`item_index` in context); entries without a `member` key are stream-less nodes (`Check`, `Let`) executed in order. The `mismatch_error` taxonomy name (default `corrupt_file`) is raised when the stream is not a ZIP archive or a required member is missing. |
| `XMLStructure` | `xml` | Parse an XML document into a nested mapping of element names; leaf elements become their text content, attributes are ignored; per-tag `converters` expressions. |
| `ZlibBlockChain` | `zlib` | Chained zlib blocks with prefix headers (`prefix_format`, the last field being the compressed size); the first block is located by scanning for a zlib header whose block decompresses to at least `min_decompressed_size` bytes. Blocks are decompressed in phase A. |
| `TextLine` | `text` | One text line, stripped of leading and trailing whitespace. |
| `TextHeader` | `text` | Line-oriented `key <sep> value` header: configurable separator or fixed-width key column, comment prefixes, INI-style or key-triggered sections, delimited by a byte count, terminator line, stop key or EOF; per-key converter expressions. |
| `TextMatrix` | `text` | Whitespace-separated number matrix of a given shape; bad-value tokens parse as NaN; optional conversion expression. Values are materialized in phase A (a text region cannot be sized without scanning it). |
| `TIFFContainer` | `tiff` | Parse the stream as a TIFF file (baseline TIFF 6.0 with the compression schemes of the corpus); a stream that is not a TIFF file is a format mismatch. Reports a `pages` list; each page holds `index`, `shape`, a `tags` mapping (standard tag names, names from `tag_names` for private tags, hexadecimal codes otherwise; enums as [name, value] pairs; byte blobs and arrays omitted), an `image` handle to the page raster (`image_conversion` expression applied on materialization), repeating private-tag blocks collected into record lists per `tag_groups` (first code, stride, offset names), and the results of `tag_layouts` (layouts executed against the byte value of a private tag, in declaration order, with lazy readers re-reading the tag on demand). |

## Two-phase reading

Engines *must* implement a two-phase protocol:

- **Phase A — metadata**: Execute the layout. Scalar fields, strings and small buffers are
  materialized into the context. `BinaryArray` nodes record a **handle** —
  source (root stream or container member), offset within that source,
  shape, dtype, conversion and mask expressions — and *must not* retain bulk
  data. Skipping over bulk data by forward seeking (or reading-and-discarding
  on non-seekable member streams) is the expected cost of this phase.

- **Phase B — data**: Materialize the handles referenced by a channel's bindings, applying the
  recorded conversion and mask expressions. Re-opening the source (including
  re-opening archive members and re-applying decompression filters) is the
  engine's responsibility.

The root stream *must* be seekable. Container member streams need only support
reading and forward seeking.

## Channel bindings

The `channels` section maps parsed metadata to the neutral result model. Every
field value is either a JSON literal or an expression over the metadata
context:

```json
{
  "name": "default",
  "dim": 2,
  "nb_grid_pts": <expr → [nx, ny]>,
  "physical_sizes": <expr → [sx, sy]>,
  "unit": "m",
  "height_scale_factor": <expr | number | null>,
  "periodic": false,
  "uniform": true,
  "info": { "<key>": <expr | literal>, ... },
  "data": <ctx path to an array handle>,
  "mask": {
    "source": <ctx path to an array handle>,
    "rule": <expr, val bound to the source array, → boolean array>
  }
}
```

- Channels are numbered in the order in which they are emitted, starting at
  zero.
- Physical heights are `raw value × height_scale_factor` in `unit`. A null or
  absent `height_scale_factor` means heights are returned unscaled. An absent
  `periodic` means non-periodic.
- **Array convention**: data arrays are indexed `(x, y)` — shape `(nx, ny)` with
  the first index running along the physical x direction. Descriptions are
  responsible for any crop/transpose needed to satisfy this (via conversion
  expressions).
- Channels with `dim` 1 are line scans: `nb_grid_pts` and `physical_sizes` have
  a single entry and the data array has shape `(nx,)`.
- Channels that do not contain heights declare `data_kind` (one of `height`,
  `voltage`, `current`, `phase`, `amplitude`, `error`, `deflection`,
  `friction`, `other`; default `height`) and may declare `data_unit`, the unit
  of the data values (`unit` remains the lateral unit).
- `mask` is optional; where present, true marks an **undefined** pixel. The
  rule sees the source array as `val` and the metadata as context. Engines map
  undefined pixels to their native representation (masked arrays in numpy, NaN
  in Eigen fields). Example rules: `(V & 3) != 0` (ZON validity codes),
  `V == 1000001.0` (PLU sentinel, with `source` = `data`). If the mask `source`
  path does not resolve — e.g. an optional archive member is absent from a
  particular file — the channel is unmasked.
- `info` values must be JSON-representable (see [Value model](#value-model) for
  datetimes). Entries that evaluate to null are omitted, except within
  `raw_metadata`, which is reported verbatim. Engines other than the reference
  implementation *may* not expose `info`; the conformance goldens do not cover
  it.
- Formats with a variable number of channels (e.g. PLU layers) declare a
  `foreach` entry: an expression evaluating to a context list. One channel is
  emitted per element; within all other expressions of the binding, the
  element is available as the context key `item` and its zero-based index as
  `item_index`.
- An optional `where` expression filters the emitted channels: when it
  evaluates falsy for a `foreach` element (or for the whole binding without
  `foreach`), no channel is emitted. Example: JPK emits channels only for TIFF
  pages whose default slot carries a length unit.
- An optional `checks` list validates each emitted channel:
  `{"condition": <expr>, "error": <error taxonomy name>, "message": <str>}` — a
  falsy condition raises the named error (default `corrupt_file`). Use this
  for per-channel constraints that a layout-level `Check` cannot express (e.g.
  an unsupported scaling type on a height channel).

## Capabilities

Schema version 1 defines: `core`, `zlib`, `zstd`, `zip`, `xml`, `text`, `tiff`.
A description requires `core` plus the capability of every capability-gated
layout node and registry function it contains.

An engine built without a capability *must* report a description requiring it
as *known but unsupported* (distinguishable from both "not this format" and
"corrupt file") using the `capabilities` list alone.

## Conformance goldens

For each fixture in the shared corpus, a golden document records the expected
parse result, generated by the reference implementation (SurfaceTopography):

```json
{
  "schema_version": 1,
  "format": "zon",
  "fixture": "zon-1.zon",
  "channels": [
    {
      "index": 0,
      "name": "default",
      "nb_grid_pts": [1779, 2588],
      "physical_sizes": [0.004378213553248556, 0.006369205551325049],
      "unit": "m",
      "nb_undefined": 15620,
      "probe_pixels": [
        {"pos": [0, 0], "value": null},
        {"pos": [889, 1294], "value": 7.619e-05},
        ...
      ],
      "masked_mean": 7.896083483856795e-05,
      "masked_rms": 8.382803508512573e-06
    }
  ]
}
```

- Probe values are physical heights in `unit`; null means the pixel is
  undefined. Probes are the four corners $(0,0)$, $(n_x-1,0)$, $(0,n_y-1)$,
  $(n_x-1,n_y-1)$, the center $(\lfloor n_x/2\rfloor, \lfloor n_y/2\rfloor)$
  and the interior point $(\lfloor n_x/3\rfloor, \lfloor 2n_y/3\rfloor)$.
- `masked_mean` is the mean and `masked_rms` the root-mean-square deviation
  from that mean (population standard deviation) of the physical heights over
  the defined pixels.
- Tolerances: probe pixels $10^{-12}$ relative (decoded values, exact up to
  float widening); `masked_mean`/`masked_rms` $10^{-9}$ relative
  (summation-order differences on multi-megapixel fixtures).
- Both engines run the same goldens; the goldens are regenerated only by the
  reference implementation.

## Error taxonomy

Engines *must* distinguish at least:

| Condition | Name in documents | Python / C++ mapping |
|-|-|-|
| Not this format (magic) | `format_mismatch` | `FileFormatMismatch` / `format_mismatch_error` |
| Corrupt or truncated file | `corrupt_file` | `CorruptFile` / `corrupt_file_error` |
| Valid file, unsupported feature variant | `unsupported_feature` | `UnsupportedFormatFeature` / `unsupported_feature_error` |
| Description requires a capability not built | — | (registry-level report) / `unsupported_capability_error` |
| Document schema too new, unknown node/function kind | — | `UnsupportedSchema` / `unsupported_schema_error` |

The "name in documents" is what `error` and `mismatch_error` fields of
validate hooks, `Check` nodes, `ZipContainer` nodes and channel `checks` carry.

Truncation *must* raise (never silently zero-fill), and engines *should*
sanity-bound array dimensions before allocation.

## Change policy

- Additions of node kinds, registry functions or capabilities increment
  `schema_version`.
- Engines reject documents newer than they implement; they *must not* guess.
- Semantic changes to existing nodes are not permitted; define a new node
  instead.
