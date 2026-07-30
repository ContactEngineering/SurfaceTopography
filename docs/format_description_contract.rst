Format description contract
===========================

:Status: Implemented (schema version 1) — Python engine in
         SurfaceTopography, C++ engine in SDSAlgorithms/libsdsio
:Schema version: 1
:Date: 2026-07-30

This document is the normative contract between producers of format
description documents (the SurfaceTopography exporter) and the engines
that execute them (SurfaceTopography's ``DeclarativeReaderBase`` and
libsdsio's ``DeclarativeReader``). Where behavior is not specified here,
engines must not rely on it agreeing across implementations.

The key words *must*, *must not*, *should* and *may* are to be interpreted
as in RFC 2119.

Sections marked **[draft]** are expected to change before the contract is
frozen; everything else is intended to be stable.

Document structure
------------------

A format description is a single JSON document::

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
      "capabilities": ["core", "zip", "zstd", "xml"],
      "layout": { ...layout node... },
      "channels": [ ...channel bindings... ]
    }

``schema_version``
    Integer. Engines *must* refuse documents with a version greater than
    the one they implement, with a distinguishable "unsupported schema"
    error. Any change to this contract that an existing engine cannot
    safely ignore requires incrementing the version.

``format.magic``
    List of alternatives; each alternative matches if the file contains
    the given bytes (base64-encoded) at the given offset. Detection
    semantics: *yes* if any alternative matches; *no* if the probe buffer
    covers all alternatives and none match; *maybe* if the buffer is too
    short. An empty list means detection always answers *maybe* (trial
    parse required).

``capabilities``
    The complete list of capabilities the description requires (see
    `Capabilities`_). Engines *must* be able to decide support from this
    list alone, without walking the layout.

Value model
-----------

Values flowing through the parser context and expressions are:

* **null** — also the representation of scalar NaN read from a file
  (see below).
* **boolean**
* **integer** — signed 64-bit. Unsigned fields are widened; values
  outside the signed 64-bit range are outside this contract.
* **float** — IEEE-754 double precision. Fields of lower precision are
  widened on read.
* **string** — Unicode. Strings decoded from files have trailing NUL
  bytes removed and leading/trailing ASCII whitespace stripped.
* **bytes** — serialized as base64 where they appear in documents.
* **array** — n-dimensional, typed; produced only by array layout
  primitives and by expression operations on arrays.
* **mapping / list** — nested context structures produced by layout
  nodes.

Scalar NaN policy
    A scalar float field whose value is NaN *must* be stored in the
    context as null. (Rationale: NaN breaks equality of metadata
    dictionaries; this mirrors long-standing SurfaceTopography
    behavior.) Arrays keep their NaN payloads; undefined *data points*
    are represented by masks, never by NaN sentinel comparisons hidden in
    engine code.

Datetimes
    The neutral representation of a point in time is an ISO-8601 string
    with offset (e.g. ``"2021-03-24T13:49:17-04:00"``). The
    ``parse_datetime`` registry function normalizes vendor date strings to
    this form. Engines *may* expose a native datetime type to their host
    language, but the serialized form is the ISO string.

Expressions
-----------

Expressions are trees of nodes, each a JSON object with a ``kind`` key.
Engines *must* reject unknown ``kind`` values with a distinguishable
error. The Python authoring API lives in ``SurfaceTopography.IO.expr``.

Node catalog
~~~~~~~~~~~~

``{"kind": "lit", "value": <json>}``
    Literal. ``value`` is any JSON scalar or list of scalars.

``{"kind": "bytes", "value": "<base64>"}``
    Bytes literal.

``{"kind": "val"}``
    The value currently being processed (the field under validation or
    conversion, the array under conversion, the stream under filtering).

``{"kind": "ctx", "path": ["header", "nb_grid_pts_x"]}``
    Context reference; resolves the path segment by segment in the
    current parser context. The segment ``__parent__`` refers to the
    enclosing context. Resolution failure is an engine error (corrupt
    description or file).

``{"kind": "binop", "op": "<op>", "args": [<node>, <node>]}``
    Binary operation. Operators: ``+ - * / // % & | ^ << >> == != < <= >
    >= in``.

``{"kind": "unop", "op": "<op>", "arg": <node>}``
    Unary operation. Operators: ``neg``, ``not``.

``{"kind": "call", "name": "<fn>", "args": [<node>, ...]}``
    Invocation of a registry function (see `Function registry`_).
    Unknown names *must* produce a distinguishable "unsupported function"
    error.

``{"kind": "tuple", "items": [<node>, ...]}``
    Fixed-length sequence (e.g. an array shape).

``{"kind": "cond", "condition": <node>, "then": <node>, "otherwise": <node>}``
    Conditional with short-circuit evaluation: only the selected branch
    is evaluated.

``{"kind": "getitem", "base": <node>, "index": [<part>, ...], "tuple": <bool>}``
    Indexing and slicing. Each part is a node or
    ``{"kind": "slice", "bounds": [<start>, <stop>, <step>]}`` with each
    bound null or a node. ``tuple`` records whether the index was
    multi-axis.

``{"kind": "dict", "items": {"<key>": <node>, ...}}``
    Mapping with static string keys, for context restructuring. Computed
    keys are not part of schema version 1.

Operator semantics
~~~~~~~~~~~~~~~~~~

These follow Python semantics; C++ implementations must reproduce them:

* ``/`` is true division and always yields a float.
* ``//`` is **floor** division and ``%`` follows the sign of the divisor
  (Python semantics). Note that C++ ``/`` on integers truncates toward
  zero and ``%`` follows the dividend — a C++ engine *must not* map these
  operators directly.
* ``==``/``!=`` compare by value; comparing values of unrelated types is
  unequal rather than an error. Ordering comparisons are defined for
  number–number and string–string (Unicode code point order) operands
  only.
* ``in`` tests membership of a scalar in a list.
* Bitwise operators require integers (or booleans, treated as 0/1).
* Arithmetic, comparison and bitwise operators applied to arrays operate
  elementwise with numpy broadcasting semantics; ``getitem`` follows
  numpy/Python indexing including negative indices and slice clamping.
  Applied to a mapping, a string index performs key lookup (so context
  paths can continue past an index, e.g. ``entries[0].prefix``).
* The condition of ``cond`` must evaluate to a boolean or integer
  (nonzero is true). Arrays as conditions are an error.

Function registry
~~~~~~~~~~~~~~~~~

The registry is a **closed list**; extending it is a contract revision.
Schema version 1 defines, under the ``core`` capability:

====================== ================================== =====================
Name                   Signature                          Semantics
====================== ================================== =====================
dtype                  (string) → dtype token             See `Data types`_.
float                  (string|number) → float            Decimal parse / cast.
int                    (string|number) → integer          Truncating cast.
str                    (value) → string                   Decimal formatting
                                                          for numbers.
len                    (string|list|array) → integer      Length / first-axis
                                                          length.
abs                    (number|array) → same              Absolute value,
                                                          elementwise on
                                                          arrays.
isnan                  (number|null|array) → bool|array   True for NaN; null
                                                          (scalar NaN) is
                                                          NaN.
transpose              (array) → array                    Axis reversal.
strip                  (string) → string                  Strip ASCII
                                                          whitespace.
to_map                 (list, string, string) → mapping   Build a mapping from
                                                          a list of records:
                                                          record[key-field] →
                                                          record[value-field].
get                    (mapping, string, value) → value   Lookup with default.
merge                  (mapping, mapping) → mapping       Right side wins on
                                                          key collisions.
unit_conversion_factor (string, string) → float|null      Length-unit ratio
                                                          from/to; null if
                                                          either unit is
                                                          unknown. Unit table:
                                                          Gm Mm km m mm µm um
                                                          nm Å A pm fm.
parse_datetime         (string) → datetime                Vendor date string →
                                                          ISO-8601 (see Value
                                                          model).
make_datetime          (6 × integer) → datetime|null      From year, month,
                                                          day, hour, minute,
                                                          second; null if
                                                          invalid (e.g.
                                                          all-zero fields).
from_timestamp         (integer) → datetime               POSIX timestamp, in
                                                          local time (historic
                                                          reader behavior).
====================== ================================== =====================

Datetime-producing functions appear only inside the ``info`` section of
channel bindings. Engines that do not expose ``info`` (see `Channel
bindings`_) *may* evaluate them to null.

Capability-gated functions (see `Capabilities`_):

============== =========== ============================================
Name           Capability  Semantics
============== =========== ============================================
zstd_reader    zstd        Wrap a stream in zstandard decompression.
zlib_reader    zlib        Wrap a stream in zlib decompression.
============== =========== ============================================

Data types
~~~~~~~~~~

Array data types are denoted by strings of the form
``[<byte order>]<type><size>`` where byte order is ``<`` (little-endian,
default), ``>`` (big-endian) or ``=`` (host); type is ``i`` (signed
integer), ``u`` (unsigned integer) or ``f`` (IEEE-754 float); and size is
the item size in bytes (``i``/``u``: 1, 2, 4, 8; ``f``: 4, 8). Examples:
``<i4``, ``>f8``, ``u1``.

Binary structure field formats use the Python ``struct`` mini-language
restricted to the codes ``b B h H i I q Q f d s`` plus the
SurfaceTopography extensions ``u`` (UTF-8 string), ``U`` (UTF-16 string),
``t``/``T`` (Pascal strings with 16/32-bit length), with optional decimal
repeat counts and optional per-field ``<``/``>`` prefix.

Byte order
    Structure-level byte order is ``<``, ``>`` or ``=`` only. The native
    byte order ``@`` *must not* appear in exported documents; the
    exporter rewrites it to ``=``. (Rationale: the Python engine unpacks
    fields individually, so ``@`` never introduces alignment padding and
    is equivalent to ``=`` — but a whole-struct reimplementation would
    disagree. Discovered the hard way in libsdsio.)

Layout nodes
------------

Layout nodes form the parse tree executed against the stream. Each is a
JSON object with a ``type`` key naming the node and an optional ``name``
under which its result is stored in the enclosing context (an absent or
null name merges the result into the enclosing context). Engines *must*
reject unknown ``type`` values.

The per-node field schemas are those produced by the reference
implementation's serializer (``SurfaceTopography.IO.description``) and are
frozen with schema version 1; the reference serializer's output *is* the
normative schema. The catalog:

Core nodes (capability ``core``):

``CompoundLayout``
    Ordered sequence of child nodes sharing one context.

``BinaryStructure``
    Sequence of scalar fields. Each field is
    ``{"name": <string|null>, "format": <struct format>,
    "hooks": [<hook>, ...]}`` where a hook is either
    ``{"validate": {"value": <literal or boolean expression>,
    "error": <taxonomy name>}}`` or ``{"convert": <expression>}``, applied
    in order. A literal validate value is an equality check; an expression
    is evaluated with the field value bound to ``val``. A null field name
    discards the value after hooks run.

``BinaryArray``
    Bulk data. Shape (tuple expression), dtype (expression), optional
    conversion expression and optional mask expression. Produces a *lazy
    array handle* (see `Two-phase reading`_), never inline data.

``RawBuffer`` / ``TextBuffer``
    Sized byte/text blocks; text decoding per the Value model.

``Skip``
    Advance the stream without storing.

``Seek``
    Move the stream to an absolute position (offset expression), for
    formats whose header carries absolute offsets to data regions.

``If``
    Condition/branch pairs plus optional default; selects a child layout.

``For``
    Repeat a child layout *n* times (count expression), collecting a
    list.

``While`` **[draft]**
    Repeat while a condition holds.

``Switch``
    Select a child layout from a list of cases keyed by a context value
    (tag-driven formats such as OIR); optional default. A key matching no
    case and no default is a corrupt-file error.

``SizedChunk``
    Execute a child layout inside a size-bounded window; the stream
    position afterwards is the window end regardless of how much the
    child consumed.

``TLVContainer``
    Tag-length-value sequences with a tag→layout mapping.

Capability-gated nodes:

=============== =========== =============================================
Node            Capability  Semantics
=============== =========== =============================================
ZipContainer    zip         Parse named members of a ZIP archive, each
                            with its own layout; optional stream-filter
                            expression (e.g. zstd) applied per member;
                            members may be optional.
XMLStructure    xml         Parse an XML document into a nested mapping;
                            per-tag converter expressions.
ZlibBlockChain  zlib        Chained zlib blocks with prefix headers
                            (MNT-style).
=============== =========== =============================================

Two-phase reading
-----------------

Engines *must* implement a two-phase protocol:

Phase A — metadata
    Execute the layout. Scalar fields, strings and small buffers are
    materialized into the context. ``BinaryArray`` nodes record a
    **handle** — source (root stream or container member), offset within
    that source, shape, dtype, conversion and mask expressions — and
    *must not* retain bulk data. Skipping over bulk data by forward
    seeking (or reading-and-discarding on non-seekable member streams) is
    the expected cost of this phase.

Phase B — data
    Materialize the handles referenced by a channel's bindings, applying
    the recorded conversion and mask expressions. Re-opening the source
    (including re-opening archive members and re-applying decompression
    filters) is the engine's responsibility.

The root stream *must* be seekable. Container member streams need only
support reading and forward seeking.

Channel bindings
----------------

The ``channels`` section maps parsed metadata to the neutral result
model. Every field value is either a JSON literal or an expression over
the metadata context::

    {
      "index": 0,
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

* Physical heights are ``raw value × height_scale_factor`` in ``unit``.
  A null ``height_scale_factor`` means heights are returned unscaled.
* **Array convention**: data arrays are indexed ``(x, y)`` — shape
  ``(nx, ny)`` with the first index running along the physical x
  direction. Descriptions are responsible for any crop/transpose needed
  to satisfy this (via conversion expressions).
* ``mask`` is optional; where present, true marks an **undefined** pixel.
  Engines map undefined pixels to their native representation (masked
  arrays in numpy, NaN in Eigen fields). Example rules:
  ``(V & 3) != 0`` (ZON validity codes), ``V == 1000001.0`` (PLU
  sentinel, with ``source`` = ``data``). If the mask ``source`` path does
  not resolve — e.g. an optional archive member is absent from a
  particular file — the channel is unmasked.
* ``info`` values must be JSON-representable (see Value model for
  datetimes). Engines other than the reference implementation *may* not
  expose ``info``; the conformance goldens do not cover it.
* Formats with a variable number of channels (e.g. PLU layers) declare a
  ``foreach`` entry: an expression evaluating to a context list. One
  channel is emitted per element; within all other expressions of the
  binding, the element is available as the context key ``item`` and its
  zero-based index as ``item_index``.

Capabilities
------------

v0.1 defines: ``core``, ``zlib``, ``zstd``, ``zip``, ``xml``.

An engine built without a capability *must* report a description
requiring it as *known but unsupported* (distinguishable from both "not
this format" and "corrupt file") using the ``capabilities`` list alone.

Conformance goldens
-------------------

For each fixture in the shared corpus, a golden document records the
expected parse result, generated by the reference implementation
(SurfaceTopography)::

    {
      "schema_version": 1,
      "format": "zon",
      "fixture": "zon-1.zon",
      "channels": [
        {
          "index": 0,
          "nb_grid_pts": [1779, 2588],
          "physical_sizes": [0.004378, 0.006369],
          "unit": "m",
          "nb_undefined": 15620,
          "probe_pixels": [
            {"pos": [0, 0], "value": null},
            {"pos": [10, 5], "value": 8.47e-05}
          ],
          "masked_mean": ...,
          "masked_rms": ...
        }
      ]
    }

* Probe values are physical heights in ``unit``; null means the pixel is
  undefined. Probes cover the four corners, the center and interior
  points.
* Tolerances: probe pixels 1e-12 relative (decoded values, exact up to
  float widening); ``masked_mean``/``masked_rms`` 1e-9 relative
  (summation-order differences on multi-megapixel fixtures).
* Both engines run the same goldens; the goldens are regenerated only by
  the reference implementation.

Error taxonomy
--------------

Engines *must* distinguish at least:

=========================== ==================================================
Condition                   Python / C++ mapping
=========================== ==================================================
Not this format (magic)     ``FileFormatMismatch`` / ``format_mismatch_error``
Corrupt or truncated file   ``CorruptFile`` / ``corrupt_file_error``
Valid file, unsupported
feature variant             ``UnsupportedFormatFeature`` /
                            ``unsupported_feature_error``
Description requires a
capability not built        (registry-level report) /
                            ``unsupported_capability_error``
Document schema too new,
unknown node/function kind  ``UnsupportedSchema`` (new) /
                            ``unsupported_schema_error``
=========================== ==================================================

Truncation *must* raise (never silently zero-fill), and engines *should*
sanity-bound array dimensions before allocation.

Change policy
-------------

* Additions of node kinds, registry functions or capabilities increment
  ``schema_version``.
* Engines reject documents newer than they implement; they *must not*
  guess.
* Semantic changes to existing nodes are not permitted; define a new
  node instead.
