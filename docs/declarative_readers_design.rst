Cross-language declarative file readers: design
================================================

:Status: Implemented (Phases 0-5; see coverage below)
:Date: 2026-07-30
:Scope: SurfaceTopography (Python) and SDSAlgorithms/libsdsio (C++)

Motivation
----------

SurfaceTopography contains readers for roughly forty topography file
formats. A growing subset of these is *declarative*: the file structure is
described by a tree of layout objects (``CompoundLayout``,
``BinaryStructure``, ``BinaryArray``, ``ZipContainer``, ...) that a generic
engine (``DeclarativeReaderBase``) executes. Much of the knowledge encoded
in these descriptions is hard-won reverse engineering (undocumented masks,
padding rules, magic constants) that exists nowhere else.

Standalone C++ applications built on SDSAlgorithms need the same readers.
The current approach in ``libsdsio`` exports the Python layout trees to
JSON and generates C++ code from them. This works for simple formats but
breaks down wherever the Python description contains a lambda: the JSON
contains an opaque placeholder, and the C++ generator compensates with
per-format special cases hard-coded into the generator itself, plus
hand-written per-format data-reading code that duplicates byte offsets,
data types, scale factors and undefined-data markers. Every such
duplication is a divergence waiting to happen.

The goal of this design is a **single, language-neutral source of format
knowledge**: a format description document that a Python engine and a C++
engine execute with identical results, verified by a shared conformance
corpus.

Goals
-----

1. One description per file format, containing *all* format knowledge:
   parse layout, semantic channel bindings (grid size, physical sizes,
   units, height scaling, undefined-data rules) and registration metadata
   (magic bytes, file extensions, MIME types).
2. Two engines — SurfaceTopography (Python) and libsdsio (C++) — that
   execute these descriptions with results equal within documented
   tolerances, enforced by shared golden tests.
3. Incremental migration: every phase leaves both codebases better off
   even if later phases never happen.

Non-goals
---------

* Covering *all* formats. Formats built on foreign container technologies
  (HDF5-based DATX, Gwyddion's own serialization, text formats) remain
  native code per language, or unsupported in C++.
* A hand-editable interchange syntax. The interchange JSON is generated,
  not authored (see below).
* Bit-identical floating-point results. The conformance tests use
  documented tolerances.

Architecture decisions
----------------------

Source of truth: Python authors, JSON is the contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format descriptions are *authored* as Python layout objects in
SurfaceTopography — this keeps the fast iteration loop for reverse
engineering (interactive exploration against real files, tests against the
example corpus) and readable descriptions (operator-overloaded
expressions, helper functions, comments). The *canonical contract* is the
exported JSON document: a versioned artifact that provably contains
everything, enforced by a CI gate requiring that

1. every declarative reader exports with **zero** opaque placeholders, and
2. a reader rehydrated *from the JSON* parses the entire example corpus
   identically to the authored one.

The JSON artifacts and conformance goldens are vendored into SDSAlgorithms
with a regeneration script and a staleness check in CI. A dedicated
format-description repository is a possible later refinement if a third
consumer appears.

Expressions instead of lambdas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-dependent quantities (array shapes, validation conditions,
conversions, conditional layouts) are expressed in a small, serializable
expression language (``SurfaceTopography.IO.expr``) instead of Python
lambdas. Expression objects are drop-in callable where the layout classes
accept lambdas, and serialize to a JSON AST that a foreign-language
interpreter evaluates. Named functions (``F.dtype``, ``F.transpose``,
``F.parse_datetime``, ...) come from a **closed registry** — this registry
is the exact list of primitives a foreign engine must provide, and
additions to it require a contract revision. See
:doc:`format_description_contract` for the normative definition.

Generic engines, descriptions as data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-format reader *code* dissolves into per-format *descriptions*:

* **Python**: ``DeclarativeReaderBase`` remains the single engine. A
  format contributes a layout, a declarative channel-binding section and
  registration metadata. Thin per-format classes remain as shims for
  backward-compatible imports and as escape hatches, but contain no
  logic.
* **C++**: the code generator is retired in favor of a runtime
  ``DeclarativeReader`` that ingests the JSON descriptions (embedded into
  the binary at build time, optionally loadable from disk). One engine,
  one expression evaluator, one format registry keyed by magic bytes.
  Interpretation costs nothing measurable: parsing time is dominated by
  bulk data reads, not dispatch.

The channel-binding layer is part of the description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experience from libsdsio shows that the semantic mapping — which parsed
field is the grid size, how physical sizes are computed, where the height
data lives, what marks an undefined pixel — is where hand-written C++
diverges first (hand-derived byte offsets, re-implemented mask rules).
This mapping therefore moves into the description as a declarative
``channels`` section using the same expression language. Bulk data access
is *lazy*: the metadata pass records array handles (offset, shape, dtype,
conversion); data is materialized only when a channel is read.

Capabilities
~~~~~~~~~~~~

Container and compression primitives (zip archives, zstd, zlib, XML)
each pull a dependency in C++. Descriptions declare the capabilities they
require; an engine built without a capability reports "format known,
capability not built" instead of failing obscurely.

Phased plan
-----------

Phase 0 — Contract specification (this document's companion)
    Normative spec of the description document: JSON schema, expression
    node catalog, function registry, semantic edge cases (byte order,
    division semantics, encodings, NaN policy), the neutral parse-result
    model, laziness semantics, capability names, conformance-golden
    schema, error taxonomy, versioning policy. See
    :doc:`format_description_contract`.

Phase 1 — Python: expressions to production (SurfaceTopography)
    Harden ``expr.py`` (bytes literals, ``DictExpr``, ``Switch`` layout
    primitive, unified ``(value, context)`` converter call sites).
    Complete ``to_dict()``/``from_dict()`` for all layout classes. Add the
    declarative channel-binding layer to ``DeclarativeReaderBase``. Port
    readers: the six formats already exported by libsdsio first (ZMG, TMD,
    AL3D, SUR, MetroPro, PLU), then ZON, MNT, and OIR as the hard-case
    validation. Land the round-trip equivalence CI gate.

Phase 2 — Contract artifacts
    Grow the export script into a proper exporter emitting format
    descriptions and machine-readable conformance goldens (probe pixels,
    masked statistics, undefined-pixel counts). Wire vendoring plus a
    staleness CI check in SDSAlgorithms.

Phase 3 — C++: interpreter completion (SDSAlgorithms)
    Expression evaluator over the AST (the ``ctx`` node maps onto the
    existing ``AttrDict::get_nested``; operators reuse its numeric
    coercion). Extend the runtime with ``If``, expression-driven
    validation, lazy array handles and mask materialization. Generic
    reader plus registry; delete the code generator and the hand-written
    per-format ``topography()`` methods; replace pasted golden values with
    a generic conformance runner.

Phase 4 — Containers and coverage
    Add capability primitives to the C++ engine as demand dictates (zlib
    for MNT, zip+zstd for ZON, XML), each behind a build option. Continue
    porting readers; maintain an explicit coverage list.

Phase 5 — Governance
    Documented workflow for new formats (author in Python → corpus tests →
    export → C++ conformance runs without writing C++) and the schema
    evolution policy.

Implementation status and coverage
----------------------------------

Fully declarative formats (exported to description documents, executed by
both engines, pinned by the conformance goldens): **ZMG, TMD, AL3D, SUR,
MetroPro, PLU, ZON**. Of these, ZON requires the ``zip``/``zstd``/``xml``
capabilities, which the C++ engine does not implement yet: it rejects the
description cleanly (``unsupported_capability_error``) while format
detection still works from the document.

Not yet declarative (Python-only, native readers): MNT (heuristic zlib
block chains), OIR (context restructuring pending a ``Switch``/``dict``
port), and the container/text formats named under Non-goals.

Key artifacts:

* Python engine: ``SurfaceTopography/IO/expr.py`` (expressions),
  ``description.py`` (serialization, generic reader factory),
  ``Reader.py`` (``DeclarativeReaderBase`` with ``_channel_bindings``,
  ``Switch``, ``Seek``), ``export.py`` (artifact exporter).
* CI gates: ``test/IO/test_description.py`` (zero-opaque, rehydrated
  generic-reader equivalence over the corpus, deterministic export),
  ``test/IO/test_expr.py``, ``test/IO/test_declarative_reader_serialization.py``.
* C++ engine: ``src/libsdsio/declarative_reader.{h,cpp}`` in
  SDSAlgorithms (expression evaluator, layout interpreter, lazy array
  handles, registry with embedded descriptions);
  ``tests/cpp/test_declarative_reader.cpp`` is the golden conformance
  runner. The former code generator (``generate_from_json.py``), exported
  layout JSONs and hand-written per-format readers are retired.
* Contract artifacts: vendored in SDSAlgorithms under
  ``src/libsdsio/descriptions/`` and ``tests/goldens/``, regenerated only
  by ``update_format_descriptions.py`` and pinned by
  ``tests/python/test_format_descriptions.py``.

Workflow for a new format
-------------------------

1. Reverse engineer and author the reader in SurfaceTopography as a
   ``DeclarativeReaderBase`` subclass: ``_file_layout`` and
   ``_channel_bindings`` built from expressions (no lambdas), ``_magic``
   for detection. Iterate against real files; add a fixture to
   ``test/file_format_examples`` and per-format tests.
2. Register the format in ``SurfaceTopography/IO/export.py`` and in
   ``test/IO/test_description.py``. The CI gate now enforces that the
   description is complete (zero opaque hooks) and that the rehydrated
   generic reader reproduces the authored one.
3. In SDSAlgorithms, copy the fixture into ``tests/datafiles`` and run
   ``python update_format_descriptions.py``; commit the regenerated
   descriptions and goldens. The C++ conformance runner picks the new
   golden up automatically — no C++ code is written unless the format
   needs a new capability or layout primitive.
4. If a new expression function or layout node is unavoidable, revise the
   contract (bump the schema version per the change policy) and implement
   the primitive in both engines in the same change set.

Risks
-----

Registry and expression creep
    The function registry must remain a closed, reviewed list; otherwise
    the expression language slowly becomes a bad general-purpose language
    (the DFDL failure mode). Formats that cannot be expressed with the
    existing primitives are candidates for native readers, not for
    ad-hoc registry growth.

Semantic divergence at the edges
    Byte order and alignment, floor vs. truncating division, string
    encodings, NaN handling. Mitigation: these are specified normatively
    in the contract, and the conformance goldens include masked
    statistics over full arrays, not only probe pixels.

Hard formats
    OIR-class formats (nested TLV, tag-driven layout switching) test the
    limits of the declarative model. If a format does not go fully
    declarative in Phase 1, the fallback — native reader on both sides —
    is decided consciously there.

Prior art
---------

* `Kaitai Struct <https://kaitai.io/>`_ — YAML-based binary format DSL
  compiled to parsers in many languages; closest existing system, and the
  design reference for the expression language.
* DFDL (Data Format Description Language) — the standards-body take;
  cautionary example of expression-language growth.
* `Wuffs <https://github.com/google/wuffs>`_ — memory-safe generated
  parsers for untrusted inputs; motivates generated/interpreted parsing
  with systematic bounds checks over hand-written C++.
