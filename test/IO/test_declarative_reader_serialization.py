#
# Copyright 2026 Lars Pastewka
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
Tests for layout serialization per the format description contract
(docs/format_description_contract.rst): contract-style JSON documents,
opaque markers for non-expression callables, and rehydration via
`layout_from_dict`.
"""

import io
import json
import struct

import pytest

from SurfaceTopography.Exceptions import (
    CorruptFile,
    FileFormatMismatch,
    UnsupportedSchema,
)
from SurfaceTopography.IO.binary import BinaryStructure, Validate
from SurfaceTopography.IO.description import (
    capabilities,
    count_opaque,
    layout_from_dict,
    layout_to_dict,
)
from SurfaceTopography.IO.expr import C, F, Tup, V
from SurfaceTopography.IO.Reader import CompoundLayout, For, If, Skip, Switch


def roundtrip(layout):
    """Serialize, push through JSON, rehydrate, serialize again."""
    d = layout_to_dict(layout)
    rehydrated = layout_from_dict(json.loads(json.dumps(d)))
    assert layout_to_dict(rehydrated) == d
    return d, rehydrated


def test_binary_structure_contract_format():
    layout = BinaryStructure(
        [
            ("magic", "4s", Validate("KPK0", FileFormatMismatch)),
            ("version", "I", Validate(V > 0, CorruptFile)),
            ("scale", "8s", F.float(V)),
            (None, "16s"),
        ],
        byte_order="<",
        name="header",
    )
    d, rehydrated = roundtrip(layout)
    assert d["type"] == "BinaryStructure"
    assert d["name"] == "header"
    assert d["byte_order"] == "<"
    assert count_opaque(d) == 0
    assert d["fields"][0]["hooks"][0]["validate"]["error"] == "format_mismatch"
    assert d["fields"][1]["hooks"][0]["validate"]["error"] == "corrupt_file"
    assert "hooks" not in d["fields"][3]

    # The rehydrated layout parses a real stream identically
    stream = struct.pack("<4sI8s16s", b"KPK0", 3, b"1.5", b"")
    parsed = rehydrated.from_stream(io.BytesIO(stream), {})
    assert parsed["header"]["magic"] == "KPK0"
    assert parsed["header"]["version"] == 3
    assert parsed["header"]["scale"] == 1.5

    # ... and validators fire with the right exception
    bad_magic = struct.pack("<4sI8s16s", b"XXXX", 3, b"1.5", b"")
    with pytest.raises(FileFormatMismatch):
        rehydrated.from_stream(io.BytesIO(bad_magic), {})
    bad_version = struct.pack("<4sI8s16s", b"KPK0", 0, b"1.5", b"")
    with pytest.raises(CorruptFile):
        rehydrated.from_stream(io.BytesIO(bad_version), {})


def test_native_byte_order_is_normalized():
    # `@` must not appear in exported documents; it is equivalent to `=`
    # because fields are unpacked individually
    layout = BinaryStructure([("a", "I")], byte_order="@", name="s")
    d = layout_to_dict(layout)
    assert d["byte_order"] == "="


def test_lambdas_serialize_to_opaque_markers():
    layout = CompoundLayout(
        [
            BinaryStructure(
                [("a", "I", Validate(lambda x, c: x > 0, CorruptFile))],
                name="s",
            ),
        ],
        context_mapper=lambda context: context,
    )
    d = layout_to_dict(layout)
    assert count_opaque(d) == 2  # the validator and the context mapper
    with pytest.raises(UnsupportedSchema):
        layout_from_dict(d)


def test_if_roundtrip():
    layout = If(
        C.header.version == 1,
        BinaryStructure([("a", "I")], name="v1"),
        C.header.version == 2,
        BinaryStructure([("b", "I")], name="v2"),
        BinaryStructure([("c", "I")], name="fallback"),
    )
    d, rehydrated = roundtrip(layout)
    assert d["type"] == "If"
    assert len(d["branches"]) == 2
    assert d["default"]["name"] == "fallback"
    assert count_opaque(d) == 0

    from SurfaceTopography.IO.binary import AttrDict

    stream = io.BytesIO(struct.pack("=I", 42))
    context = AttrDict({"header": AttrDict({"version": 2})})
    assert rehydrated.from_stream(stream, context) == {"v2": {"b": 42}}


def test_switch_roundtrip_and_semantics():
    layout = Switch(
        C.block_type,
        {
            "alpha": BinaryStructure([("a", "I")], name="alpha"),
            "beta": BinaryStructure([("b", "I")], name="beta"),
        },
    )
    d, rehydrated = roundtrip(layout)
    assert d["type"] == "Switch"
    assert count_opaque(d) == 0

    from SurfaceTopography.IO.binary import AttrDict

    stream = io.BytesIO(struct.pack("=I", 7))
    parsed = rehydrated.from_stream(stream, AttrDict({"block_type": "beta"}))
    assert parsed == {"beta": {"b": 7}}

    # Unknown key without default raises CorruptFile
    with pytest.raises(CorruptFile):
        rehydrated.from_stream(
            io.BytesIO(struct.pack("=I", 7)), AttrDict({"block_type": "gamma"})
        )


def test_for_and_skip_roundtrip():
    layout = CompoundLayout(
        [
            BinaryStructure([("n", "I")], name="header"),
            Skip(4, comment="reserved"),
            For(
                C.header.n,
                BinaryStructure([("value", "I")]),
                name="entries",
            ),
        ]
    )
    d, rehydrated = roundtrip(layout)
    assert count_opaque(d) == 0

    stream = io.BytesIO(struct.pack("=IIIII", 3, 0xDEAD, 10, 20, 30))
    parsed = rehydrated.from_stream(stream, {})
    assert [e["value"] for e in parsed["entries"]] == [10, 20, 30]


def test_binary_array_roundtrip():
    import numpy as np

    from SurfaceTopography.IO.binary import BinaryArray

    layout = CompoundLayout(
        [
            BinaryStructure([("nx", "I"), ("ny", "I")], name="header"),
            BinaryArray(
                "data",
                Tup(C.header.ny, C.header.nx),
                F.dtype("<i2"),
                conversion_fun=F.transpose(V),
                mask_fun=V == 9999,
            ),
        ]
    )
    d, rehydrated = roundtrip(layout)
    assert count_opaque(d) == 0
    assert d["structures"][1]["type"] == "BinaryArray"

    payload = struct.pack("=II", 3, 2) + np.array(
        [[1, 2, 3], [4, 9999, 6]], dtype="<i2"
    ).tobytes()
    stream = io.BytesIO(payload)
    parsed = rehydrated.from_stream(stream, {})
    arr = parsed["data"](stream)
    assert arr.shape == (3, 2)
    assert arr[1, 1] is np.ma.masked
    assert arr[2, 1] == 6


def test_fixed_dtype_serializes_as_expression():
    import numpy as np

    from SurfaceTopography.IO.binary import BinaryArray

    # A plain np.dtype (not an expression) must serialize to F.dtype(...)
    layout = BinaryArray("data", (2, 2), np.dtype("<f4"))
    d, rehydrated = roundtrip(layout)
    assert d["dtype"] == {
        "kind": "call",
        "name": "dtype",
        "args": [{"kind": "lit", "value": "<f4"}],
    }


def test_unknown_node_type_raises():
    with pytest.raises(UnsupportedSchema):
        layout_from_dict({"type": "FancyNewNode"})
    with pytest.raises(UnsupportedSchema):
        layout_from_dict({"kind": "lit", "value": 1})


def test_capabilities_computation():
    layout = BinaryStructure([("a", "I")], name="s")
    assert capabilities(layout_to_dict(layout)) == ["core"]
