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
Tests for the serializable expression mini-language (prototype), including
lambda-free re-implementations of the ZMG and ZON file layouts that are
checked for equivalence against the production readers.
"""

import json
import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.Exceptions import (
    CorruptFile,
    FileFormatMismatch,
    UnsupportedFormatFeature,
)
from SurfaceTopography.IO.binary import (
    AttrDict,
    BinaryArray,
    BinaryStructure,
    RawBuffer,
    Validate,
    XMLStructure,
    ZipContainer,
)
from SurfaceTopography.IO.expr import C, Cond, F, Tup, V, from_dict, register_function
from SurfaceTopography.IO.Reader import CompoundLayout
from SurfaceTopography.IO.ZMG import ZMGReader
from SurfaceTopography.IO.ZON import ZONReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


###
# Expression language unit tests
###


def test_context_paths_and_arithmetic():
    context = AttrDict(
        {"header": AttrDict({"nb_grid_pts_y": 3, "row_bytes": 18})}
    )
    assert (C.header.row_bytes // 4)(context) == 4
    assert (C.header.nb_grid_pts_y * 2 + 1)(context) == 7
    assert (16 - C.header.nb_grid_pts_y)(context) == 13  # reflected operand
    assert Tup(C.header.nb_grid_pts_y, C.header.row_bytes // 2)(context) == (3, 9)


def test_value_comparisons_and_membership():
    assert (V > 0)(5, {})
    assert not (V > 0)(-1, {})
    assert V.isin("KPK0", "KPK1")("KPK1", {})
    assert not V.isin("KPK0", "KPK1")("GARBAGE", {})
    assert ((V & 0x03) == 0)(0x80, {})
    assert not ((V & 0x03) == 0)(0x02, {})


def test_value_against_context():
    context = AttrDict({"header": AttrDict({"image_size": 100})})
    assert (V == C.header.image_size)(100, context)
    assert not (V == C.header.image_size)(99, context)


def test_conditional():
    expr = Cond(C.magic == "KPK1", "compressed", "raw")
    assert expr(AttrDict({"magic": "KPK1"})) == "compressed"
    assert expr(AttrDict({"magic": "KPK0"})) == "raw"


def test_getitem_and_slices():
    context = AttrDict({"n": 2})
    arr = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(
        F.transpose(V[:, : C.n])(arr, context), arr[:, :2].T
    )
    assert (C.items[0])(AttrDict({"items": [10, 20]})) == 10


def test_named_functions():
    assert F.dtype("<i2")({}).itemsize == 2
    assert F.float(V)("1.5", {}) == 1.5
    register_function("twice", lambda x: 2 * x)
    assert F.twice(C.n)(AttrDict({"n": 21})) == 42


def test_json_roundtrip():
    arr = np.arange(12).reshape(3, 4)
    exprs = [
        ((C.header.row_bytes // 4) * C.header.nb_grid_pts_y, None),
        ((V & 0x03) != 0, 0x81),
        (V.isin("KPK0", "KPK1"), "KPK0"),
        (Cond(C.magic == "KPK1", F.float(V), V), "1.5"),
        (F.transpose(V[:, : C.header.nb_grid_pts_x]), arr),
        (Tup(C.a, -C.b, 5), None),
    ]
    context = AttrDict(
        {
            "header": AttrDict({"row_bytes": 16, "nb_grid_pts_y": 3, "nb_grid_pts_x": 2}),
            "magic": "KPK0",
            "a": 1,
            "b": 2,
        }
    )
    for expr, value in exprs:
        # Serialize, push through JSON, rehydrate
        d = json.loads(json.dumps(expr.to_dict()))
        rehydrated = from_dict(d)
        # The AST is stable under round-tripping
        assert rehydrated.to_dict() == expr.to_dict()
        # ... and evaluates identically
        np.testing.assert_array_equal(
            np.asarray(expr(value, context)), np.asarray(rehydrated(value, context))
        )


def test_truthiness_is_an_error():
    with pytest.raises(TypeError):
        bool(C.a == 1)


def test_bytes_literal():
    # Bytes literals arise e.g. from magic validation (MetroPro's magic is
    # not valid UTF-8)
    expr = V == b"\x88\x1b\x03q"
    assert expr(b"\x88\x1b\x03q", {})
    assert not expr(b"XXXX", {})
    d = json.loads(json.dumps(expr.to_dict()))
    assert d["args"][1] == {"kind": "bytes", "value": "iBsDcQ=="}
    rehydrated = from_dict(d)
    assert rehydrated(b"\x88\x1b\x03q", {})
    assert rehydrated.to_dict() == expr.to_dict()


def test_dict_expr():
    from SurfaceTopography.IO.expr import DictExpr

    expr = DictExpr({"data": C.entries[0].prefix, "n": C.nb_entries * 2})
    context = AttrDict(
        {"entries": [AttrDict({"prefix": "p0"})], "nb_entries": 3}
    )
    assert expr(context) == {"data": "p0", "n": 6}
    rehydrated = from_dict(json.loads(json.dumps(expr.to_dict())))
    assert rehydrated(context) == {"data": "p0", "n": 6}
    assert rehydrated.to_dict() == expr.to_dict()


def test_stream_filter_functions():
    import io
    import zlib

    import zstandard

    payload = b"declarative readers" * 10
    compressed = zstandard.ZstdCompressor().compress(payload)
    assert F.zstd_reader(V)(io.BytesIO(compressed), {}).read() == payload
    compressed = zlib.compress(payload)
    assert F.zlib_reader(V)(io.BytesIO(compressed), {}).read() == payload


###
# Equivalence of lambda-free layouts with the production readers
###

# The ZMG file layout, re-expressed without lambdas. Compare with
# `ZMGReader._file_layout`.
zmg_file_layout = CompoundLayout(
    [
        BinaryStructure(
            [
                ("magic", "16s", Validate("Zeta-Instruments", FileFormatMismatch)),
                (None, "69s"),
                ("nb_grid_pts_x", "I", Validate(V > 0, CorruptFile)),
                ("nb_grid_pts_y", "I", Validate(V > 0, CorruptFile)),
                (None, "4s"),
                ("step_x", "f", Validate(V > 0, CorruptFile)),
                ("step_y", "f", Validate(V > 0, CorruptFile)),
                ("step_z", "f"),
                (None, "8s"),
                ("comment_size", "I"),
                (None, "84s"),
            ],
            name="header",
        ),
        RawBuffer("comment", C.header.comment_size),
        BinaryArray(
            "data",
            Tup(C.header.nb_grid_pts_y, C.header.nb_grid_pts_x),
            F.dtype("<i2"),
            conversion_fun=F.transpose(V),
        ),
    ]
)


def test_zmg_layout_equivalence(file_format_examples):
    file_path = os.path.join(file_format_examples, "zmg-1.zmg")

    reader = ZMGReader(file_path)
    with open(file_path, "rb") as f:
        metadata = zmg_file_layout.from_stream(f, {})
        assert metadata.header == reader.metadata.header
        heights = metadata.data(f)

    np.testing.assert_array_equal(
        heights * reader.channels[0].height_scale_factor,
        reader.topography().heights(),
    )


# The ZON file layout, re-expressed without lambdas. Compare with
# `ZONReader._file_layout`. The zstd decompression filter is the
# `zstd_reader` registry function.
def zon_array_member(name, dtype_str, element_size):
    itemsize = np.dtype(dtype_str).itemsize
    return CompoundLayout(
        [
            BinaryStructure(
                [
                    ("nb_grid_pts_x", "i", Validate(V > 0, CorruptFile)),
                    ("nb_grid_pts_y", "i", Validate(V > 0, CorruptFile)),
                    (
                        "element_size",
                        "i",
                        Validate(element_size, UnsupportedFormatFeature),
                    ),
                    (
                        "row_bytes",
                        "i",
                        Validate(
                            (V % itemsize == 0)
                            & (V >= C.nb_grid_pts_x * element_size),
                            CorruptFile,
                        ),
                    ),
                ],
                byte_order="<",
                name="header",
            ),
            BinaryArray(
                "data",
                Tup(C.header.nb_grid_pts_y, C.header.row_bytes // itemsize),
                F.dtype(dtype_str),
                conversion_fun=F.transpose(V[:, : C.header.nb_grid_pts_x]),
            ),
        ],
        name=name,
    )


zon_file_layout = CompoundLayout(
    [
        BinaryStructure(
            [
                ("magic", "4s", Validate(V.isin("KPK0", "KPK1"), FileFormatMismatch)),
                ("bmp_size", "L"),
            ],
            byte_order="<",
            name="header",
        ),
        ZipContainer(
            [
                (
                    "686613b8-27b5-4a29-8ffc-438c2780873e",
                    XMLStructure(
                        name="calibration",
                        converters={
                            "MeterPerPixel": F.float(V),
                            "MeterPerUnit": F.float(V),
                        },
                    ),
                ),
                (
                    "4cdb0c75-5706-48cc-a9a1-adf395d609ae",
                    zon_array_member("height_data", "<i4", 4),
                ),
                (
                    "9e7ca6dd-0e6a-4d7b-8986-f0a7b773c916",
                    zon_array_member("valid_pixel_data", "u1", 1),
                    True,
                ),
            ],
            stream_filter=Cond(C.header.magic == "KPK1", F.zstd_reader(V), V),
        ),
    ]
)

# The mask rule, as it would appear in a declarative channel-binding layer
zon_mask_rule = (V & 0x03) != 0


def test_zon_layout_equivalence(file_format_examples):
    file_path = os.path.join(file_format_examples, "zon-1.zon")

    reader = ZONReader(file_path)
    with open(file_path, "rb") as f:
        metadata = zon_file_layout.from_stream(f, {})

        header = metadata.height_data.header
        assert (header.nb_grid_pts_x, header.nb_grid_pts_y) == reader.channels[
            0
        ].nb_grid_pts
        np.testing.assert_allclose(
            (
                header.nb_grid_pts_x * metadata.calibration.XYCalibration.MeterPerPixel,
                header.nb_grid_pts_y * metadata.calibration.XYCalibration.MeterPerPixel,
            ),
            reader.channels[0].physical_sizes,
        )

        heights = np.ma.masked_array(
            metadata.height_data.data(f),
            mask=zon_mask_rule(metadata.valid_pixel_data.data(f), {}),
        )

    assert np.ma.count_masked(heights) == 15620
    reference = reader.topography().heights()
    np.testing.assert_array_equal(heights.mask, reference.mask)
    np.testing.assert_allclose(
        heights * reader.channels[0].height_scale_factor, reference
    )
