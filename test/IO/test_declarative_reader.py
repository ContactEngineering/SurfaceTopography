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
Features of `DeclarativeReaderBase` exercised with a small synthetic format
"""

import io
import json
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.IO.binary import BinaryArray, BinaryStructure
from SurfaceTopography.IO.description import layout_from_dict, layout_to_dict
from SurfaceTopography.IO.expr import C, F, Tup
from SurfaceTopography.IO.Reader import (
    CompoundLayout,
    DataKind,
    DeclarativeReaderBase,
    While,
)
from SurfaceTopography.UniformLineScanAndTopography import UniformLineScan

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


class _ProfileReader(DeclarativeReaderBase):
    """Header with the number of points, followed by heights and phases."""

    _format = "test-profile"
    _name = "Test profile"
    _description = "Synthetic format for testing"

    _file_layout = CompoundLayout(
        [
            BinaryStructure([("nb_points", "<I")], name="header"),
            BinaryArray("heights", Tup(C.header.nb_points), F.dtype("<f8")),
            BinaryArray("phases", Tup(C.header.nb_points), F.dtype("<f8")),
        ]
    )

    _channel_bindings = [
        {
            "name": "Height",
            "dim": 1,
            "nb_grid_pts": Tup(C.header.nb_points),
            "physical_sizes": Tup(2.0),
            "unit": "µm",
            "height_scale_factor": 0.5,
            "uniform": True,
            "data": C.heights,
        },
        {
            "name": "Phase",
            "dim": 1,
            "nb_grid_pts": Tup(C.header.nb_points),
            "physical_sizes": Tup(2.0),
            "unit": "µm",
            "data_kind": "phase",
            "data_unit": "deg",
            "uniform": True,
            "data": C.phases,
        },
    ]


def _profile_file(heights, phases):
    return io.BytesIO(
        struct.pack("<I", len(heights))
        + np.asarray(heights, dtype="<f8").tobytes()
        + np.asarray(phases, dtype="<f8").tobytes()
    )


def test_line_scan_channel():
    reader = _ProfileReader(_profile_file([1.0, 2.0, 4.0], [10.0, 20.0, 30.0]))
    height, phase = reader.channels

    assert height.dim == 1
    assert height.nb_grid_pts == (3,)
    # Grid sizes are plain integers, so channel information is JSON-clean
    assert all(type(n) is int for n in height.nb_grid_pts)
    json.dumps(height.nb_grid_pts)

    t = reader.topography(channel_index=0)
    assert t.dim == 1
    assert isinstance(t.parent_topography, UniformLineScan)
    assert t.physical_sizes == (2.0,)
    assert t.unit == "µm"
    np.testing.assert_allclose(t.heights(), [0.5, 1.0, 2.0])


def test_data_kind():
    reader = _ProfileReader(_profile_file([1.0, 2.0], [10.0, 20.0]))
    height, phase = reader.channels

    assert height.data_kind == DataKind.HEIGHT
    assert height.is_height_channel
    assert phase.data_kind == DataKind.PHASE
    assert not phase.is_height_channel
    assert phase.data_unit == "deg"
    assert [c.name for c in reader.height_channels] == ["Height"]


def test_while_with_expression_condition():
    # Records of a length byte and that many payload bytes, terminated by a
    # record of length zero
    layout = While(
        BinaryStructure([("length", "B")]),
        C.length != 0,
        name="records",
    )

    # The condition is an expression, so it must be serialized as such and
    # not as an opaque layout
    encoded = layout_to_dict(layout)
    assert "condition" in encoded["body"][1]
    assert "type" not in encoded["body"][1]

    data = bytes([3, 2, 0, 7])
    for parsed_layout in (layout, layout_from_dict(json.loads(json.dumps(encoded)))):
        records = parsed_layout.from_stream(io.BytesIO(data), {})["records"]
        assert [r["length"] for r in records] == [3, 2, 0]
