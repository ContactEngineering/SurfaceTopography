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
Round-trip CI gate for the format description contract
(docs/format_description_contract.rst): for every reader that declares
channel bindings,

1. its exported description document contains zero opaque markers,
2. a *generic* reader rehydrated from the JSON document parses every
   corpus fixture with identical channel metadata, heights and undefined
   pixels as the authored reader, and
3. the document structure is contract-valid.

These tests are functional, not smoke: they read every fixture twice and
compare the full height maps.
"""

import json
import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.IO import (
    AL3DReader,
    MetroProReader,
    PLUReader,
    SURReader,
    TMDReader,
    ZMGReader,
    ZONReader,
)
from SurfaceTopography.IO.description import (
    SCHEMA_VERSION,
    count_opaque,
    reader_description,
    reader_from_dict,
)
from SurfaceTopography.IO.Reader import MagicMatch

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)

# All readers with declarative channel bindings, and their corpus fixtures
DECLARATIVE_READERS = [
    (ZMGReader, ["zmg-1.zmg", "zmg-2.zmg"]),
    (TMDReader, ["tmd-1.tmd"]),
    (AL3DReader, ["al3d-1.al3d"]),
    (SURReader, ["sur-1.sur", "sur-2.sur", "sur-3.sur", "sur-4.sur", "sur-5.sur"]),
    (MetroProReader, ["metropro-1.dat"]),
    (PLUReader, ["plu-1.plu"]),
    (ZONReader, ["zon-1.zon"]),
]

_IDS = [cls.__name__ for cls, _ in DECLARATIVE_READERS]


def undefined_filled(topography):
    """Height map as float array with undefined pixels as NaN."""
    return np.ma.filled(
        np.ma.asarray(topography.heights(), dtype=float), np.nan
    )


@pytest.fixture(params=DECLARATIVE_READERS, ids=_IDS)
def reader_and_fixtures(request):
    return request.param


def test_document_is_complete(reader_and_fixtures):
    """Gate (1): zero opaque markers in the exported document."""
    reader_class, _ = reader_and_fixtures
    document = reader_description(reader_class)
    assert count_opaque(document) == 0, (
        f"The description of {reader_class.__name__} contains "
        f"non-serializable hooks."
    )


def test_document_structure(reader_and_fixtures):
    """Gate (3): contract-required document structure."""
    reader_class, _ = reader_and_fixtures
    document = reader_description(reader_class)

    assert document["schema_version"] == SCHEMA_VERSION
    fmt = document["format"]
    assert fmt["id"] == reader_class._format
    assert fmt["name"] == reader_class._name
    assert isinstance(fmt["file_extensions"], list)
    assert isinstance(fmt["mime_types"], list)
    # An empty magic list is allowed (e.g. PLU has no reliable magic and
    # detection answers "maybe")
    for magic in fmt["magic"]:
        assert isinstance(magic["offset"], int)
        assert isinstance(magic["bytes"], str)  # base64
    assert "core" in document["capabilities"]
    assert isinstance(document["layout"], dict)
    assert len(document["channels"]) > 0

    # The document is valid JSON
    json.dumps(document)


def test_generic_reader_equivalence(reader_and_fixtures, file_format_examples):
    """
    Gate (2): the generic reader built from the JSON document reproduces
    the authored reader on every fixture — channel metadata, heights and
    undefined pixels.
    """
    reader_class, fixtures = reader_and_fixtures
    document = json.loads(json.dumps(reader_description(reader_class)))
    generic_class = reader_from_dict(document)

    for fixture in fixtures:
        file_path = os.path.join(file_format_examples, fixture)
        authored = reader_class(file_path)
        generic = generic_class(file_path)

        authored_channels = authored.channels
        generic_channels = generic.channels
        assert len(authored_channels) == len(generic_channels), fixture

        for a, g in zip(authored_channels, generic_channels):
            assert a.name == g.name, fixture
            assert a.dim == g.dim, fixture
            assert a.nb_grid_pts == g.nb_grid_pts, fixture
            np.testing.assert_allclose(
                a.physical_sizes, g.physical_sizes, err_msg=fixture
            )
            assert a.unit == g.unit, fixture
            if a.height_scale_factor is None:
                assert g.height_scale_factor is None, fixture
            else:
                np.testing.assert_allclose(
                    a.height_scale_factor, g.height_scale_factor, err_msg=fixture
                )

        for index in range(len(authored_channels)):
            ta = authored.topography(channel_index=index)
            tg = generic.topography(channel_index=index)
            np.testing.assert_array_equal(
                undefined_filled(ta), undefined_filled(tg), err_msg=fixture
            )


def test_magic_detection_from_document(reader_and_fixtures, file_format_examples):
    """The generic reader detects its own fixtures from magic bytes."""
    reader_class, fixtures = reader_and_fixtures
    document = json.loads(json.dumps(reader_description(reader_class)))
    generic_class = reader_from_dict(document)

    if not document["format"]["magic"]:
        # No reliable magic: detection must answer "maybe"
        assert generic_class.can_read(b"\x00" * 512) == MagicMatch.MAYBE
        return

    for fixture in fixtures:
        with open(os.path.join(file_format_examples, fixture), "rb") as f:
            buffer = f.read(512)
        assert generic_class.can_read(buffer) == MagicMatch.YES, fixture
        assert reader_class.can_read(buffer) == MagicMatch.YES, fixture
    assert generic_class.can_read(b"\x00" * 512) == MagicMatch.NO


def test_export_is_deterministic(reader_and_fixtures):
    """Two exports of the same reader produce identical documents."""
    reader_class, _ = reader_and_fixtures
    a = json.dumps(reader_description(reader_class), sort_keys=True)
    b = json.dumps(reader_description(reader_class), sort_keys=True)
    assert a == b
