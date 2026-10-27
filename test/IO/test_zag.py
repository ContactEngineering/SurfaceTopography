#
# Copyright 2020, 2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import io
import os
import struct
import zipfile

import numpy as np
import pytest

from SurfaceTopography.Container.IO import ZAGReader, detect_format, read_container
from SurfaceTopography.IO import ZONReader


def build_zag(fn, measurements):
    """
    Build a minimal ZAG container (header + BMP thumbnail + ZIP archive),
    mirroring the layout parsed by `ZAGReader`.

    Parameters
    ----------
    fn : path
        Where to write the container.
    measurements : list of dict
        One entry per measurement, with keys ``payload`` (bytes of the
        wrapped ZON file) and optionally ``original_file_name`` and
        ``visible`` (omitted from the XML when absent, mimicking older
        container layouts).
    """
    buf = io.BytesIO()
    fake_bmp = b"BM" + b"\x00" * 62  # dummy thumbnail
    buf.write(b"KPK0" + struct.pack("<L", len(fake_bmp)) + fake_bmp)
    data_entries = []
    with zipfile.ZipFile(buf, "w") as z:
        for i, measurement in enumerate(measurements):
            entry = f"<Path>data{i}</Path>"
            if "original_file_name" in measurement:
                entry += (
                    "<OriginalFileName>"
                    f"{measurement['original_file_name']}"
                    "</OriginalFileName>"
                )
            if "visible" in measurement:
                entry += f"<Visible>{measurement['visible']}</Visible>"
            data_entries.append(f"<MeasurementData>{entry}</MeasurementData>")
            z.writestr(
                f"abc/data{i}/{ZAGReader._ZON_UUID}", measurement["payload"]
            )
        z.writestr(
            ZAGReader._INVENTORY_UUID,
            "<DeserializeDataMap><Item><Path>abc/item.xml</Path></Item>"
            "</DeserializeDataMap>",
        )
        z.writestr(
            "abc/item.xml",
            f"<MeasurementDataMap>{''.join(data_entries)}</MeasurementDataMap>",
        )

    fn.write_bytes(buf.getvalue())
    return str(fn)


@pytest.fixture
def zon_bytes(file_format_examples):
    with open(os.path.join(file_format_examples, "zon-1.zon"), "rb") as f:
        return f.read()


@pytest.fixture
def synthetic_zag(zon_bytes, tmp_path):
    """A single measurement, without name and visibility tags (as written by
    older versions of the Keyence software)."""
    return build_zag(tmp_path / "zag-1.zag", [{"payload": zon_bytes}])


@pytest.fixture
def synthetic_zag_multi(zon_bytes, tmp_path):
    """Three measurements: a named visible one, a named hidden one, and one
    without name and visibility tags."""
    return build_zag(
        tmp_path / "zag-2.zag",
        [
            {
                "payload": zon_bytes,
                "original_file_name": r"C:\Users\sol\Documents\VR-20240405_102737.zon",
                "visible": "True",
            },
            {
                "payload": zon_bytes,
                "original_file_name": r"C:\Users\sol\Documents\VR-20240405_103039.zon",
                "visible": "False",
            },
            {"payload": zon_bytes},
        ],
    )


def test_zag(synthetic_zag, file_format_examples):
    with ZAGReader(synthetic_zag) as r:
        c = r.container(0)
        assert len(c) == 1
        t = c[0]
        assert t.dim == 2

    # Heights must match a direct read of the wrapped ZON file
    t_ref = ZONReader(
        os.path.join(file_format_examples, "zon-1.zon")
    ).topography()
    np.testing.assert_allclose(c[0].heights(), t_ref.heights())


def test_zag_read_container_outlives_reader(synthetic_zag, file_format_examples):
    # `read_container` closes the reader (and its stream) before returning
    # the lazy container; element access must still work afterwards
    (c,) = read_container(synthetic_zag)
    t = c[0]
    t_ref = ZONReader(
        os.path.join(file_format_examples, "zon-1.zon")
    ).topography()
    np.testing.assert_allclose(t.heights(), t_ref.heights())


def test_zag_detect_format(synthetic_zag):
    assert detect_format(synthetic_zag) == "zag"


def test_zag_hidden_measurements_are_excluded_from_container(synthetic_zag_multi):
    with ZAGReader(synthetic_zag_multi) as r:
        c = r.container(0)
        # The hidden measurement must not appear in the analysis view
        assert len(c) == 2
        for t in c:
            assert t.dim == 2


def test_zag_members(synthetic_zag_multi, zon_bytes):
    with ZAGReader(synthetic_zag_multi) as r:
        members = r.members(0)

        # All measurements are enumerated, including the hidden one
        assert len(members) == 3

        # Names come from `OriginalFileName` (a Windows path); measurements
        # without that tag get a generated name
        assert members[0].name == "VR-20240405_102737.zon"
        assert members[1].name == "VR-20240405_103039.zon"
        assert members[2].name == "measurement-data2.zon"

        assert [m.visible for m in members] == [True, False, True]

        # ZAG containers carry no per-measurement schema metadata
        assert all(m.metadata is None for m in members)

        # The member streams must return the verbatim bytes of the wrapped
        # raw data files -- this is what ingestion into a web application
        # stores
        for member in members:
            with member.open() as f:
                assert f.read() == zon_bytes


def test_zag_members_outlive_reader(synthetic_zag_multi, zon_bytes):
    # Like the container, members must be readable after the reader has been
    # closed: ingestion enumerates first and copies the data afterwards
    with ZAGReader(synthetic_zag_multi) as r:
        members = r.members(0)
    with members[0].open() as f:
        assert f.read() == zon_bytes
