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

from SurfaceTopography.Container.IO import ZAGReader, read_container
from SurfaceTopography.IO import ZONReader


@pytest.fixture
def synthetic_zag(file_format_examples, tmp_path):
    """
    Build a minimal ZAG container (header + BMP thumbnail + ZIP archive)
    that wraps the `zon-1.zon` example file, mirroring the layout parsed
    by `ZAGReader`.
    """
    with open(os.path.join(file_format_examples, "zon-1.zon"), "rb") as f:
        zon = f.read()

    buf = io.BytesIO()
    fake_bmp = b"BM" + b"\x00" * 62  # dummy thumbnail
    buf.write(b"KPK0" + struct.pack("<L", len(fake_bmp)) + fake_bmp)
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            ZAGReader._INVENTORY_UUID,
            "<DeserializeDataMap><Item><Path>abc/item.xml</Path></Item>"
            "</DeserializeDataMap>",
        )
        z.writestr(
            "abc/item.xml",
            "<MeasurementDataMap><MeasurementData><Path>data0</Path>"
            "</MeasurementData></MeasurementDataMap>",
        )
        z.writestr(f"abc/data0/{ZAGReader._ZON_UUID}", zon)

    fn = tmp_path / "zag-1.zag"
    fn.write_bytes(buf.getvalue())
    return str(fn)


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
