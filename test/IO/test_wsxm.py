#
# Copyright 2020-2023 Lars Pastewka
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

import os

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import WSXMReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_wsxm_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'top-1.top')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_stp_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'stp-1.stp')

    r = WSXMReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 256
    assert ny == 256

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 600, rtol=1e-6)
    np.testing.assert_allclose(sy, 600, rtol=1e-6)

    assert t.unit == 'nm'

    # Double data is stored in the unit of the z amplitude and is not
    # rescaled (as in Gwyddion's wsxmfile.c); the reader previously
    # rescaled with the z amplitude divided by the (rounded) data range from
    # the header, which yielded 1.654156.
    np.testing.assert_allclose(t.rms_height_from_area(), 1.6541594, rtol=1e-6)


def test_top_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'top-1.top')

    r = WSXMReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 256
    assert ny == 256

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 600, rtol=1e-6)
    np.testing.assert_allclose(sy, 600, rtol=1e-6)

    assert t.unit == 'nm'

    np.testing.assert_allclose(t.rms_height_from_area(), 3.153099, rtol=1e-6)


def _write_wsxm(raw, data_type, y_amplitude=True):
    """Write a minimal WSxM image file."""
    import io

    ny, nx = raw.shape
    dtype = {"double": "<f8", "float": "<f4", "short": "<i2"}[data_type]
    lines = [
        "[Control]",
        "",
        "    X Amplitude: 2 µm",
    ]
    if y_amplitude:
        lines += ["    Y Amplitude: 1 µm"]
    lines += [
        "",
        "[General Info]",
        "",
        f"    Image Data Type: {data_type}",
        f"    Number of columns: {nx}",
        f"    Number of rows: {ny}",
        "    Z Amplitude: 10 nm",
        "",
        "[Miscellaneous]",
        "",
        f"    Maximum: {raw.max()}",
        f"    Minimum: {raw.min()}",
        "",
        "[Header end]",
        "",
    ]
    header = "\r\n".join(lines)
    header = (
        "WSxM file copyright UAM\r\nSxM Image file\r\n"
        f"Image header size: {len(header)}\r\n" + header
    )
    return io.BytesIO(header.encode("latin-1") + raw.astype(dtype).tobytes())


@pytest.mark.parametrize("data_type", ["double", "float", "short"])
def test_wsxm_synthetic(data_type):
    """
    Compare with the interpretation of Gwyddion's wsxmfile.c: Floating-point
    data is in the unit of the z amplitude, integer data is normalized to
    the z amplitude; the image is stored rotated by 180 degrees.
    """
    raw = np.array([[1, 2, 3], [4, 5, 7]])
    t = WSXMReader(_write_wsxm(raw, data_type)).topography()
    assert t.unit == "nm"
    assert t.nb_grid_pts == (3, 2)
    np.testing.assert_allclose(t.physical_sizes, (2000, 1000))
    if data_type == "short":
        expected = raw * 10 / (raw.max() - raw.min())
    else:
        expected = raw
    np.testing.assert_allclose(t.heights(), expected[::-1, ::-1].T, rtol=1e-6)


def test_wsxm_missing_y_amplitude():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = WSXMReader(_write_wsxm(raw, "double", y_amplitude=False)).topography()
    np.testing.assert_allclose(t.physical_sizes, (2000, 2000))
