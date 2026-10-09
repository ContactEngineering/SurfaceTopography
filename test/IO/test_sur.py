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

import io
import os
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import SURReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest",
)


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, "sur-1.sur")

    read_topography(file_path)

    with open(file_path, "r") as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_sur_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, "sur-1.sur")

    r = SURReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 2560
    assert ny == 2560

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.631917268037796, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.631917268037796, rtol=1e-6)

    assert t.unit == "mm"
    assert t.info["instrument"]["vendor"] == "Digital Surf"

    np.testing.assert_allclose(
        t.rms_height_from_area(), 0.0002894642370000746, rtol=1e-6
    )


def test_sur3_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, "sur-3.sur")

    r = SURReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 512
    assert ny == 512

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.151319552, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.151319552, rtol=1e-6)

    assert t.unit == "m"

    np.testing.assert_allclose(
        t.rms_height_from_area(), 0.0011131994265859693, rtol=1e-6
    )


def test_sur4_metadata(file_format_examples, plot=False):
    file_path = os.path.join(file_format_examples, "sur-4.sur")

    r = SURReader(file_path)
    t = r.topography()

    if plot:
        import matplotlib.pyplot as plt

        t.plot()
        plt.show()

    nx, ny = t.nb_grid_pts
    assert nx == 1232
    assert ny == 1028

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.3400320075452328, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.2837280062958598, rtol=1e-6)

    assert t.unit == "mm"

    np.testing.assert_allclose(
        t.rms_height_from_area(), 0.0008984264822337376, rtol=1e-6
    )


def test_sur5_metadata(file_format_examples, plot=False):
    file_path = os.path.join(file_format_examples, "sur-5.sur")

    r = SURReader(file_path)
    t = r.topography()

    if plot:
        import matplotlib.pyplot as plt

        t.plot()
        plt.show()

    nx, ny = t.nb_grid_pts
    assert nx == 2560
    assert ny == 2560

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.631917268037796, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.631917268037796, rtol=1e-6)

    assert t.unit == "mm"

    np.testing.assert_allclose(
        t.rms_height_from_area(), 0.001165488804116226, rtol=1e-6
    )


# Number of non-measured points (marked by zmin - 2) in the example files
@pytest.mark.parametrize("index, nb_undefined", [
    (1, 877), (2, 117), (3, 233009), (4, 9550), (5, 595635)])
def test_sur_non_measured_points(file_format_examples, index, nb_undefined):
    """
    All example files have `special_points` set and mark non-measured
    points with zmin - 2 (sur-3 only has a band of valid rows).
    """
    file_path = os.path.join(file_format_examples, f"sur-{index}.sur")
    t = SURReader(file_path).topography()
    assert t.has_undefined_data
    h = t.heights()
    assert h.mask.sum() == nb_undefined
    zmin = t.info["raw_metadata"]["zmin"]
    np.testing.assert_allclose(
        h.min(), zmin * t.info["raw_metadata"]["height_scale_factor"],
        rtol=1e-6)


def _make_sur(data, inversion=0, special_points=0, unit="mm", zmin=None):
    """Build a minimal single-object SUR file; `data` is (ny, nx) int32"""
    ny, nx = data.shape
    if zmin is None:
        zmin = int(data.min())

    def s(text, n):
        return text.encode("latin-1").ljust(n, b" ")

    header = b"DIGITAL SURF" + struct.pack("<HHHH", 0, 1, 1, 2)
    header += s("object", 30) + s("operator", 30)
    header += struct.pack("<HHHHH", 0, 0, 0, special_points, 1)
    header += struct.pack("<fIH", 0, 0, 32)
    header += struct.pack("<iiiiI", zmin, int(data.max()), nx, ny, nx * ny)
    header += struct.pack("<fff", 0.5, 0.25, 1e-3)
    header += s("X", 16) + s("Y", 16) + s("Z", 16)
    header += s(unit, 16) * 6
    header += struct.pack("<fffHHH", 1, 1, 1, 0, inversion, 0)
    header += b"\0" * 12
    header += struct.pack("<HHHHHHHf", 1, 2, 3, 4, 5, 2020, 0, 0)
    header += b"\0" * 10 + struct.pack("<HH", 0, 0) + b"\0" * 128
    header += struct.pack("<fff", 0, 0, 0)
    header = header.ljust(512, b"\0")
    assert len(header) == 512
    return header + data.astype("<i4").tobytes()


@pytest.mark.parametrize("inversion", [0, 1, 2, 3])
def test_sur_inversion(inversion):
    data = np.arange(12).reshape(3, 4) * 10 + 5
    t = SURReader(io.BytesIO(_make_sur(data, inversion=inversion))).topography()
    assert t.nb_grid_pts == (4, 3)
    np.testing.assert_allclose(t.physical_sizes, (2.0, 0.75))
    expected = 1e-3 * data.T  # (nx, ny) order
    if inversion > 0:
        expected = -expected
    if inversion == 2:
        expected = expected[::-1, :]  # left-right mirrored
    elif inversion == 3:
        expected = expected[:, ::-1]  # upside-down
    np.testing.assert_allclose(t.heights(), expected)


def test_sur_ms_dos_micro_sign():
    data = np.arange(6).reshape(2, 3)
    t = SURReader(io.BytesIO(_make_sur(data, unit="\xe6m"))).topography()
    assert t.unit == "µm"


def test_sur_special_points_flag():
    data = np.array([[10, 8, 12], [13, 14, 15]])  # zmin = 10, 8 = zmin - 2
    t = SURReader(io.BytesIO(_make_sur(data, special_points=1, zmin=10))).topography()
    np.testing.assert_array_equal(t.heights().mask, (data == 8).T)
    # Without the flag, all points are valid
    t = SURReader(io.BytesIO(_make_sur(data, special_points=0, zmin=10))).topography()
    assert not t.has_undefined_data
