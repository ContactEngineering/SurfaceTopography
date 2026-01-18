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

import gzip
import os
from test.test_topography import DATADIR

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import Topography, UniformLineScan
from SurfaceTopography.IO import AscReader, open_topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_example1():
    surf = AscReader(os.path.join(DATADIR, "matrix-1.txt")).topography()
    assert isinstance(surf, Topography)
    assert surf.nb_grid_pts == (1024, 1024)
    np.testing.assert_allclose(surf.physical_sizes[0], 2000)
    np.testing.assert_allclose(surf.physical_sizes[1], 2000)
    np.testing.assert_allclose(surf.rms_height_from_area(), 17.22950485567042)
    np.testing.assert_allclose(surf.rms_gradient(), 0.4560243831362324)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "nm"


def test_example2():
    surf = AscReader(os.path.join(DATADIR, "matrix-2.txt")).topography()
    assert surf.nb_grid_pts == (650, 650)
    np.testing.assert_allclose(surf.physical_sizes[0], 0.0002404103)
    np.testing.assert_allclose(surf.physical_sizes[1], 0.0002404103)
    np.testing.assert_allclose(surf.rms_height_from_area(), 2.7722350402740072e-07)
    np.testing.assert_allclose(surf.rms_gradient(), 0.35152685030417763)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"


def test_example3():
    surf = AscReader(os.path.join(DATADIR, "matrix-3.txt")).topography()
    assert surf.nb_grid_pts == (256, 256)
    np.testing.assert_allclose(surf.physical_sizes[0], 10e-6)
    np.testing.assert_allclose(surf.physical_sizes[1], 10e-6)
    np.testing.assert_allclose(surf.rms_height_from_area(), 3.5222918750198742e-08)
    np.testing.assert_allclose(surf.rms_gradient(), 0.19235602282848963)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"


@pytest.mark.parametrize("fn", ["matrix-4.txt", "matrix-4.txt.gz"])
def test_example4(fn):
    if fn.endswith(".gz"):
        surf = AscReader(gzip.open(os.path.join(DATADIR, fn))).topography()
    else:
        surf = AscReader(os.path.join(DATADIR, fn)).topography()
    assert surf.nb_grid_pts == (75, 305)
    np.testing.assert_allclose(surf.physical_sizes[0], 2.773965e-05)
    np.testing.assert_allclose(surf.physical_sizes[1], 0.00011280791)
    np.testing.assert_allclose(surf.rms_height_from_area(), 1.1745891510991089e-07)
    np.testing.assert_allclose(surf.rms_height_from_profile(), 7.198047e-08)
    np.testing.assert_allclose(surf.rms_gradient(), 0.06776316911544318)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"

    # test setting the physical_sizes
    with pytest.raises(AttributeError):
        surf.physical_sizes = 1, 2


def test_example5():
    r = AscReader(os.path.join(DATADIR, "matrix-5.txt"))
    assert r.default_channel.physical_sizes is None

    surf = r.topography(physical_sizes=(1, 2))
    assert isinstance(surf, Topography)
    assert surf.nb_grid_pts == (10, 10)
    assert surf.physical_sizes == (1, 2)
    np.testing.assert_allclose(surf.rms_height_from_area(), 1.0)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert "unit" not in surf.info

    # test setting the physical_sizes
    surf = AscReader(os.path.join(DATADIR, "matrix-5.txt")).topography(
        physical_sizes=(1, 2)
    )
    np.testing.assert_allclose(surf.physical_sizes[0], 1)
    np.testing.assert_allclose(surf.physical_sizes[1], 2)

    bw = surf.bandwidth()
    np.testing.assert_allclose(bw[0], 1.5 / 10)
    np.testing.assert_allclose(bw[1], 1.5)

    reader = AscReader(os.path.join(DATADIR, "matrix-5.txt"))
    assert reader.default_channel.physical_sizes is None


def test_example6():
    topography_file = open_topography(os.path.join(DATADIR, "not-yet-working-1.txt"))
    surf = topography_file.topography(physical_sizes=(1,))
    assert isinstance(surf, UniformLineScan)
    np.testing.assert_allclose(surf.heights(), [1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_wyko_matrix8(file_format_examples, filename="matrix-8.txt"):
    file_path = os.path.join(file_format_examples, filename)

    r = AscReader(file_path)
    assert len(r.channels) == 1

    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (950400, 1267200))
    np.testing.assert_allclose(t.rms_height_from_area(), 74424.357775)


def test_single_column(file_format_examples, filename="single_column.txt"):
    r = open_topography(os.path.join(file_format_examples, filename))
    assert r.format() == "asc"
    assert r.default_channel.dim == 1
    assert r.default_channel.nb_grid_pts == (9,)
    assert r.default_channel.physical_sizes is None
    t = r.topography(physical_sizes=(2.3,))
    assert t.nb_grid_pts == (9,)
    assert t.physical_sizes == (2.3,)
    np.testing.assert_allclose(t.heights(), np.arange(9) + 1)
