#
# Copyright 2024 Lars Pastewka
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
Tests for the downsample pipeline function.
"""

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.UniformLineScanAndTopography import Topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_downsample_nth():
    nx, ny = 10, 8
    sx, sy = 2.0, 1.5
    heights = np.arange(nx * ny).reshape(nx, ny).astype(float)
    t = Topography(heights, physical_sizes=(sx, sy))

    # Downsample by factor 2
    t_down = t.downsample(2, mode="nth")
    assert t_down.nb_grid_pts == (5, 4)
    assert t_down.physical_sizes == (sx, sy)

    expected_heights = heights[::2, ::2]
    np.testing.assert_allclose(t_down.heights(), expected_heights)

    # Downsample with different factors
    t_down = t.downsample((2, 4), mode="nth")
    assert t_down.nb_grid_pts == (5, 2)
    expected_heights = heights[::2, ::4]
    np.testing.assert_allclose(t_down.heights(), expected_heights)


def test_downsample_average():
    nx, ny = 10, 8
    sx, sy = 2.0, 1.5
    heights = np.arange(nx * ny).reshape(nx, ny).astype(float)
    t = Topography(heights, physical_sizes=(sx, sy))

    # Downsample by factor 2
    t_down = t.downsample(2, mode="average")
    assert t_down.nb_grid_pts == (5, 4)
    assert t_down.physical_sizes == (sx, sy)

    # Manual average for (2, 2) patches
    expected_heights = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            expected_heights[i, j] = heights[
                2 * i: 2 * i + 2, 2 * j: 2 * j + 2
            ].mean()

    np.testing.assert_allclose(t_down.heights(), expected_heights)


def test_downsample_non_integer_multiple():
    nx, ny = 11, 9
    sx, sy = 2.0, 1.5
    heights = np.arange(nx * ny).reshape(nx, ny).astype(float)
    t = Topography(heights, physical_sizes=(sx, sy))

    # Downsample by factor 2, remaining points should be dropped
    t_down = t.downsample(2, mode="nth")
    assert t_down.nb_grid_pts == (5, 4)
    expected_heights = heights[:10, :8][::2, ::2]
    np.testing.assert_allclose(t_down.heights(), expected_heights)

    t_down = t.downsample(2, mode="average")
    assert t_down.nb_grid_pts == (5, 4)
    expected_heights = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            expected_heights[i, j] = heights[
                2 * i: 2 * i + 2, 2 * j: 2 * j + 2
            ].mean()
    np.testing.assert_allclose(t_down.heights(), expected_heights)


def test_downsample_errors():
    nx, ny = 10, 8
    heights = np.ones((nx, ny))
    t = Topography(heights, physical_sizes=(1.0, 1.0))

    with pytest.raises(ValueError):
        t.downsample(2, mode="invalid")

    with pytest.raises(ValueError):
        t.downsample((2, 2, 2))

    # 1D not supported yet based on implementation
    from SurfaceTopography.UniformLineScanAndTopography import UniformLineScan

    ls = UniformLineScan(np.ones(10), physical_sizes=1.0)
    with pytest.raises(ValueError):
        ls.downsample(2)


def test_downsample_positions():
    nx, ny = 10, 8
    sx, sy = 2.0, 1.5
    heights = np.ones((nx, ny))
    t = Topography(heights, physical_sizes=(sx, sy))

    t_down = t.downsample(2)
    x, y = t_down.positions()
    assert x.shape == (5, 4)
    assert y.shape == (5, 4)

    # Check spacing
    assert x[1, 0] - x[0, 0] == sx / 5
    assert y[0, 1] - y[0, 0] == sy / 4
