#
# Copyright 2017, 2020 Lars Pastewka
#           2020 Antoine Sanner
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
Tests for linear interpolation
"""

import pytest
import numpy as np

from NuMPI import MPI

from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_grid_values_1d():
    nx = 17
    sx = 1.3
    topography = fourier_synthesis((nx,), (sx,), 0.8, rms_height=1.)
    interp = topography.interpolate_linear()
    interp_heights = interp(topography.positions())
    np.testing.assert_allclose(interp_heights, topography.heights())


def test_center_values_1d():
    nx = 17
    sx = 2.3
    topography = fourier_synthesis((nx,), (sx,), 0.8, rms_height=1.)
    interp = topography.interpolate_linear()
    interp_heights = interp(topography.positions() + sx / (2 * nx))
    heights = topography.heights()
    np.testing.assert_allclose(interp_heights, (heights + np.roll(heights, -1)) / 2)


def test_grid_values_2d():
    nx, ny = 17, 22
    sx, sy = 1.3, 1.4
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.)
    interp = topography.interpolate_linear()
    interp_heights = interp(*topography.positions())
    np.testing.assert_allclose(interp_heights, topography.heights())


def test_center_values_2d():
    nx, ny = 3, 4
    sx, sy = 3.7, 4.3
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.)
    interp = topography.interpolate_linear()
    x, y = topography.positions()
    x += sx / (2 * nx)
    y += sy / (2 * ny)
    interp_heights = interp(x, y)
    heights = topography.heights()
    interp_check = (np.roll(heights, -1, axis=0) + np.roll(heights, -1, axis=1)) / 2
    np.testing.assert_allclose(interp_heights, interp_check)


def test_offcenter_values_2d():
    nx, ny = 7, 3
    sx, sy = 1.1, 1.2
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.)
    heights = topography.heights()
    interp = topography.interpolate_linear()
    x, y = topography.positions()
    x += sx / (4 * nx)
    y += sy / (4 * ny)
    interp_heights = interp(x, y)
    interp_check = (2 * heights + np.roll(heights, -1, axis=0) + np.roll(heights, -1, axis=1)) / 4
    np.testing.assert_allclose(interp_heights, interp_check)
    x += sx / (2 * nx)
    y += sy / (2 * ny)
    interp_heights = interp(x, y)
    interp_check = (2 * np.roll(heights, (-1, -1), axis=(0, 1)) + np.roll(heights, -1, axis=0) +
                    np.roll(heights, -1, axis=1)) / 4
    np.testing.assert_allclose(interp_heights, interp_check)
