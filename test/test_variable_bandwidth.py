#
# Copyright 2017, 2020 Lars Pastewka
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
"""
Test tools for variable bandwidth analysis.
"""

import pytest

import numpy as np

from NuMPI import MPI

from SurfaceTopography import Topography, UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_checkerboard_detrend_1d():
    arr = np.zeros([6])
    arr[:2] = 1.0
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(3)
    np.testing.assert_allclose(outarr, np.zeros([6]), atol=1e-12)

    arr[0] = -1.0
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(3)
    np.testing.assert_allclose(outarr, np.zeros([6]), atol=1e-12)


def test_checkerboard_detrend_order2_1d():
    arr = np.zeros([6])
    arr[:3] = 1.0
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(2, order=2)
    np.testing.assert_allclose(outarr, np.zeros([6]), atol=1e-12)

    arr[0] = -1.0
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(2, order=2)
    np.testing.assert_allclose(outarr, np.zeros([6]), atol=1e-12)

    arr[1] = 1.5
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(2, order=2)
    np.testing.assert_allclose(outarr, np.zeros([6]), atol=1e-12)

    # Order 1 (linear) cannot detrend this topography to zeros
    arr[1] = 1.5
    outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend_profile(2, order=1)
    np.testing.assert_allclose(outarr, [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0], atol=1e-12)


def test_checkerboard_detrend_2d():
    arr = np.zeros([4, 4])
    arr[:2, :2] = 1.0
    outarr = Topography(arr, arr.shape).checkerboard_detrend_area((2, 2))
    np.testing.assert_allclose(outarr, np.zeros([4, 4]), atol=1e-12)

    arr = np.zeros([4, 4])
    arr[:2, :2] = 1.0
    arr[:2, 1] = 2.0
    outarr = Topography(arr, arr.shape).checkerboard_detrend_area((2, 2))
    np.testing.assert_allclose(outarr, np.zeros([4, 4]), atol=1e-12)


def test_checkerboard_detrend_profile_2d():
    arr = np.zeros([4, 4])
    arr[:2, :2] = 1.0
    outarr = Topography(arr, arr.shape).checkerboard_detrend_profile(2)
    np.testing.assert_allclose(outarr, np.zeros([4, 4]), atol=1e-12)

    arr = np.zeros([4, 4])
    arr[:2, :2] = 1.0
    arr[:2, 1] = 2.0
    arr[:, 1] += 1  # offsets do not matter since all profile are independently tilt corrected
    arr[:, 2] += 1.5
    arr[:, 3] += 0.7
    outarr = Topography(arr, arr.shape).checkerboard_detrend_profile(2)
    np.testing.assert_allclose(outarr, np.zeros([4, 4]), atol=1e-12)


def test_checkerboard_detrend_order2_2d():
    arr = np.zeros([6, 6])
    arr[:3, :3] = 1.0
    outarr = Topography(arr, arr.shape).checkerboard_detrend_area((2, 2), order=2)
    np.testing.assert_allclose(outarr, np.zeros([6, 6]), atol=1e-12)

    arr = np.zeros([6, 6])
    arr[:3, :3] = 1.0
    arr[:3, 1] = 2.0
    outarr = Topography(arr, arr.shape).checkerboard_detrend_area((2, 2), order=2)
    np.testing.assert_allclose(outarr, np.zeros([6, 6]), atol=1e-12)

    arr = np.zeros([6, 6])
    arr[:3, :3] = 1.0
    arr[:3, 1] = 2.0
    arr[:3, 2] = -0.5
    outarr = Topography(arr, arr.shape).checkerboard_detrend_area((2, 2), order=2)
    np.testing.assert_allclose(outarr, np.zeros([6, 6]), atol=1e-12)


def test_checkerboard_detrend_with_no_subdivisions():
    r = 32
    x, y = np.mgrid[:r, :r]
    h = 1.3 * x - 0.3 * y + 0.02 * x * x + 0.03 * y * y - 0.013 * x * y
    t = Topography(h, (1, 1), periodic=False)
    # This should be the same as a detrend with detrend_mode='height'
    ut1 = t.checkerboard_detrend_area((1, 1))
    ut2 = t.detrend().heights()
    np.testing.assert_allclose(ut1, ut2)


def test_self_affine_topography_1d():
    r = 16384
    for H in [0.3, 0.8]:
        t0 = fourier_synthesis((r,), (1,), H, rms_slope=0.1,
                               amplitude_distribution=lambda n: 1.0)

        for t in [t0, t0.to_nonuniform()]:
            mag, bwidth, rms = t.variable_bandwidth_from_profile(nb_grid_pts_cutoff=r // 32)
            np.testing.assert_allclose(rms[0], t.detrend().rms_height_from_profile())
            np.testing.assert_allclose(bwidth, t.physical_sizes[0] / mag)
            # Since this is a self-affine surface, rms(mag) ~ mag^-H
            b, a = np.polyfit(np.log(mag[1:]), np.log(rms[1:]), 1)
            # The error is huge...
            assert abs(H + b) < 0.1


def test_self_affine_topography_2d():
    r = 2048
    res = [r, r]
    for H in [0.3, 0.8]:
        t = fourier_synthesis(res, (1, 1), H, rms_slope=0.1,
                              amplitude_distribution=lambda n: 1.0)
        mag, bwidth, rms = t.variable_bandwidth_from_area(nb_grid_pts_cutoff=r // 32)
        np.testing.assert_allclose(rms[0], t.detrend().rms_height_from_area())
        # Since this is a self-affine surface, rms(mag) ~ mag^-H
        b, a = np.polyfit(np.log(mag[1:]), np.log(rms[1:]), 1)
        # The error is huge...
        assert abs(H + b) < 0.1
