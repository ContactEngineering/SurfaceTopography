#
# Copyright 2020-2021 Lars Pastewka
#           2019, 2021 Antoine Sanner
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
Tests for scale-dependent slope analysis
"""

import numpy as np
from NuMPI.Testing.Fixtures import serial_only
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from SurfaceTopography import UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis

pytestmark = serial_only


def test_limiting_values():
    nb_grid_pts = (128, 128)
    physical_sizes = (1.2, 2.1)
    Hurst = 0.8
    slope = 0.1

    t = fourier_synthesis(nb_grid_pts, physical_sizes, Hurst,
                          rms_slope=slope,
                          short_cutoff=0.1 * np.mean(physical_sizes))

    r, s = t.scale_dependent_slope_from_profile(resampling_method=None)

    px, py = t.pixel_size

    rms_slope = np.sqrt(np.mean(((np.roll(t.heights(), 1, 0) - t.heights()) / px) ** 2))
    assert_almost_equal(s[0], rms_slope)
    assert_almost_equal(s[0], t.rms_slope_from_profile())


def test_numeric_vs_analytical():
    nx = 1001
    L = 7.3
    qs = 2 * np.pi * 8 / L
    t1 = UniformLineScan(np.sin(np.arange(nx) * L * qs / nx), L, periodic=True)

    x, y = t1.scale_dependent_slope_from_profile(resampling_method=None)
    assert_array_almost_equal(y, np.sqrt(1 - np.cos(qs * x)) / x)


def test_scale_dependent_slope_starts_at_smallest_distance():
    # Only distance zero is removed from the ACF; the smallest distance (one
    # pixel) is kept, also when the ACF is averaged over log-spaced bins
    t = fourier_synthesis((128, 128), (1., 1.), 0.8, rms_height=1., rng=2)
    dx = t.pixel_size[0]
    r, s = t.scale_dependent_slope_from_profile(resampling_method=None, reliable=False)
    assert_almost_equal(r[0], dx)
    assert_almost_equal(s[0], t.Rdq())
    for function in [t.scale_dependent_slope_from_profile, t.scale_dependent_slope_from_area]:
        r, s = function(nb_points_per_decade=10, reliable=False)
        assert_almost_equal(r.compressed()[0] / dx, 1)
