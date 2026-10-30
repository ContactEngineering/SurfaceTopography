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


import itertools

import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only

from SurfaceTopography import Topography, UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Uniform.Autocorrelation import (
    _masked_height_difference_correlation,
    _masked_profile_autocorrelation,
)

pytestmark = serial_only


def _with_missing_points(heights, fraction, rng):
    return np.ma.masked_array(heights, mask=rng.random(heights.shape) < fraction)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("nb_grid_pts", [(64,), (63,), (32, 24), (33, 25)])
def test_reduces_to_unmasked_computation(periodic, nb_grid_pts):
    # Without undefined points, the masked computation gives the result of
    # the computation without mask
    t = fourier_synthesis(nb_grid_pts, (1.,) * len(nb_grid_pts), 0.8, rms_height=1., rng=1, periodic=periodic)
    heights = np.ma.masked_array(t.heights(), mask=False)
    r, A = t.autocorrelation_from_profile(resampling_method=None, reliable=False)
    np.testing.assert_allclose(_masked_profile_autocorrelation(heights, periodic), A, rtol=1e-10, atol=1e-12)
    if len(nb_grid_pts) == 2:
        _, _, A_xy = t.autocorrelation_from_area(reliable=False, return_map=True)
        squared_differences, nb_pairs = _masked_height_difference_correlation(heights, periodic, axes=(0, 1))
        if not periodic:
            squared_differences = squared_differences[:nb_grid_pts[0], :nb_grid_pts[1]]
            nb_pairs = nb_pairs[:nb_grid_pts[0], :nb_grid_pts[1]]
        np.testing.assert_allclose(squared_differences / nb_pairs, A_xy, rtol=1e-10, atol=1e-12)


def _brute_force(h, mask, periodic, distance):
    # Average of (h(x+d) - h(x))^2 / 2 over all pairs of defined points
    total, count = 0.0, 0
    for x in itertools.product(*[range(n) for n in h.shape]):
        y = tuple(xi + di for xi, di in zip(x, distance))
        if periodic:
            y = tuple(yi % n for yi, n in zip(y, h.shape))
        elif any(yi >= n for yi, n in zip(y, h.shape)):
            continue
        if not mask[x] and not mask[y]:
            total += (h[y] - h[x]) ** 2 / 2
            count += 1
    return total, count


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("shape", [(17,), (9, 7)])
def test_brute_force(periodic, shape):
    rng = np.random.default_rng(3)
    h = rng.normal(size=shape)
    mask = rng.random(shape) < 0.3
    heights = np.ma.masked_array(h, mask=mask)
    # Profile ACF: pairs along the first axis, all profiles together
    A = _masked_profile_autocorrelation(heights, periodic)
    for k in range(len(A)):
        total, count = _brute_force(h, mask, periodic, (k,) + (0,) * (len(shape) - 1))
        if count == 0:
            assert A.mask[k]
        else:
            np.testing.assert_allclose(A[k], total / count, atol=1e-12)
    # Areal ACF: all distance vectors with nonnegative components
    if len(shape) == 2:
        squared_differences, nb_pairs = _masked_height_difference_correlation(heights, periodic, axes=(0, 1))
        for k, l in itertools.product(range(shape[0]), range(shape[1])):
            total, count = _brute_force(h, mask, periodic, (k, l))
            assert nb_pairs[k, l] == count
            np.testing.assert_allclose(squared_differences[k, l], total, atol=1e-12)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("kind", ["random", "block"])
def test_missing_points_leave_acf_unchanged(periodic, kind):
    # Removing points from a homogeneous surface changes the ACF only
    # within its statistical error
    t = fourier_synthesis((256, 256), (1., 1.), 0.8, rms_height=1., short_cutoff=8 / 256, rng=2, periodic=periodic)
    if kind == "random":
        mask = np.random.default_rng(0).random(t.nb_grid_pts) < 0.2
    else:
        mask = np.zeros(t.nb_grid_pts, dtype=bool)
        mask[100:104, 30:200] = True  # a scratch
    masked = Topography(np.ma.masked_array(t.heights(), mask=mask), t.physical_sizes, periodic=periodic)
    assert masked.has_undefined_data
    for function in ["autocorrelation_from_profile", "autocorrelation_from_area"]:
        r, A = getattr(t, function)(reliable=False)
        r_masked, A_masked = getattr(masked, function)(reliable=False)
        np.testing.assert_allclose(r_masked, r, rtol=0.05)
        np.testing.assert_allclose(A_masked, A, rtol=0.1)
    np.testing.assert_allclose(masked.variance_half_derivative_via_autocorrelation_from_profile(reliable=False),
                               t.variance_half_derivative_via_autocorrelation_from_profile(reliable=False), rtol=0.05)
    np.testing.assert_allclose(masked.variance_half_derivative_via_autocorrelation_from_area(reliable=False),
                               t.variance_half_derivative_via_autocorrelation_from_area(reliable=False), rtol=0.05)


def test_line_scan_with_missing_points():
    t = fourier_synthesis((512,), (1.,), 0.8, rms_height=1., short_cutoff=8 / 512, rng=4, periodic=False)
    mask = np.zeros(512, dtype=bool)
    mask[200:230] = True
    masked = UniformLineScan(np.ma.masked_array(t.heights(), mask=mask), 1., periodic=False)
    r, A = masked.autocorrelation_from_profile(resampling_method=None, reliable=False)
    r0, A0 = t.autocorrelation_from_profile(resampling_method=None, reliable=False)
    # At short distances, most pairs are unaffected by the gap
    np.testing.assert_allclose(A[:20], A0[:20], rtol=0.1, atol=1e-3 * A0.max())
    # The scale-dependent slope and the half derivative are defined
    assert np.isfinite(masked.variance_half_derivative_via_autocorrelation_from_profile(reliable=False))


def test_no_pairs_are_masked():
    # Two defined points at distance 3 in a nonperiodic line scan: only
    # distances 0 and 3 have pairs
    h = np.ma.masked_array([1., 0., 0., 4., 0.], mask=[False, True, True, False, True])
    A = _masked_profile_autocorrelation(h, periodic=False)
    np.testing.assert_array_equal(A.mask, [False, True, True, False, True])
    np.testing.assert_allclose(A[3], (4 - 1) ** 2 / 2)
