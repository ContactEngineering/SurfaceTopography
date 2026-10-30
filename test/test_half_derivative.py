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


import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only

from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Generic.Fractional import (
    container_variance_half_derivative_from_autocorrelation,
    variance_half_derivative_via_scale_dependent_slope,
)

pytestmark = serial_only


def test_masked_bins_are_dropped():
    r = np.ma.masked_array([1., 2., 3., 4., 5.], mask=[False, True, False, False, True])
    slope = np.ma.masked_array([2., 0., 1., 0.5, 0.], mask=[False, True, False, False, True])
    np.testing.assert_allclose(variance_half_derivative_via_scale_dependent_slope(r, slope),
                               variance_half_derivative_via_scale_dependent_slope(r.compressed(),
                                                                                  slope.compressed()))
    slope_nan = np.array([2., np.nan, 1., 0.5, np.nan])
    np.testing.assert_allclose(variance_half_derivative_via_scale_dependent_slope(r.data, slope_nan),
                               variance_half_derivative_via_scale_dependent_slope(r.compressed(),
                                                                                  slope.compressed()))


def test_empty_bins_do_not_change_half_derivative():
    # At 50 points per decade, many bins at short distances are empty
    # (masked); they must not contribute to the integral. 20 points per
    # decade has no empty bins and differs only by the resolution of the grid.
    t = fourier_synthesis((256, 256), (1., 1.), 0.8, rms_height=1., rng=1)
    for function in [t.variance_half_derivative_via_autocorrelation_from_profile,
                     t.variance_half_derivative_via_autocorrelation_from_area]:
        np.testing.assert_allclose(function(nb_points_per_decade=50, reliable=False),
                                   function(nb_points_per_decade=20, reliable=False), rtol=0.01)


def test_container_with_empty_bins():
    t = fourier_synthesis((256, 256), (1., 1.), 0.8, rms_height=1., rng=1, unit='m')
    c = InMemorySurfaceContainer([t])
    coarse = container_variance_half_derivative_from_autocorrelation(c, unit='m', nb_points_per_decade=20,
                                                                     reliable=False)
    fine = container_variance_half_derivative_from_autocorrelation(c, unit='m', nb_points_per_decade=50,
                                                                   reliable=False)
    np.testing.assert_allclose(fine, coarse, rtol=0.01)


def test_container_profile_half_derivative():
    # Containers provide the profile half derivative under both names
    t = fourier_synthesis((64, 64), (1., 1.), 0.8, rms_height=1., rng=1, unit='m')
    c = InMemorySurfaceContainer([t])
    np.testing.assert_allclose(c.variance_half_derivative_via_autocorrelation_from_profile(unit='m'),
                               c.variance_half_derivative_from_autocorrelation(unit='m'))


def test_container_and_topography_agree():
    # A container with a single topography gives the result of the
    # topography, up to the binning of the container average
    t = fourier_synthesis((64, 64), (1., 1.), 0.8, rms_height=1., rng=1, unit='m')
    c = InMemorySurfaceContainer([t])
    reference = t.variance_half_derivative_via_autocorrelation_from_profile(resampling_method=None, reliable=False)
    for nb_points_per_decade in [5, 20]:
        np.testing.assert_allclose(
            t.variance_half_derivative_via_autocorrelation_from_profile(nb_points_per_decade=nb_points_per_decade,
                                                                        reliable=False), reference, rtol=0.03)
        np.testing.assert_allclose(
            c.variance_half_derivative_from_autocorrelation(unit='m', nb_points_per_decade=nb_points_per_decade,
                                                            reliable=False), reference, rtol=0.03)


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("short_cutoff,rtol", [(None, 0.035), (16 / 512, 0.02)])
def test_autocorrelation_and_spectral_definitions_agree(hurst, short_cutoff, rtol):
    # On periodic surfaces no window is needed, so the autocorrelation route
    # and the spectral definition differ only by the finite range of
    # distances: the tail beyond the largest distance (included with
    # <h^2>/r_max) and the lattice scale (constant slope below the pixel size
    # vs. the sum up to the Nyquist wavevector). Without the tail term, the
    # autocorrelation route is up to 21% too small here.
    t = fourier_synthesis((512, 512), (1., 1.), hurst, rms_height=1., short_cutoff=short_cutoff, rng=2)
    # Profiles: (1/2) int_0^inf dq q C^1D(q)
    profile_spectral = np.pi / 2 * t.integrate_psd_from_profile(lambda q: np.abs(q), window=None, reliable=False)
    for kwargs in [dict(resampling_method=None), dict(nb_points_per_decade=10)]:
        np.testing.assert_allclose(
            t.variance_half_derivative_via_autocorrelation_from_profile(reliable=False, **kwargs),
            profile_spectral, rtol=rtol)
    # Areal: int d^2q/(2 pi)^2 |q| C^2D(q)
    np.testing.assert_allclose(t.variance_half_derivative_via_autocorrelation_from_area(reliable=False),
                               t.variance_half_derivative(window=None, reliable=False), rtol=rtol)


def test_container_tail_uses_topographies_at_largest_distance():
    # The tail <h^2>/r_max uses the mean square height of the topographies
    # that contribute at r_max, here only the large scan; the scale-dependent
    # curve has no tail
    large = fourier_synthesis((128, 128), (10., 10.), 0.8, rms_height=1., rng=1, unit='m')
    small = fourier_synthesis((128, 128), (0.1, 0.1), 0.8, rms_height=0.01, rng=2, unit='m')
    c = InMemorySurfaceContainer([small, large])
    r, V = c.variance_half_derivative_from_autocorrelation(unit='m', scale_dependent=True, reliable=False)
    total = c.variance_half_derivative_from_autocorrelation(unit='m', reliable=False)
    np.testing.assert_allclose(total, V[-1] + large.Rq() ** 2 / r[-1])
