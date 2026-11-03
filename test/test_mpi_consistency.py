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


"""
Results for domain-decomposed (MPI-parallel) topographies must equal those
of the serial topography, or the analysis must raise NotImplementedError.
"""

import numpy as np
import pytest
from NuMPI.FFT import make_fft_engine

from SurfaceTopography.Generation import fourier_synthesis


def _serial_and_decomposed(comm, periodic=True, nb_grid_pts=(64, 48), physical_sizes=(1., 0.75)):
    serial = fourier_synthesis(nb_grid_pts, physical_sizes, 0.8, rms_height=1., rng=0, periodic=periodic, unit="µm")
    serial = serial.squeeze()
    fft = make_fft_engine(nb_grid_pts, communicator=comm)
    decomposed = serial.domain_decompose(fft.subdomain_locations, fft.nb_subdomain_grid_pts, communicator=comm)
    return serial, decomposed


def _local(serial_heights, decomposed):
    return serial_heights[decomposed.subdomain_slices]


def test_pipeline_keeps_decomposition(comm):
    serial, decomposed = _serial_and_decomposed(comm)
    for pipeline in [lambda t: t.scale(2.), lambda t: t.to_unit("nm"), lambda t: t.detrend("center"),
                     lambda t: t.scale(2.).squeeze(), lambda t: t.superpose(t)]:
        s, d = pipeline(serial), pipeline(decomposed)
        assert d.is_domain_decomposed == (comm.size > 1)
        assert d.nb_grid_pts == s.nb_grid_pts
        if d.is_domain_decomposed:
            assert d.communicator is decomposed.communicator
        np.testing.assert_allclose(d.heights(), _local(s.heights(), d))
        assert np.isclose(d.mean(), s.mean())


def test_unsupported_pipeline_steps_raise(comm):
    if comm.size == 1:
        pytest.skip("Only relevant for domain-decomposed topographies")
    _, decomposed = _serial_and_decomposed(comm)
    for pipeline in [lambda t: t.transpose(), lambda t: t.translate((1, 1)),
                     lambda t: t.shortcut(cutoff_wavelength=0.1), lambda t: t.downsample(2),
                     lambda t: t.interpolate_undefined_data()]:
        with pytest.raises(NotImplementedError):
            pipeline(decomposed)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_robust_statistics(comm, periodic, masked):
    serial, decomposed = _serial_and_decomposed(comm, periodic=periodic)
    if masked:
        mask = np.zeros(serial.nb_grid_pts, dtype=bool)
        mask[10:20, 5:30] = True
        serial = serial.__class__(np.ma.masked_array(serial.heights(), mask=mask), serial.physical_sizes,
                                  periodic=periodic, unit=serial.unit)
        decomposed = serial.domain_decompose(decomposed.subdomain_locations, decomposed.nb_subdomain_grid_pts,
                                             communicator=comm)
    heights = np.linspace(-2, 2, 7)
    np.testing.assert_allclose(decomposed.bearing_area(heights), serial.bearing_area(heights), rtol=1e-12)
    np.testing.assert_allclose(decomposed.median(), serial.median(), rtol=1e-10)
    np.testing.assert_allclose(decomposed.mad_height(), serial.mad_height(), rtol=1e-10)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("mode,rtol", [("center", 1e-12), ("median", 1e-10), ("height", 1e-10),
                                       ("curvature", 1e-10), ("mad-tilt", 1e-5)])
def test_detrend(comm, periodic, mode, rtol):
    serial, decomposed = _serial_and_decomposed(comm, periodic=periodic)
    s, d = serial.detrend(mode), decomposed.detrend(mode)
    np.testing.assert_allclose(d.coeffs, s.coeffs, rtol=rtol, atol=rtol * np.ptp(s.heights()))
    np.testing.assert_allclose(d.heights(), _local(s.heights(), d), rtol=rtol, atol=rtol * np.ptp(s.heights()))
    assert d.is_periodic == s.is_periodic


def _assert_same_result(serial_result, decomposed_result, rtol=1e-10):
    for s, d in zip(serial_result, decomposed_result):
        np.testing.assert_array_equal(np.ma.getmaskarray(d), np.ma.getmaskarray(s))
        np.testing.assert_allclose(np.ma.compressed(np.ma.asarray(d)), np.ma.compressed(np.ma.asarray(s)), rtol=rtol,
                                   atol=rtol * np.max(np.abs(np.ma.compressed(np.ma.asarray(s)))))


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_profile_analyses(comm, periodic, masked):
    serial, decomposed = _serial_and_decomposed(comm, periodic=periodic)
    if masked:
        mask = np.zeros(serial.nb_grid_pts, dtype=bool)
        mask[10:20, 5:30] = True
        serial = serial.__class__(np.ma.masked_array(serial.heights(), mask=mask), serial.physical_sizes,
                                  periodic=periodic, unit=serial.unit)
        decomposed = serial.domain_decompose(decomposed.subdomain_locations, decomposed.nb_subdomain_grid_pts,
                                             communicator=comm)
    analyses = [lambda t: t.autocorrelation_from_profile(resampling_method=None, reliable=False),
                lambda t: t.autocorrelation_from_profile(reliable=False),
                lambda t: t.scale_dependent_slope_from_profile(reliable=False),
                lambda t: t.scale_dependent_curvature_from_profile(resampling_method=None, reliable=False)]
    if not masked:
        analyses += [lambda t: t.power_spectrum_from_profile(resampling_method=None, reliable=False),
                     lambda t: t.power_spectrum_from_profile(reliable=False),
                     lambda t: (t.integrate_psd_from_profile(lambda q: q ** 2, reliable=False),)]
    for analysis in analyses:
        _assert_same_result(analysis(serial), analysis(decomposed))
    np.testing.assert_allclose(decomposed.variance_half_derivative_via_autocorrelation_from_profile(reliable=False),
                               serial.variance_half_derivative_via_autocorrelation_from_profile(reliable=False),
                               rtol=1e-10)


def test_serial_only_analyses_raise(comm):
    # Analyses that do not support domain decomposition raise instead of
    # returning the result of the local subdomain
    if comm.size == 1:
        pytest.skip("Only relevant for domain-decomposed topographies")
    _, decomposed = _serial_and_decomposed(comm)
    for name in ["rms_gradient", "autocorrelation_from_area", "variance_half_derivative",
                 "scale_dependent_slope_from_area"]:
        with pytest.raises(NotImplementedError, match=name):
            getattr(decomposed, name)()


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("nb_grid_pts,physical_sizes", [((64, 48), (1., 0.75)), ((63, 49), (1., 0.75)),
                                                        ((100, 100), (3., 3.))])
@pytest.mark.parametrize("kwargs", [dict(), dict(nb_points_per_decade=20), dict(collocation="linear", nb_points=30),
                                    dict(collocation="quadratic", nb_points=30)])
def test_power_spectrum_from_area(comm, periodic, nb_grid_pts, physical_sizes, kwargs):
    # (100, 100) with sizes (3, 3) has a lower bin edge that is sensitive to
    # the floating-point arithmetic of the wavevector spacing
    serial, decomposed = _serial_and_decomposed(comm, periodic=periodic, nb_grid_pts=nb_grid_pts,
                                                physical_sizes=physical_sizes)
    _assert_same_result(serial.power_spectrum_from_area(reliable=False, **kwargs),
                        decomposed.power_spectrum_from_area(reliable=False, **kwargs))


def test_superpose_requires_same_decomposition(comm):
    if comm.size == 1:
        pytest.skip("Only relevant for domain-decomposed topographies")
    serial, decomposed = _serial_and_decomposed(comm)
    # Same local shapes, but every process holds the slab of its neighbor
    nx, ny = serial.nb_grid_pts
    lny = decomposed.nb_subdomain_grid_pts[1]
    shifted = serial.domain_decompose((0, (decomposed.subdomain_locations[1] + lny) % ny),
                                      decomposed.nb_subdomain_grid_pts, communicator=comm)
    with pytest.raises(ValueError, match="different decompositions"):
        decomposed.superpose(shifted)
    with pytest.raises(ValueError, match="not decomposed"):
        decomposed.superpose(serial)
    np.testing.assert_allclose(decomposed.superpose(decomposed.scale(2.)).heights(), 3 * decomposed.heights())


def test_rank_deficient_polyfit(comm):
    # With heights defined only at y=0, the plane fit has no information on
    # the slope along y; serial and parallel return the minimum-norm fit
    serial, _ = _serial_and_decomposed(comm)
    mask = np.ones(serial.nb_grid_pts, dtype=bool)
    mask[:, 0] = False
    serial = serial.__class__(np.ma.masked_array(serial.heights(), mask=mask), serial.physical_sizes, unit=serial.unit)
    fft = make_fft_engine(serial.nb_grid_pts, communicator=comm)
    decomposed = serial.domain_decompose(fft.subdomain_locations, fft.nb_subdomain_grid_pts, communicator=comm)
    np.testing.assert_allclose(decomposed.polyfit(1), serial.polyfit(1), rtol=1e-8, atol=1e-10)


def test_window_is_local(comm):
    # The radial window of a decomposed topography has the size of the
    # subdomain and equals the corresponding part of the serial window
    serial, decomposed = _serial_and_decomposed(comm, periodic=False)
    s, d = serial.window(), decomposed.window()
    assert d.window_data.shape == decomposed.nb_subdomain_grid_pts
    np.testing.assert_allclose(d.window_data, s.window_data[decomposed.subdomain_slices], rtol=1e-12)
    np.testing.assert_allclose(d.heights(), _local(s.heights(), d), rtol=1e-12, atol=1e-14)


def test_profile_analyses_require_decomposition_along_y(comm):
    # Profile analyses need complete profiles along x on every process
    if comm.size == 1:
        pytest.skip("Only relevant for domain-decomposed topographies")
    serial, _ = _serial_and_decomposed(comm)
    nx, ny = serial.nb_grid_pts
    lnx = nx // comm.size
    x_slabs = serial.domain_decompose((comm.rank * lnx, 0), (lnx, ny), communicator=comm)
    for analysis in [lambda t: t.power_spectrum_from_profile(), lambda t: t.autocorrelation_from_profile(),
                     lambda t: t.integrate_psd_from_profile(lambda q: q ** 2), lambda t: t.Rq()]:
        with pytest.raises(NotImplementedError, match="decomposition along y"):
            analysis(x_slabs)


def test_ill_conditioned_polyfit(comm):
    # Heights defined only on a narrow strip along y make the quadratic fit
    # ill-conditioned (condition number ~1e6); parallel and serial fits agree
    # (the normal equations would square the condition number)
    serial, _ = _serial_and_decomposed(comm, nb_grid_pts=(64, 1024), physical_sizes=(1., 16.))
    mask = np.ones(serial.nb_grid_pts, dtype=bool)
    mask[:, 510:513] = False
    serial = serial.__class__(np.ma.masked_array(serial.heights(), mask=mask), serial.physical_sizes, unit=serial.unit)
    fft = make_fft_engine(serial.nb_grid_pts, communicator=comm)
    decomposed = serial.domain_decompose(fft.subdomain_locations, fft.nb_subdomain_grid_pts, communicator=comm)
    serial_coeffs = serial.polyfit(2)
    np.testing.assert_allclose(decomposed.polyfit(2), serial_coeffs, rtol=1e-8,
                               atol=1e-8 * np.max(np.abs(serial_coeffs)))


def test_unsupported_partition_raises_on_all_processes(comm):
    # Full-x slabs on some processes and x-split rectangles on others: all
    # processes raise (instead of some of them waiting in a collective)
    if comm.size < 3:
        pytest.skip("Requires a partition with full-x slabs and x-split rectangles")
    serial, _ = _serial_and_decomposed(comm)
    nx, ny = serial.nb_grid_pts
    if comm.rank == 0:
        location, size = (0, 0), (nx, ny // 2)
    else:
        # The upper half is split along x among the other processes
        bounds = np.linspace(0, nx, comm.size).astype(int)
        location = (bounds[comm.rank - 1], ny // 2)
        size = (bounds[comm.rank] - bounds[comm.rank - 1], ny - ny // 2)
    mixed = serial.domain_decompose(location, size, communicator=comm)
    for analysis in [lambda t: t.bearing_area(0.), lambda t: t.power_spectrum_from_profile(),
                     lambda t: t.autocorrelation_from_profile(), lambda t: t.Rq()]:
        with pytest.raises(NotImplementedError, match="decomposition along y"):
            analysis(mixed)

