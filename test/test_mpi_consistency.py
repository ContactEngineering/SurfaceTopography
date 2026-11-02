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


def _serial_and_decomposed(comm, periodic=True, nb_grid_pts=(64, 48)):
    serial = fourier_synthesis(nb_grid_pts, (1., 0.75), 0.8, rms_height=1., rng=0, periodic=periodic, unit="µm")
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
