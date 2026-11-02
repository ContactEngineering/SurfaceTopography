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
