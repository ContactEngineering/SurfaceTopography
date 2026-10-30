#
# Copyright 2020-2021 Lars Pastewka
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

"""Test for resampling of noisy data."""

import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only

from SurfaceTopography.Support.Regression import make_grid, resample

pytestmark = serial_only


def test_make_log_grid():
    for i in range(4):
        # This makes bin edges [0, 1, 10, 100, ...]
        x, e = make_grid('log', 1, 10 ** i, nb_points_per_decade=1, dectol=0.0)
        np.testing.assert_allclose(e, 10 ** np.arange(i + 1))
        assert len(e) == len(x) + 1


def test_make_quadratic_grid():
    x, e = make_grid('quadratic', 0, 100, nb_points=2)
    assert len(e) == len(x) + 1
    np.testing.assert_allclose(e, [0, 100 / np.sqrt(2), 100])


def test_make_linear_grid():
    x, e = make_grid('linear', 2, 33, nb_points=5)
    assert len(e) == len(x) + 1
    np.testing.assert_allclose(e, np.linspace(2, 33, 6))


@pytest.mark.parametrize('method,func,tol_kwargs', [
    ('bin-average', lambda x: x, dict(atol=1e-12)),
    ('bin-average', lambda x: np.sin(x), dict(atol=1e-3)),
    ('bin-average', lambda x: x ** 2, dict(atol=1e-3)),
])
def test_resample(method, func, tol_kwargs):
    x = np.linspace(0, 1, 101)
    y = func(x)

    x_resampled, bin_edges, y_resampled, _ = resample(x, y, collocation='linear', nb_points=13, method=method)
    np.testing.assert_allclose(y_resampled, func(x_resampled), **tol_kwargs)


@pytest.mark.parametrize('method,func,tol_kwargs', [
    ('bin-average', lambda x: x ** 2, dict(rtol=0.03)),
    ('bin-average', lambda x: x ** -2.8, dict(rtol=0.2)),
])
def test_logresample(method, func, tol_kwargs):
    x = np.linspace(0.1, 10, 1001)
    y = func(x)

    x_resampled, bin_edges, y_resampled, _ = resample(x, y, collocation='log', nb_points_per_decade=3, method=method)
    np.testing.assert_allclose(y_resampled, func(x_resampled), **tol_kwargs)


def test_unknown_resampling_method():
    # Only bin averaging is supported; Gaussian process regression was removed
    from SurfaceTopography.Generation import fourier_synthesis
    x = np.linspace(0.1, 10, 101)
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resample(x, x ** 2, method='gaussian-process')
    t = fourier_synthesis((64,), (1.,), 0.8, rms_height=1, rng=0)
    with pytest.raises(ValueError, match="Unknown resampling method"):
        t.power_spectrum_from_profile(resampling_method='gaussian-process')
    with pytest.raises(ValueError, match="Unknown resampling method"):
        t.autocorrelation_from_profile(resampling_method='gaussian-process')


def test_explicit_bin_edges():
    # Bin edges given as list or array give the same result as the
    # equivalent automatically generated grid
    x = np.linspace(0, 1, 101)
    y = np.sin(x)
    x_auto, edges_auto, y_auto, _ = resample(x, y, collocation='linear', nb_points=10, min_value=0, max_value=1)
    for edges in [edges_auto, list(edges_auto)]:
        x_explicit, edges_explicit, y_explicit, _ = resample(x, y, collocation=edges)
        np.testing.assert_allclose(edges_explicit, edges_auto)
        np.testing.assert_allclose(x_explicit, x_auto)
        np.testing.assert_allclose(y_explicit, y_auto)
    with pytest.raises(ValueError):
        resample(x, y, collocation=[0.5, 0.2, 1])
    with pytest.raises(ValueError, match="Unknown collocation"):
        resample(x, y, collocation='cubic', nb_points=3)


@pytest.mark.parametrize("dim", [1, 2])
def test_explicit_bin_edges_for_spectra(dim):
    # PSD and ACF accept explicit bin edges as array or list
    from SurfaceTopography.Generation import fourier_synthesis
    t = fourier_synthesis((128,) * dim, (1.,) * dim, 0.8, rms_height=1, rng=0)
    for analysis in ["power_spectrum_from_profile", "autocorrelation_from_profile"] + \
            (["power_spectrum_from_area", "autocorrelation_from_area"] if dim == 2 else []):
        x_log, y_log = getattr(t, analysis)(collocation='linear', nb_points=7, reliable=False)
        edges = np.linspace(np.min(x_log) * 0.999, np.max(x_log) * 1.001, 5)
        x_edges, y_edges = getattr(t, analysis)(collocation=edges, reliable=False)
        x_list, y_list = getattr(t, analysis)(collocation=list(edges), reliable=False)
        assert len(x_edges) == 4
        np.testing.assert_allclose(x_list, x_edges)
        np.testing.assert_allclose(y_list, y_edges)
        assert np.all((x_edges.compressed() >= edges[0]) & (x_edges.compressed() <= edges[-1]))
