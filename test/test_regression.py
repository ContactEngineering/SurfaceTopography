#
# Copyright 2021 Lars Pastewka
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

import pytest

import numpy as np

from SurfaceTopography.Support.Regression import make_grid, resample


def test_make_log_grid():
    for i in range(4):
        # This makes bin edges [0, 1, 10, 100, ...]
        x, e = make_grid('log', 1, 10 ** i, nb_points_per_decade=1)
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
    ('gaussian-process', lambda x: x, dict(atol=1e-3)),
    ('gaussian-process', lambda x: np.sin(x), dict(atol=1e-4)),
    ('gaussian-process', lambda x: x ** 2, dict(atol=1e-3)),
])
def test_resample(method, func, tol_kwargs):
    x = np.linspace(0, 1, 101)
    y = func(x)

    x_resampled, bin_edges, y_resampled, _ = resample(x, y, collocation='linear', nb_points=13, method=method)
    np.testing.assert_allclose(y_resampled, func(x_resampled), **tol_kwargs)


@pytest.mark.parametrize('method,func,tol_kwargs', [
    ('bin-average', lambda x: x ** 2, dict(rtol=0.03)),
    ('bin-average', lambda x: x ** -2.8, dict(rtol=0.2)),
    ('gaussian-process', lambda x: x ** 2, dict(rtol=0.001)),
    ('gaussian-process', lambda x: x ** -2.8, dict(rtol=0.03)),
])
def test_logresample(method, func, tol_kwargs):
    x = np.linspace(0.1, 10, 1001)
    y = func(x)

    x_resampled, bin_edges, y_resampled, _ = resample(x, y, collocation='log', nb_points_per_decade=3, method=method)
    np.testing.assert_allclose(y_resampled, func(x_resampled), **tol_kwargs)
