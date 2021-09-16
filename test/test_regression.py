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

import numpy as np

from SurfaceTopography.Support.Regression import make_grid, resample_radial


def test_make_log_grid():
    for i in range(4):
        # This makes bin edges [0, 1, 10, 100, ...]
        x, e = make_grid('log', 1, 10 ** i, nb_points_per_decade=1)
        np.testing.assert_allclose(e[1:], 10 ** np.arange(i + 1))
        assert len(e) == len(x) + 1


def test_make_quadratic_grid():
    x, e = make_grid('quadratic', 0, 100, nb_points=2)
    assert len(e) == len(x) + 1
    np.testing.assert_allclose(e, [0, 100 / np.sqrt(2), 100])


def test_make_linear_grid():
    x, e = make_grid('linear', 2, 33, nb_points=5)
    assert len(e) == len(x) + 1
    np.testing.assert_allclose(e, np.linspace(2, 33, 6))
