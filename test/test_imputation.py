#
# Copyright 2020-2021 Lars Pastewka
#           2019 Antoine Sanner
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

from NuMPI import MPI

from SurfaceTopography import Topography, UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_randomly_rough_1d(plot=False):
    t = fourier_synthesis((128,), (1.,), 0.8, rms_height=1)
    mn, mx = t.min(), t.max()
    val = mn + (mx - mn) * 0.8

    h = t.heights()
    t2 = UniformLineScan(np.ma.array(h.copy(), mask=h > val), t.physical_sizes, periodic=True)
    assert t2.has_undefined_data
    t3 = t2.interpolate_undefined_data()
    assert not t3.has_undefined_data
    h = t3.heights()

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(*t.positions_and_heights(), 'ko-')
        plt.plot(*t2.positions_and_heights(), 'rx-')
        plt.plot(*t3.positions_and_heights(), 'b+-')
        plt.show()

    assert not np.ma.is_masked(h)
    assert np.ma.is_masked(t2.heights())
    assert np.max(h) <= val
    assert t3.max() <= val


def test_randomly_rough_2d(plot=False):
    t = fourier_synthesis((128, 128), (1., 1.), 0.8, rms_height=1)
    mn, mx = t.min(), t.max()
    val = mn + (mx - mn) * 0.8

    h = t.heights()
    t2 = Topography(np.ma.array(h.copy(), mask=h > val), t.physical_sizes, periodic=True)
    assert t2.has_undefined_data
    t3 = t2.interpolate_undefined_data()
    assert not t3.has_undefined_data
    h3 = t3.heights()

    if plot:
        import matplotlib.pyplot as plt
        plt.pcolormesh(t.heights(), vmin=mn, vmax=mx)
        plt.show()
        plt.pcolormesh(t2.heights(), vmin=mn, vmax=mx)
        plt.show()
        plt.pcolormesh(t3.heights(), vmin=mn, vmax=mx)
        plt.show()

    assert not np.ma.is_masked(h)
    assert np.ma.is_masked(t2.heights())
    assert np.max(h3) <= val
    assert t3.max() <= val


def test_linear_1d(plot=False):
    a, b = -1.3, 2.7
    nx = 128
    sx = 1.3
    x = np.linspace(-sx / 2, sx / 2, nx)
    h = a * x + b
    t = UniformLineScan(np.ma.array(h.copy(), mask=abs(x) < sx / 8), (sx,))
    assert t.has_undefined_data
    t2 = t.interpolate_undefined_data()

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(*t.positions_and_heights(), 'kx-')
        plt.plot(*t2.positions_and_heights(), 'k+-')
        plt.show()

    np.testing.assert_allclose(h, t2.heights())


def test_linear_2d(plot=False):
    a, b, c = -1.3, 2.7, 1.8
    nx, ny = 128, 234
    sx, sy = 1.3, 1.6
    x = np.linspace(-sx / 2, sx / 2, nx).reshape(-1, 1)
    y = np.linspace(-sy / 2, sy / 2, ny).reshape(1, -1)
    h = a * x + b * y + c
    t = Topography(np.ma.array(h.copy(), mask=np.logical_and(abs(x) < sx / 8, abs(y) < sy / 8)), (sx, sy))
    assert t.has_undefined_data
    t2 = t.interpolate_undefined_data()

    if plot:
        import matplotlib.pyplot as plt
        plt.pcolormesh(t.heights())
        plt.show()
        plt.pcolormesh(t2.heights())
        plt.show()

    assert np.ma.is_masked(t.heights())
    np.testing.assert_allclose(h, t2.heights())


def test_laplace_2d():
    nx, ny = 127, 129
    x = np.arange(nx).reshape(-1, 1) - nx / 2
    y = np.arange(ny).reshape(1, -1) - ny / 2
    h = x ** 2 - y ** 2  # This is a simple solution to the Laplace equation

    # We use this to create random cutouts
    t = fourier_synthesis((nx, ny), (1, 1), 0.8, rms_height=1).detrend('center')
    mask = t.heights() > 0  # Remove half of the points
    mask[0, :] = False  # Keep boundary points
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    t = Topography(np.ma.array(h, mask=mask), (1, 1), periodic=False)
    t2 = t.interpolate_undefined_data()
    h2 = t2.heights()

    np.testing.assert_allclose(h, h2, atol=1e-9)
