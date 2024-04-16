#
# Copyright 2020-2022, 2024 Lars Pastewka
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

import os

import numpy as np
import pytest
import scipy

from SurfaceTopography import NonuniformLineScan, Topography, read_topography
from SurfaceTopography.Generation import fourier_synthesis


def test_bearing_area_nonuniform(plot=False):
    n = 2048
    hm = 0.1
    X = np.arange(n)  # n+1 because we need the endpoint
    # sinsurf = np.sin(2 * np.pi * X / L) * hm
    trisurf = hm * scipy.signal.windows.triang(n)

    t = NonuniformLineScan(X, trisurf)

    h = np.linspace(0, hm, 101)
    P = t.bearing_area(h)

    P_analytic = 1 - np.linspace(0, hm, 101) / hm

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(h, P_analytic, 'k-')
        plt.plot(h, P, 'r--')
        plt.xlabel('Height')
        plt.ylabel('Bearing area')
        plt.show()

    np.testing.assert_allclose(P, P_analytic, atol=1e-3)

    np.testing.assert_allclose(t.bearing_area([hm / 4, hm / 3, hm / 2]), [0.75036639, 0.66699235, 0.50024426])


def test_bearing_area_uniform_is_monotonous(plot=False):
    t = fourier_synthesis((64,), (1,), 0.8, rms_slope=0.1, periodic=False)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test nonuniform
    P = t.to_nonuniform().bearing_area(heights)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(heights, P, 'x-')
        plt.xlabel('Height')
        plt.ylabel('Bearing area')
        plt.show()

    assert (np.diff(P) < 0).all()

    # Test uniform
    P = t.bearing_area(heights)
    assert (np.diff(P) < 0).all()

    # Test uniform periodic
    t = fourier_synthesis((64,), (1,), 0.8, rms_slope=0.1, periodic=True)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    P = t.bearing_area(heights)
    assert (np.diff(P) < 0).all()


def test_bearing_area_bounds_1d():
    t = fourier_synthesis((64,), (1,), 0.8, rms_slope=0.1, periodic=False)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test uniform
    ba = t.bearing_area()

    # Test bounds
    lower, upper = ba.bounds(heights)
    eps = 1e-12
    np.testing.assert_array_less(lower, upper + eps)
    np.testing.assert_array_less(lower, ba(heights) + eps)
    np.testing.assert_array_less(ba(heights), upper + eps)

    # Test nonuniform
    ba = t.to_nonuniform().bearing_area()

    # Test bounds
    lower, upper = ba.bounds(heights)
    eps = 1e-12
    np.testing.assert_array_less(lower, upper + eps)
    np.testing.assert_array_less(lower, ba(heights) + eps)
    np.testing.assert_array_less(ba(heights), upper + eps)

    # Test uniform periodic
    t = fourier_synthesis((64,), (1,), 0.8, rms_slope=0.1, periodic=True)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test bounds
    ba = t.bearing_area()
    lower, upper = ba.bounds(heights)
    eps = 1e-12
    np.testing.assert_array_less(lower, upper + eps)
    np.testing.assert_array_less(lower, ba(heights) + eps)
    np.testing.assert_array_less(ba(heights), upper + eps)


def test_bearing_area_bounds_2d():
    t = fourier_synthesis((64, 63), (1, 1), 0.8, rms_slope=0.1, periodic=False)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test uniform
    ba = t.bearing_area()

    # Test bounds
    lower, upper = ba.bounds(heights)
    assert (lower <= upper).all()
    assert (lower <= ba(heights)).all()
    assert (ba(heights) <= upper).all()

    # Test uniform periodic
    t = fourier_synthesis((64, 63), (1, 1), 0.8, rms_slope=0.1, periodic=True)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test bounds
    ba = t.bearing_area()
    lower, upper = ba.bounds(heights)
    eps = 1e-12
    np.testing.assert_array_less(lower, upper + eps)
    np.testing.assert_array_less(lower, ba(heights) + eps)
    np.testing.assert_array_less(ba(heights), upper + eps)


@pytest.mark.parametrize('periodic', [True, False])
def test_bearing_area_topography(periodic, plot=False):
    nx, ny = 9, 4
    hm = 0.1
    trisurf = hm * np.linspace(-1, 3, nx)
    trisurf[nx // 2:] = 2 * hm - trisurf[nx // 2:]
    if periodic:
        nx -= 1
        trisurf = trisurf[:-1]
    trisurf = np.repeat(trisurf.reshape(-1, 1), ny, axis=1)

    t = Topography(trisurf, (1, 1), periodic=periodic)

    h = np.linspace(-hm, hm, 101)
    P = t.bearing_area(h)

    P_analytic = 0.5 - np.linspace(-hm, hm, 101) / (2 * hm)

    if plot:
        import matplotlib.pyplot as plt

        x, y, heights = t.positions_and_heights()
        plt.figure()
        plt.subplot(211)
        plt.plot(x[:, 0], heights[:, 0], 'kx-')
        plt.xlabel('Position')
        plt.ylabel('Height')

        plt.subplot(212)
        plt.plot(h, P_analytic, 'r--', lw=4)
        plt.plot(h, P, 'k-')
        plt.xlabel('Height')
        plt.ylabel('Bearing area')

        plt.tight_layout()
        plt.show()

    np.testing.assert_allclose(P, P_analytic, atol=1e-3)


@pytest.mark.parametrize('nb_grid_pts,periodic', [((64, 65), False), ((16, 3), True), ((256, 256), True)])
def test_bearing_area_topography_is_monotonous(nb_grid_pts, periodic, plot=False):
    t = fourier_synthesis(nb_grid_pts, (1, 1), 0.8, rms_slope=0.1, periodic=periodic)
    mn = t.min()
    mx = t.max()
    heights = np.linspace(mn, mx, 100)

    # Test nonuniform
    P = t.bearing_area(heights)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(heights, P, 'x-')
        plt.xlabel('Height')
        plt.ylabel('Bearing area')
        plt.show()

    assert (np.diff(P) < 0).all()

    # Test uniform
    P = t.bearing_area(heights)
    assert (np.diff(P) < 0).all()


def test_tilt_unit_conversion(file_format_examples):
    t = read_topography(os.path.join(file_format_examples, 'frt-1.frt'))
    assert t.unit == 'm'
    t1 = t.detrend('rms-tilt')
    t2 = t.detrend('rms-tilt').to_unit('m')
    ba1 = t1.bearing_area()
    ba2 = t2.bearing_area()

    assert ba1.min == ba2.min
    assert ba1.max == ba2.max


@pytest.mark.parametrize('fn', ['opd-2.opd', 'plux-1.plux'])
def test_bounds_on_topography_with_missing_data(file_format_examples, fn):
    t = read_topography(os.path.join(file_format_examples, fn))
    ba = t.bearing_area()
    x = np.linspace(ba.min, ba.max, 101)
    b = ba(x)
    bl, bu = ba.bounds(x)
    assert (bl <= b).all()
    assert (b <= bu).all()
    assert (b == b).all()  # np.nan != np.nan, so this will fail if there are NaNs
