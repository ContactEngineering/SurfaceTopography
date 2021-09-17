#
# Copyright 2020 Lars Pastewka
#           2020 Antoine Sanner
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

import pytest

import numpy as np

import SurfaceTopography
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography import Topography, UniformLineScan


def test_longcut():
    # high number of points required because of binning in the isotropic psd
    n = 200
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * n / 4
    q, psd = t.longcut(cutoff_wavevector=cutoff_wavevector).power_spectrum_from_area(nb_points_per_decade=10)
    m = np.isfinite(psd)
    q = q[m]
    psd = psd[m]
    np.testing.assert_allclose(psd[q < 0.9 * cutoff_wavevector], 0, atol=1e-12)
    # the cut is not clean because of the binning in the 2D PSD (Ciso)

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.loglog(q, psd)
        ax.loglog(*t.power_spectrum_from_area())
        ax.axvline(cutoff_wavevector)
        fig.show()


def test_shortcut():
    n = 100
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * 0.4 * n
    q, psd = t.shortcut(cutoff_wavevector=cutoff_wavevector
                        ).power_spectrum_from_area()
    np.testing.assert_allclose(psd[q > 1.5 * cutoff_wavevector], 0)

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="filtered")
        ax.loglog(*t.power_spectrum_from_area(), label="original")
        ax.legend()
        fig.show()


def test_shortcut_vs_isotropic_filter():
    print(SurfaceTopography.__file__)
    n = 100
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    np.random.seed(0)
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * 0.4 * n
    hc = t.shortcut(cutoff_wavevector=cutoff_wavevector)
    fhc = t.filter(filter_function=lambda q: q <= cutoff_wavevector)

    assert hc.is_filter_isotropic
    assert fhc.is_filter_isotropic
    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*hc.power_spectrum_from_area(), "+", label="shortcut")
        ax.loglog(*fhc.power_spectrum_from_area(), "x", label="filter")

        ax.loglog(*t.power_spectrum_from_area(), label="original")
        ax.legend()
        fig.show()

    np.testing.assert_allclose(fhc.heights(), hc.heights())


def test_shortcut_vs_square_filter():
    print(SurfaceTopography.__file__)
    n = 100
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    np.random.seed(0)
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * 0.2 * n
    hc = t.shortcut(cutoff_wavevector=cutoff_wavevector, kind="square step")
    fhc = t.filter(
        filter_function=lambda qx, qy: (np.abs(qx) <= cutoff_wavevector)
        * (np.abs(qy) <= cutoff_wavevector),
        isotropic=False)

    assert not hc.is_filter_isotropic
    assert not fhc.is_filter_isotropic

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*hc.power_spectrum_from_area(), "+", label="shortcut")
        ax.loglog(*fhc.power_spectrum_from_area(), "x", label="filter")

        ax.loglog(*t.power_spectrum_from_area(), label="original")
        ax.legend()
        fig.show()

    np.testing.assert_allclose(fhc.heights(), hc.heights())


def test_isotropic_1d():
    n = 32

    t = fourier_synthesis((n,), (13,), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * n / 4
    q, psd = t.filter(
        filter_function=lambda q: q > cutoff_wavevector).power_spectrum_from_profile()
    m = np.isfinite(psd)
    q = q[m]
    psd = psd[m]
    assert (psd[q < 0.9 * cutoff_wavevector] < 1e-10).all()


def test_mirror_stitch():
    t = Topography(np.array(((0, 1),
                             (0, 0))),
                   (2., 3.))
    tp = t.mirror_stitch()

    assert tp.is_periodic

    np.testing.assert_allclose(tp.physical_sizes, (4., 6.))

    np.testing.assert_allclose(tp.heights(),
                               [[0, 1, 1, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 1, 0]
                                ]
                               )


def test_window_line_scan():
    nx = 27
    sx = 1.3
    h = np.ones((nx,))

    t = UniformLineScan(h, physical_sizes=(sx,), periodic=True).window()
    np.testing.assert_almost_equal(t.heights()[0], 1.0)
    np.testing.assert_almost_equal((t.heights()**2).sum() / nx, 1.0)

    t = UniformLineScan(h, physical_sizes=(sx,), periodic=False).window()
    np.testing.assert_almost_equal(t.heights()[0], 0.0)
    np.testing.assert_almost_equal((t.heights()**2).sum() / nx, 1.0)

    with pytest.raises(ValueError):
        UniformLineScan(h, physical_sizes=(sx,), periodic=False).window(direction='y').window_data

    with pytest.raises(ValueError):
        UniformLineScan(h, physical_sizes=(sx,), periodic=False).window(direction='radial').window_data


def test_window_topography():
    nx, ny = 27, 13
    sx, sy = 1.3, 1.7
    h = np.ones((nx, ny))

    t = Topography(h, physical_sizes=(sx, sy), periodic=True).window(direction='x')
    np.testing.assert_almost_equal(t.heights()[0, ny//2], 1.0)
    np.testing.assert_almost_equal((t.heights()**2).sum() / (nx*ny), 1.0)

    t = Topography(h, physical_sizes=(sx, sy), periodic=False).window(direction='x')
    np.testing.assert_almost_equal(t.heights()[0, ny//2], 0.0)
    assert t.heights()[nx//2, 0] > 1.0
    np.testing.assert_almost_equal((t.heights()**2).sum() / (nx*ny), 1.0)

    t = Topography(h, physical_sizes=(sx, sy), periodic=False).window(direction='y')
    np.testing.assert_almost_equal(t.heights()[nx//2, 0], 0.0)
    assert t.heights()[0, ny//2] > 1.0
    np.testing.assert_almost_equal((t.heights()**2).sum() / (nx*ny), 1.0)

    t = Topography(h, physical_sizes=(sx, sy), periodic=False).window(direction='radial')
    np.testing.assert_almost_equal(t.heights()[0, 0], 0.0)
    np.testing.assert_almost_equal(t.heights()[nx//2, 0], 0.0)
    np.testing.assert_almost_equal((t.heights()**2).sum() / (nx*ny), 1.0)
