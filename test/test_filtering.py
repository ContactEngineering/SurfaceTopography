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

import numpy as np

import SurfaceTopography
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography import Topography


def test_longcut():
    # high number of points required because of binning in the isotropic psd
    n = 200
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * n / 4
    q, psd = t.longcut(cutoff_wavevector=cutoff_wavevector).power_spectrum_2D()
    assert (psd[q < 0.9 * cutoff_wavevector] < 1e-10).all()
    # the cut is not clean because of the binning in the 2D PSD (Ciso)

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.loglog(q, psd)
        ax.loglog(*t.power_spectrum_2D())
        ax.axvline(cutoff_wavevector)
        fig.show()


def test_shortcut():
    n = 100
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * 0.4 * n
    q, psd = t.shortcut(cutoff_wavevector=cutoff_wavevector
                        ).power_spectrum_2D()
    assert (psd[q > 1.5 * cutoff_wavevector] < 1e-10).all()

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="filtered")
        ax.loglog(*t.power_spectrum_2D(), label="original")
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
        ax.loglog(*hc.power_spectrum_2D(), "+", label="shortcut")
        ax.loglog(*fhc.power_spectrum_2D(), "x", label="filter")

        ax.loglog(*t.power_spectrum_2D(), label="original")
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
        ax.loglog(*hc.power_spectrum_2D(), "+", label="shortcut")
        ax.loglog(*fhc.power_spectrum_2D(), "x", label="filter")

        ax.loglog(*t.power_spectrum_2D(), label="original")
        ax.legend()
        fig.show()

    np.testing.assert_allclose(fhc.heights(), hc.heights())


def test_isotropic_1d():
    n = 32

    t = fourier_synthesis((n,), (13,), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * n / 4
    q, psd = t.filter(
        filter_function=lambda q: q > cutoff_wavevector).power_spectrum_1D()
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
