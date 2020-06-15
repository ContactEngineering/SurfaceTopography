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

from SurfaceTopography.Generation import fourier_synthesis


def test_lowcut():
    # high number of points required because of binning in the isotropic psd
    n = 200
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * n / 4
    q, psd = t.lowcut(cutoff_wavevector=cutoff_wavevector).power_spectrum_2D()
    assert (psd[q < 0.9 * cutoff_wavevector] < 1e-10).all()
    # the cut is not clean because of the binning in the 2D PSD (Ciso)

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.loglog(q, psd)
        ax.loglog(*t.power_spectrum_2D())
        ax.axvline(cutoff_wavevector)
        fig.show()


def test_highcut():
    n = 100
    # t = SurfaceTopography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13, 13), 0.9, 1.)

    cutoff_wavevector = 2 * np.pi / 13 * 0.4 * n
    q, psd = t.highcut(cutoff_wavevector=cutoff_wavevector).power_spectrum_2D()
    assert (psd[q > 1.5 * cutoff_wavevector] < 1e-10).all()

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="filtered")
        ax.loglog(*t.power_spectrum_2D(), label="original")
        ax.legend()
        fig.show()
