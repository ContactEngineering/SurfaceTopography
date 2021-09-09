#
# Copyright 2020 Lars Pastewka
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

import pytest
import numpy as np

from NuMPI import MPI

from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


@pytest.mark.parametrize("n", [128, 129])
def test_fourier_synthesis(n):
    H = 0.74
    rms_slope = 1.2
    s = 2e-6

    topography = fourier_synthesis((n, n), (s, s),
                                   H,
                                   rms_slope=rms_slope,
                                   long_cutoff=s / 4,
                                   short_cutoff=4 * s / n)

    qx, psdx = topography.power_spectrum_from_profile()
    qy, psdy = topography.transpose().power_spectrum_from_profile()

    assert psdy[-1] < 10 * psdx[-1]  # assert psdy is not much bigger
    assert abs(topography.rms_gradient() - rms_slope) / rms_slope < 1e-1


def test_fourier_synthesis_rms_height_more_wavevectors(comm_self):
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(50):
        topography = fourier_synthesis(
            (n, n), (s, s),
            H,
            rms_height=rms_height,
            rolloff=0,
            long_cutoff=s / 8,
            short_cutoff=4 * s / n,
            # amplitude_distribution=lambda n: np.ones(n)
            )

        realised_rms_heights.append(topography.rms_height_from_area())
    # print(abs(np.mean(realised_rms_heights) - rms_height) / rms_height)
    # TODO: this is not very accurate !
    assert abs(np.mean(
        realised_rms_heights) - rms_height) / rms_height < 0.1


def test_fourier_synthesis_rms_height():
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(50):
        topography = fourier_synthesis(
            (n, n), (s, s),
            H,
            rms_height=rms_height,
            long_cutoff=None,
            short_cutoff=4 * s / n,
            # amplitude_distribution=lambda n: np.ones(n)
            )
        realised_rms_heights.append(topography.rms_height_from_area())
    # TODO: this is not very accurate !
    assert abs(np.mean(
        realised_rms_heights) - rms_height) / rms_height < 0.5


def test_fourier_synthesis_c0():
    H = 0.7
    c0 = 8.

    n = 512
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    topography = fourier_synthesis((n, n), (s, s),
                                   H,
                                   c0=c0,
                                   long_cutoff=s / 2,
                                   short_cutoff=ls,
                                   amplitude_distribution=lambda n: np.ones(n)
                                   )
    ref_slope = np.sqrt(1 / (4 * np.pi) * c0 / (1 - H) * qs ** (2 - 2 * H))
    assert abs(topography.rms_gradient() - ref_slope) / ref_slope < 1e-1

    if False:
        import matplotlib.pyplot as plt
        q, psd = topography.power_spectrum_from_area()

        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="generated data")
        ax.loglog(q, c0 * q ** (-2 - 2 * H), "--", label=r"$c_0 q^{-2-2H}$")

        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{iso}$")
        ax.legend()
        ax.set_ylim(bottom=1)
        plt.show(block=True)

        q, psd = topography.power_spectrum_from_profile()
        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="generated data")
        ax.loglog(q, c0 / np.pi * q ** (-1 - 2 * H), "--",
                  label=r"$c_0 q^{-1-2H}$")

        ax.legend()
        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{1D}$")
        plt.show(block=True)


def test_fourier_synthesis_1D_input():
    H = 0.7
    c0 = 1.

    n = 512
    s = n * 4.
    ls = 8
    np.random.seed(0)
    fourier_synthesis(
        (n,), (s,),
        H,
        c0=c0,
        long_cutoff=s / 2,
        short_cutoff=ls,
        amplitude_distribution=lambda n: np.ones(n)
        )
    # TODO: What's the point of this test? There is nothing that is tested


@pytest.mark.parametrize("n", (256, 1024))
def test_fourier_synthesis_linescan_c0(n):
    H = 0.7
    c0 = 8.

    s = n * 4.
    ls = 32
    qs = 2 * np.pi / ls
    np.random.seed(0)
    t = fourier_synthesis(
        (n,), (s,),
        c0=c0,
        hurst=H,
        long_cutoff=s / 2,
        short_cutoff=ls,
        amplitude_distribution=lambda n: np.ones(n)
        )

    if False:
        import matplotlib.pyplot as plt
        q, psd = t.power_spectrum_from_profile()

        fig, ax = plt.subplots()
        ax.plot(q, psd)
        ax.plot(q, c0 * q ** (-1 - 2 * H))

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylim(bottom=1)
        fig.show()

    ref_slope = np.sqrt(1 / (2 * np.pi) * c0 / (1 - H) * qs ** (2 - 2 * H))
    assert abs(t.rms_slope_from_profile() - ref_slope) / ref_slope < 1e-1


def test_fourier_synthesis_linescan_hprms():
    H = 0.7
    hprms = .2

    n = 2048
    s = n * 4.
    ls = 64
    # np.random.seed(0)
    realised_rms_slopes = []
    for i in range(20):
        t = fourier_synthesis((n,), (s,),
                              rms_slope=hprms,
                              hurst=H,
                              long_cutoff=s / 2,
                              short_cutoff=ls,
                              )
        realised_rms_slopes.append(t.rms_slope_from_profile())
    ref_slope = hprms
    assert abs(np.mean(realised_rms_slopes) - ref_slope) / ref_slope < 1e-1


def test_fourier_synthesis_linescan_hrms_more_wavevectors():
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    H = 0.7
    hrms = 4.
    n = 4096
    s = n * 4.
    ls = 8
    np.random.seed(0)
    realised_rms_heights = []
    for i in range(50):
        t = fourier_synthesis((n,), (s,),
                              rms_height=hrms,
                              hurst=H,
                              rolloff=0,
                              long_cutoff=s / 8,
                              short_cutoff=ls,
                              )
        realised_rms_heights.append(t.rms_height_from_profile())
    realised_rms_heights = np.array(realised_rms_heights)
    ref_height = hrms
    # print(np.sqrt(np.mean(
    # (realised_rms_heights - np.mean(realised_rms_heights))**2)))
    assert abs(
        np.mean(realised_rms_heights) - ref_height) / ref_height < 0.1  #


def test_prescribed_psd_2D():
    np.random.seed(0)
    reference = fourier_synthesis((128, 128), (2.5, 2.5),
                                  hurst=0.8,
                                  c0=3., ).detrend("center")
    np.random.seed(0)
    from_function = fourier_synthesis((128, 128), (2.5, 2.5),
                                      psd=lambda q: np.where(q > 0, 3 * q ** (-2 - 2 * 0.8), 0))

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*reference.power_spectrum_from_area())
        ax.loglog(*from_function.power_spectrum_from_area(), "+")
        fig.show()

    np.testing.assert_allclose(reference.power_spectrum_from_area()[1],
                               from_function.power_spectrum_from_area()[1],
                               atol=1e-5, rtol=1e-6)


def test_prescribed_psd_2D_shortcut():
    np.random.seed(0)
    reference = fourier_synthesis((128, 128), (2.5, 2.5),
                                  hurst=0.8,
                                  c0=3.,
                                  short_cutoff=0.1, ).detrend("center")
    np.random.seed(0)
    from_function = fourier_synthesis((128, 128), (2.5, 2.5),
                                      psd=lambda q: np.where(q > 0, 3 * q ** (-2 - 2 * 0.8), 0),
                                      short_cutoff=0.1, )

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*reference.power_spectrum_from_area())
        ax.loglog(*from_function.power_spectrum_from_area(), "+")
        fig.show()

    np.testing.assert_allclose(reference.power_spectrum_from_area()[1],
                               from_function.power_spectrum_from_area()[1],
                               atol=1e-5, rtol=1e-6)


def test_prescribed_psd_1D():
    np.random.seed(0)
    reference = fourier_synthesis((128,), (2.5,),
                                  hurst=0.8,
                                  c0=3.,
                                  ).detrend("center")
    np.random.seed(0)
    from_function = fourier_synthesis((128,), (2.5,),
                                      psd=lambda q: np.where(q > 0, 3 * q ** (-1 - 2 * 0.8), 0),
                                      )

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*reference.power_spectrum_from_profile())
        ax.loglog(*from_function.power_spectrum_from_profile(), "+")
        fig.show()

    np.testing.assert_allclose(reference.power_spectrum_from_profile()[1],
                               from_function.power_spectrum_from_profile()[1],
                               atol=1e-5, rtol=1e-6)
