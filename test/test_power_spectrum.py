#
# Copyright 2018, 2020 Lars Pastewka
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

"""
Tests for power-spectral density analysis
"""

import os
import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from NuMPI import MPI

from SurfaceTopography import read_topography, UniformLineScan, \
    NonuniformLineScan, Topography
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Nonuniform.PowerSpectrum import sinc, dsinc

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(os.path.dirname(__file__), 'file_format_examples')

###


def test_uniform():
    for periodic in [True, False]:
        for L in [1.3, 10.6]:
            for k in [2, 4]:
                for n in [16, 128]:
                    x = np.arange(n) * L / n
                    h = np.sin(2 * np.pi * k * x / L)
                    t = UniformLineScan(h, physical_sizes=L, periodic=periodic)
                    q, C = t.power_spectrum_from_profile()

                    # The ms height of the sine is 1/2. The sum over the PSD
                    # (from -q to +q) is the ms height. Our PSD only contains
                    # *half* of the full PSD (on the +q branch, the -q branch
                    # is identical), therefore the sum over it is 1/4.
                    assert_almost_equal(C.sum() / L, 1 / 4)

                    if periodic:
                        # The value at the individual wavevector must also
                        # equal 1/4. This is only exactly true for the
                        # periodic case. In the nonperiodic, this is convolved
                        # with the Fourier transform of the window function.
                        C /= L
                        r = np.zeros_like(C)
                        r[k] = 1 / 4
                        assert_array_almost_equal(C, r)


def test_invariance():
    for a, b, c in [(2.3, 1.2, 1.7),
                    (1.5, 3.1, 3.1),
                    (0.5, 1.0, 1.0),
                    (0.5, -0.5, 0.5)]:
        q = np.linspace(0.0, 2 * np.pi / a, 101)

        x = np.array([-a, a])
        h = np.array([b, c])
        _, C1 = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            short_cutoff=None,
            window='None')

        x = np.array([-a, 0, a])
        h = np.array([b, (b + c) / 2, c])
        _, C2 = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            short_cutoff=None,
            window='None')

        x = np.array([-a, 0, a / 2, a])
        h = np.array([b, (b + c) / 2, (3 * c + b) / 4, c])
        _, C3 = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            short_cutoff=None,
            window='None')

        assert_array_almost_equal(C1, C2)
        assert_array_almost_equal(C2, C3)


def test_rectangle():
    for a, b in [(2.3, 1.45), (10.2, 0.1)]:
        x = np.array([-a, a])
        h = np.array([b, b])

        q = np.linspace(0.01, 8 * np.pi / a, 101)

        q, C = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            short_cutoff=None,
            window='None')

        C_ana = (2 * b * np.sin(a * q) / q) ** 2
        C_ana /= 2 * a

        assert_array_almost_equal(C, C_ana)


def test_triangle():
    for a, b in [(0.5, -0.5), (1, 1), (2.3, 1.45), (10.2, 0.1)]:
        x = np.array([-a, a])
        h = np.array([-b, b])

        q = np.linspace(0.01, 8 * np.pi / a, 101)

        _, C = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            short_cutoff=None,
            window='None')

        C_ana = (2 * b * (a * q * np.cos(a * q) - np.sin(a * q)) / (
                a * q ** 2)) ** 2
        C_ana /= 2 * a

        assert_array_almost_equal(C, C_ana)


def test_rectangle_and_triangle():
    for a, b, c, d in [(0.123, 1.45, 10.1, 9.3),
                       (-0.1, 5.4, -0.1, 3.43),
                       (-1, 1, 1, 1)]:
        x = np.array([a, b])
        h = np.array([c, d])

        q = np.linspace(0.01, 8 * np.pi / (b - a), 101)

        q, C = NonuniformLineScan(x, h).power_spectrum_from_profile(
            wavevectors=q,
            algorithm='brute-force',
            window='None')

        C_ana = np.exp(-1j * (a + b) * q) * (
                np.exp(1j * a * q) * (c - d + 1j * (a - b) * d * q) +
                np.exp(1j * b * q) * (d - c - 1j * (a - b) * c * q)
        ) / ((a - b) * q ** 2)
        C_ana = np.abs(C_ana) ** 2 / (b - a)

        assert_array_almost_equal(C, C_ana)


def test_dsinc():
    assert_almost_equal(dsinc(0), 0)
    assert_almost_equal(dsinc(np.pi) * np.pi, -1)
    assert_almost_equal(dsinc(2 * np.pi) * np.pi, 1 / 2)
    assert_almost_equal(dsinc(3 * np.pi) * np.pi, -1 / 3)
    assert_array_almost_equal(dsinc([0, np.pi]) * np.pi, [0, -1])
    assert_array_almost_equal(dsinc([0, 2 * np.pi]) * np.pi, [0, 1 / 2])
    assert_array_almost_equal(dsinc([0, 3 * np.pi]) * np.pi, [0, -1 / 3])

    dx = 1e-9
    for x in [0, 0.5e-6, 1e-6, 0.5, 1]:
        v1 = sinc(x + dx)
        v2 = sinc(x - dx)
        assert_almost_equal(dsinc(x), (v1 - v2) / (2 * dx), decimal=5)


def test_NaNs():
    surf = fourier_synthesis([1024, 512], [2, 1], 0.8, rms_slope=0.1)
    q, C = surf.power_spectrum_from_area(nbins=1000)
    assert np.isnan(C).sum() == 0


def test_brute_force_vs_fft():
    t = read_topography(os.path.join(DATADIR, 'example.xyz'))
    q, A = t.detrend().power_spectrum_from_profile(window="None")
    q2, A2 = t.detrend().power_spectrum_from_profile(algorithm='brute-force',
                                                     wavevectors=q, nb_interpolate=5,
                                                     window="None")
    length = len(A2)
    x = A[1:length // 16] / A2[1:length // 16]
    assert np.alltrue(np.logical_and(x > 0.90, x < 1.35))


def test_short_cutoff():
    t = read_topography(os.path.join(DATADIR, 'example.xyz'))
    q1, C1 = t.detrend().power_spectrum_from_profile()
    q2, C2 = t.detrend().power_spectrum_from_profile(short_cutoff=np.min)
    q3, C3 = t.detrend().power_spectrum_from_profile(short_cutoff=np.max)

    assert len(q3) < len(q1)
    assert len(q1) < len(q2)

    assert np.max(q3) < np.max(q1)
    assert np.max(q1) < np.max(q2)

    x, y = t.positions_and_heights()

    assert abs(np.max(q1) - 2 * np.pi / np.mean(np.diff(x))) < 0.01
    assert abs(np.max(q2) - 2 * np.pi / np.min(np.diff(x))) < 0.03
    assert abs(np.max(q3) - 2 * np.pi / np.max(np.diff(x))) < 0.02


@pytest.mark.skip(reason="just plotting")
def test_default_window_1D():
    x = np.linspace(0, 1, 200)
    heights = np.cos(2 * np.pi * x * 8.3)
    topography = UniformLineScan(heights,
                                 physical_sizes=1, periodic=True)

    if True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(*topography.power_spectrum_from_profile(), label="periodic=True")
        ax.set_xscale("log")
        ax.set_yscale("log")

    topography = UniformLineScan(heights,
                                 physical_sizes=1, periodic=False)

    if True:
        import matplotlib.pyplot as plt
        ax.plot(*topography.power_spectrum_from_profile(), label="periodic=False")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.legend()
        plt.show(block=True)


@pytest.mark.skip(reason="just plotting")
def test_default_window_2D():
    x = np.linspace(0, 1, 200).reshape(-1, 1)

    heights = np.cos(2 * np.pi * x * 8.3) * np.ones((1, 200))
    topography = Topography(heights,
                            physical_sizes=(1, 1), periodic=True)

    if True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(*topography.power_spectrum_from_profile(), label="periodic=True")
        ax.plot(*topography.power_spectrum_from_area(nbins=20),
                label="2D, periodic=True")
        ax.set_xscale("log")
        ax.set_yscale("log")

    topography = Topography(heights,
                            physical_sizes=(1, 1), periodic=False)

    if True:
        import matplotlib.pyplot as plt
        ax.plot(*topography.power_spectrum_from_profile(), label="periodic=False")
        ax.plot(*topography.power_spectrum_from_area(nbins=20),
                label="2D, periodic=False")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_ylim(bottom=1e-6)
        ax.legend()
        plt.show(block=True)


def test_q0_1D():
    surf = fourier_synthesis([1024, 512], [2.3, 1.5], 0.8, rms_height=0.87)
    rms_height = surf.rms_height_from_profile()  # Need to measure it since it can fluctuate wildly
    q, C = surf.power_spectrum_from_profile()
    ratio = rms_height**2 / (np.trapz(C, q)/np.pi)
    assert ratio > 0.2
    assert ratio < 5


def test_q0_2D():
    surf = fourier_synthesis([1024, 512], [2.3, 1.5], 0.8, rms_height=0.87)
    rms_height = surf.rms_height_from_area()  # Need to measure it since it can fluctuate wildly
    q, C = surf.power_spectrum_from_area(nbins=200, bin_edges='quadratic')
    # This is really not quantitative, it's just checking whether it's the right ballpark.
    # Any bug in normalization would show up here as an order of magnitude
    ratio = rms_height**2 / (np.trapz(q*C, q)/np.pi)
    assert ratio > 0.2
    assert ratio < 5


def test_reliability_cutoff():
    surf = fourier_synthesis([1024, 512], [2.3, 2.4], 0.8, rms_height=0.87, unit='um', info={
        'instrument': {
            'tip_radius': {
                'value': 0.001,
                'unit': 'um'
            }
        }
    })

    # 1D
    q1, C1 = surf.power_spectrum_from_profile(reliable=True)
    q2, C2 = surf.power_spectrum_from_profile(reliable=False)
    assert len(q1) < len(q2)
    assert q1.max() < q2.max()

    # 2D
    q1, C1 = surf.power_spectrum_from_area(reliable=True)
    q2, C2 = surf.power_spectrum_from_area(reliable=False)
    assert len(q1) < len(q2)
    assert q1.max() < q2.max()
