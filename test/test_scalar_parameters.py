#
# Copyright 2018, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2019 Michael Röttger
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
import unittest
import numpy as np

from NuMPI import MPI
from muFFT import FFT

from SurfaceTopography import Topography, NonuniformLineScan, UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis


@pytest.fixture
def sinewave2D(comm):
    n = 256
    X, Y = np.mgrid[slice(0, n), slice(0, n)]

    fftengine = FFT((n, n), fft="mpi", communicator=comm)

    hm = 0.1
    L = float(n)
    sinsurf = np.sin(2 * np.pi / L * X) * np.sin(2 * np.pi / L * Y) * hm
    size = (L, L)

    top = Topography(sinsurf, decomposition='domain',
                     nb_subdomain_grid_pts=fftengine.nb_subdomain_grid_pts,
                     subdomain_locations=fftengine.subdomain_locations,
                     physical_sizes=size, communicator=comm)

    return (L, hm, top)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_curvature(sinewave2D):
    L, hm, top = sinewave2D
    numerical = top.rms_curvature_from_area()
    analytical = np.sqrt(4 * (16 * np.pi ** 4 / L ** 4) * hm ** 2 / 4 / 4)
    #                 rms(∆)^2 = (qx^2 + qy^2)^2 * hm^2 / 4
    # print(numerical-analytical)
    np.testing.assert_almost_equal(numerical, analytical, 5)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_slope(sinewave2D):
    L, hm, top = sinewave2D
    numerical = top.rms_gradient()
    analytical = np.sqrt(2 * np.pi ** 2 * hm ** 2 / L ** 2)
    # print(numerical-analytical)
    np.testing.assert_almost_equal(numerical, analytical, 5)


def test_rms_height(comm, sinewave2D):
    L, hm, top = sinewave2D
    numerical = top.rms_height_from_area()
    analytical = np.sqrt(hm ** 2 / 4)

    assert numerical == analytical


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("periodic", [False, True])
def test_rms_curvature_sinewave_2D(periodic):
    precision = 5

    n = 256
    X, Y = np.mgrid[slice(0, n), slice(0, n)]
    hm = 0.3
    L = float(n)
    size = (L, L)

    surf = Topography(np.sin(2 * np.pi / L * X) * hm, physical_sizes=size,
                      periodic=periodic)
    numerical_lapl = surf.rms_laplacian()
    analytical_lapl = np.sqrt((2 * np.pi / L) ** 4 * hm ** 2 / 2)
    # print(numerical-analytical)
    np.testing.assert_almost_equal(numerical_lapl, analytical_lapl, precision)

    np.testing.assert_almost_equal(surf.rms_curvature_from_area(), analytical_lapl / 2,
                                   precision)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_curvature_paraboloid_uniform_1D():
    n = 16
    x = np.arange(n)
    curvature = 0.1
    heights = 0.5 * curvature * x ** 2

    surf = UniformLineScan(heights, physical_sizes=(n,),
                           periodic=False)
    # central finite differences are second order and so exact for the parabola
    assert abs((surf.rms_curvature_from_profile() - curvature) / curvature) < 1e-14


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_curvature_paraboloid_uniform_2D():
    n = 16
    X, Y = np.mgrid[slice(0, n), slice(0, n)]
    curvature = 0.1
    heights = 0.5 * curvature * (X ** 2 + Y ** 2)
    surf = Topography(heights, physical_sizes=(n, n), periodic=False)
    # central finite differences are second order and so exact for the
    # paraboloid
    assert abs((surf.rms_curvature_from_area() - curvature) / curvature) < 1e-15


@unittest.skipIf(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
class SinewaveTest(unittest.TestCase):
    def setUp(self):
        n = 256

        self.hm = 0.1
        self.L = n
        self.X = np.arange(n + 1)  # n+1 because we need the endpoint
        self.sinsurf = np.sin(2 * np.pi * self.X / self.L) * self.hm

        self.precision = 5

    # def test_rms_curvature(self):
    #    numerical = NonuniformLineScan(self.X, self.sinsurf).rms_curvature()
    #    analytical = np.sqrt(16 * np.pi ** 4 * self.hm ** 2 / self.L ** 4)
    #    self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_slope_nonuniform(self):
        numerical = NonuniformLineScan(self.X, self.sinsurf).rms_slope_from_profile()
        analytical = np.sqrt(2 * np.pi ** 2 * self.hm ** 2 / self.L ** 2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height_nonuniform(self):
        numerical = NonuniformLineScan(self.X, self.sinsurf).rms_height_from_profile()
        analytical = np.sqrt(self.hm ** 2 / 2)
        # numerical = np.sqrt(np.trapz(self.sinsurf**2, self.X))

        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_slope_uniform(self):
        numerical = UniformLineScan(self.sinsurf[:-1], self.L).rms_slope_from_profile()
        analytical = np.sqrt(2 * np.pi ** 2 * self.hm ** 2 / self.L ** 2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_slope_uniform_topography(self):
        numerical = Topography(np.transpose([self.sinsurf[:-1]]*5), (self.L, 1)).rms_slope_from_profile()
        analytical = np.sqrt(2 * np.pi ** 2 * self.hm ** 2 / self.L ** 2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)


def test_rms_slope_from_profile():
    r = 4096
    res = (r,)
    for H in [0.3, 0.8]:
        for s in [(1,), (1.4,)]:
            t = fourier_synthesis(res, s, H,
                                  short_cutoff=32 / r * np.mean(s),
                                  rms_slope=0.1,
                                  amplitude_distribution=lambda n: 1.0)
            last_rms_slope = t.rms_slope_from_profile()
            np.testing.assert_almost_equal(last_rms_slope, 0.1, decimal=2)
            # rms slope should not depend on filter for these cutoffs...
            for cutoff in [1, 2, 4, 8, 16]:
                rms_slope = t.rms_slope_from_profile(short_wavelength_cutoff=s[0]/r*cutoff)
                np.testing.assert_almost_equal(rms_slope, last_rms_slope)
            # ...but starts being a monotonously decreasing function here
            for cutoff in [64, 128, 256]:
                rms_slope = t.rms_slope_from_profile(short_wavelength_cutoff=s[0]/r*cutoff)
                assert rms_slope < last_rms_slope
                last_rms_slope = rms_slope


def test_rms_slope_from_area():
    r = 2048
    res = [r, r]
    for H in [0.3, 0.8]:
        for s in [(1, 1), (1.4, 3.3)]:
            t = fourier_synthesis(res, s, H,
                                  short_cutoff=8 / r * np.mean(s),
                                  rms_slope=0.1,
                                  amplitude_distribution=lambda n: 1.0)
            last_rms_slope = t.rms_gradient()
            np.testing.assert_almost_equal(last_rms_slope, 0.1, decimal=2)
            np.testing.assert_almost_equal(last_rms_slope, t.scale(1.3).rms_gradient() / 1.3)
            np.testing.assert_almost_equal(last_rms_slope, t.scale(1.3, 1.3).rms_gradient())
            # rms slope should not depend on filter for these cutoffs...
            for cutoff in [4]:
                rms_slope = t.rms_gradient(short_wavelength_cutoff=s[0]/r*cutoff)
                np.testing.assert_almost_equal(rms_slope, last_rms_slope)
            # ...but starts being a monotonously decreasing function here
            for cutoff in [16, 32]:
                rms_slope = t.rms_gradient(short_wavelength_cutoff=s[0]/r*cutoff)
                assert rms_slope < last_rms_slope
                last_rms_slope = rms_slope
