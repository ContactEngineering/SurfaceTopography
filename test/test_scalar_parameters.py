#
# Copyright 2018-2021, 2023-2024 Lars Pastewka
#           2018-2021, 2023 Antoine Sanner
#           2019, 2021 Michael Röttger
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
import unittest

import numpy as np
import pytest
from muFFT import FFT
from NuMPI import MPI

from SurfaceTopography import (NonuniformLineScan, Topography, UniformLineScan,
                               read_published_container, read_topography)
from SurfaceTopography.Container.SurfaceContainer import \
    InMemorySurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis

# import necessary to get tip artefact emulation function
# import Sura  # noqa: F401


def sinewave2D(comm=None):
    n = 256
    X, Y = np.mgrid[slice(0, n), slice(0, n)]

    fftengine = FFT((n, n), engine="mpi", communicator=comm)

    hm = 0.1
    L = float(n)
    sinsurf = np.sin(2 * np.pi / L * X) * np.sin(2 * np.pi / L * Y) * hm
    size = (L, L)

    top = Topography(sinsurf, decomposition='domain',
                     nb_subdomain_grid_pts=fftengine.nb_subdomain_grid_pts,
                     subdomain_locations=fftengine.subdomain_locations,
                     physical_sizes=size, communicator=MPI.COMM_SELF if comm is None else comm)

    return L, hm, top


def test_rms_curvature(comm):
    L, hm, top = sinewave2D(comm)
    numerical = top.rms_curvature_from_area()
    analytical = np.sqrt(4 * (16 * np.pi ** 4 / L ** 4) * hm ** 2 / 4 / 4)
    #                 rms(∆)^2 = (qx^2 + qy^2)^2 * hm^2 / 4
    # print(numerical-analytical)
    np.testing.assert_almost_equal(numerical, analytical, 5)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_slope():
    L, hm, top = sinewave2D()
    numerical = top.rms_gradient()
    analytical = np.sqrt(2 * np.pi ** 2 * hm ** 2 / L ** 2)
    # print(numerical-analytical)
    np.testing.assert_almost_equal(numerical, analytical, 5)


def test_rms_height(comm):
    L, hm, top = sinewave2D(comm)
    numerical = top.rms_height_from_area()
    analytical = np.sqrt(hm ** 2 / 4)

    assert numerical == analytical


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_mad_height(comm, plot=False):
    L, hm, top = sinewave2D(comm)
    numerical = top.mad_height()
    analytical = np.sqrt(hm ** 2 / 4)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        x, y, h = top.positions_and_heights()
        plt.pcolormesh(h)
        plt.show()
        plt.figure()
        h = np.linspace(top.min()-hm/10, top.max()+hm/10, 101)
        plt.plot(h, top.bearing_area(h), 'k-')
        plt.show()

    assert numerical > analytical  # Fixme! Derive exact analytic expression


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

    np.testing.assert_almost_equal(surf.rms_curvature_from_area(), analytical_lapl / 2, precision)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_rms_curvature_paraboloid_uniform_1D():
    n = 16
    x = np.arange(n)
    curvature = 0.1
    heights = 0.5 * curvature * x ** 2

    surf = UniformLineScan(heights, physical_sizes=(n,), periodic=False)
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
        t = NonuniformLineScan(self.X, self.sinsurf)
        numerical = t.rms_height_from_profile()
        analytical = np.sqrt(self.hm ** 2 / 2)
        # numerical = np.sqrt(np.trapz(self.sinsurf**2, self.X))

        self.assertAlmostEqual(numerical, analytical, self.precision)

        numerical = np.sqrt(t.moment(2))

        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_slope_uniform(self):
        numerical = UniformLineScan(self.sinsurf[:-1], self.L).rms_slope_from_profile()
        analytical = np.sqrt(2 * np.pi ** 2 * self.hm ** 2 / self.L ** 2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_slope_uniform_topography(self):
        numerical = Topography(np.transpose([self.sinsurf[:-1]] * 5), (self.L, 1)).rms_slope_from_profile()
        analytical = np.sqrt(2 * np.pi ** 2 * self.hm ** 2 / self.L ** 2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
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
                rms_slope = t.rms_slope_from_profile(short_wavelength_cutoff=s[0] / r * cutoff)
                np.testing.assert_almost_equal(rms_slope, last_rms_slope)
            # ...but starts being a monotonously decreasing function here
            for cutoff in [64, 128, 256]:
                rms_slope = t.rms_slope_from_profile(short_wavelength_cutoff=s[0] / r * cutoff)
                assert rms_slope < last_rms_slope
                last_rms_slope = rms_slope


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
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
                rms_slope = t.rms_gradient(short_wavelength_cutoff=s[0] / r * cutoff)
                np.testing.assert_almost_equal(rms_slope, last_rms_slope)
            # ...but starts being a monotonously decreasing function here
            for cutoff in [16, 32]:
                rms_slope = t.rms_gradient(short_wavelength_cutoff=s[0] / r * cutoff)
                assert rms_slope < last_rms_slope
                last_rms_slope = rms_slope


def test_rms_height_with_undefined_data(file_format_examples):
    t = read_topography(os.path.join(file_format_examples, 'opd-3.opd'))
    assert t.has_undefined_data
    np.testing.assert_allclose(t.rms_height_from_profile(), 0.011449207840819613)
    np.testing.assert_allclose(t.rms_height_from_area(), 0.011782116700392838)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("nx", (255, 256), )
def test_moment_0_1d(nx, ):
    sx = 1
    t = fourier_synthesis((nx,), (sx,), hurst=0.8, rms_height=1, short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 4).detrend(detrend_mode="center")
    hrms_r = t.rms_height_from_profile()
    hrms_f = np.sqrt(t.moment_power_spectrum())

    assert abs(1 - hrms_r / hrms_f) < 0.001

    hrms_f = np.sqrt(t.integrate_psd(lambda qx: 1))
    assert abs(1 - hrms_r / hrms_f) < 0.001

    hrms_f = np.sqrt(t.integrate_psd_from_profile(lambda qx: 1))
    assert abs(1 - hrms_r / hrms_f) < 0.001


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("nx", (255, 256), )
def test_moment_1_2d(nx, ):
    sx = 1
    t = fourier_synthesis((nx,), (sx,), hurst=0.8, rms_height=1, short_cutoff=16 * (sx / nx),
                          long_cutoff=sx / 4).detrend(detrend_mode="center")
    hrms_r = t.rms_slope_from_profile()
    hrms_f = np.sqrt(t.moment_power_spectrum(order=2))

    assert abs(1 - hrms_r / hrms_f) < 0.01

    hrms_f = np.sqrt(t.integrate_psd(lambda qx: qx ** 2))
    assert abs(1 - hrms_r / hrms_f) < 0.01

    hrms_f = np.sqrt(t.integrate_psd_from_profile(lambda qx: qx ** 2))
    assert abs(1 - hrms_r / hrms_f) < 0.01


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("nb_grid_pts", ((255, 256), (256, 255), (256, 256), (255, 255)))
@pytest.mark.parametrize("physical_sizes", ((1, 1), (1, 2)))
def test_moment_2_2d(nb_grid_pts, physical_sizes):
    sx, sy = physical_sizes
    nx, ny = nb_grid_pts
    t = fourier_synthesis(nb_grid_pts, physical_sizes=physical_sizes, hurst=0.8, rms_height=1,
                          short_cutoff=16 * (sx / nx),
                          long_cutoff=sx / 4).detrend(detrend_mode="center")
    hrms_r = t.rms_gradient()
    hrms_f = np.sqrt(t.moment_power_spectrum(order=2))

    assert abs(1 - hrms_r / hrms_f) < 0.01

    hrms_f = np.sqrt(t.integrate_psd(lambda qx, qy: qx ** 2 + qy ** 2))
    assert abs(1 - hrms_r / hrms_f) < 0.01


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("nb_grid_pts", ((255, 256), (256, 255), (256, 256), (255, 255)))
@pytest.mark.parametrize("physical_sizes", ((1, 1), (1, 2)))
def test_moment_0_2d(nb_grid_pts, physical_sizes):
    sx, sy = physical_sizes
    nx, ny = nb_grid_pts
    t = fourier_synthesis(nb_grid_pts, physical_sizes=physical_sizes, hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 4).detrend(detrend_mode="center")
    hrms_r = t.rms_height_from_area()
    hrms_f = np.sqrt(t.moment_power_spectrum())

    assert abs(1 - hrms_r / hrms_f) < 0.001

    hrms_f = np.sqrt(t.integrate_psd(lambda qx, qy: 1))
    assert abs(1 - hrms_r / hrms_f) < 0.001


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("nb_grid_pts", ((255, 256), (256, 255), (256, 256), (255, 255)))
@pytest.mark.parametrize("physical_sizes", ((1, 1), (1, 2)))
def test_integrate_psd_from_profile_2d(nb_grid_pts, physical_sizes):
    sx, sy = physical_sizes
    nx, ny = nb_grid_pts
    t = fourier_synthesis(nb_grid_pts, physical_sizes=physical_sizes, hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 4).detrend(detrend_mode="center")
    hrms_r = t.rms_height_from_area()
    hrms_f = np.sqrt(t.integrate_psd_from_profile(lambda qx: 1))
    assert abs(1 - hrms_r / hrms_f) < 0.001

    hrms_r = t.rms_slope_from_profile()
    dx = sx / nx
    #                     finite difference stencil in fourier space   -v
    hrms_f = np.sqrt(t.integrate_psd_from_profile(lambda qx: np.abs((np.exp(1j * qx * dx) - 1) / dx) ** 2))
    assert abs(1 - hrms_r / hrms_f) < 0.001

    hrms_f = np.sqrt(t.integrate_psd(lambda qx, qy: np.abs((np.exp(1j * qx * dx) - 1) / dx) ** 2))
    assert abs(1 - hrms_r / hrms_f) < 0.001


@pytest.mark.parametrize("seed", range(4))
def test_integrate_psd_remove_tip_artefacts_profile(seed):
    """
    Makes sure that the tip radius removal is actually applied when integrating the psd

    Also tests that it is still applied when the topography is inside a container.
    """
    np.random.seed(seed)
    unit = "m"
    nx = 16384
    sx = 1

    t = fourier_synthesis((nx,), (sx,), hurst=0.8, rms_slope=0.1, short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 4, unit=unit).detrend(detrend_mode="center")

    R = 1 / t.rms_curvature_from_profile() * 10

    t_artefacted = t.scan_with_rigid_sphere(R)

    plot = False

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(*t.positions_and_heights(), c="k")
        ax.plot(*t_artefacted.positions_and_heights(), c="cyan")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        def func(dx, dy=None):
            return -np.min(dx)

        l, c = t.scale_dependent_statistical_property(
            n=2, func=func, reliable=True)
        ax.loglog(l, c, "+k", label="original")
        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            l, c = t_artefacted.scale_dependent_statistical_property(
                n=2, func=func, reliable=reliable)
            ax.loglog(l, c, ".", c=color, label=label)

        ax.legend()

        ax.axhline(1 / R)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*t.power_spectrum_from_profile(resampling_method=None), "+k", label="original")
        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.power_spectrum_from_profile(reliable=reliable, resampling_method=None), ".",
                      c=color, label=label)
        ax.legend()

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*t.autocorrelation_from_profile(resampling_method=None), "+k", label="original")

        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.autocorrelation_from_profile(reliable=reliable, resampling_method=None), ".",
                      c=color, label=label)
        ax.legend()

    hrms_r = [t.rms_height_from_profile(), t.rms_slope_from_profile(), t.rms_curvature_from_profile()]
    # hrms_artefacted = [t_artefacted.rms_slope_from_profile(),
    #                    t_artefacted.rms_slope_from_profile(),
    #                    t_artefacted.rms_curvature_from_profile()]

    hrms_tip_artefacts_removed = [
        np.max(t_artefacted.autocorrelation_from_profile(reliable=True, resampling_method=None)[1]),
        np.max(t_artefacted.scale_dependent_slope_from_profile(reliable=True, resampling_method=None)[1]),
        np.max(t_artefacted.scale_dependent_curvature_from_profile(reliable=True, resampling_method=None)[1]),
    ]

    # dx = sx / nx
    # hrms_ffd_unreliable, hrms_ffd_reliable = [
    #     [
    #         np.sqrt(t_artefacted.integrate_psd_from_profile(fun, reliable=reliable)) for fun in [
    #         lambda qx: 1,
    #         lambda qx: np.abs((np.exp(1j * qx * dx) - 1) / dx) ** 2,
    #         lambda qx: np.abs((np.exp(1j * qx * dx) - 2 + np.exp(-1j * qx * dx)) / dx ** 2) ** 2
    #     ]
    #     ] for reliable in [False, True]
    #
    # ]

    hrms_f_unreliable, hrms_f_reliable = [
        [
            np.sqrt(t_artefacted.integrate_psd_from_profile(fun, reliable=reliable)) for fun in [
                lambda qx: 1,
                lambda qx: qx ** 2,
                lambda qx: qx ** 4
            ]
        ] for reliable in [False, True]
    ]

    c_artefacted = InMemorySurfaceContainer([t_artefacted, ])
    c_hrms_f_unreliable, c_hrms_f_reliable = [
        [
            np.sqrt(c_artefacted.integrate_psd_from_profile(fun, reliable=reliable, unit=unit)) for fun in [
                lambda qx: 1,
                lambda qx: qx ** 2,
                lambda qx: qx ** 4
            ]
        ] for reliable in [False, True]
    ]

    np.testing.assert_allclose(c_hrms_f_unreliable, hrms_f_unreliable)
    np.testing.assert_allclose(c_hrms_f_reliable, hrms_f_reliable)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(*t.scale_dependent_curvature_from_profile(resampling_method=None), "+k", label="original")

        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.scale_dependent_curvature_from_profile(reliable=reliable, resampling_method=None),
                      ".", c=color, label=label)
        ax.axhline(hrms_f_unreliable[2])
        ax.axhline(hrms_f_reliable[2])

        ax.legend()
        plt.show(block=True)

    for derivative in [1, 2]:
        assert hrms_f_unreliable[derivative] < hrms_r[derivative]

    for derivative in [0, 1, 2]:
        assert hrms_f_reliable[derivative] < hrms_f_unreliable[derivative]

    # Makes sure there is a significant difference by removing tip artefacts, so we are doing a meaningful test
    assert hrms_r[1] / hrms_tip_artefacts_removed[1] > 1.5
    # now we test that we indeed removed the tip artefacts when integrating the PSD
    assert abs(hrms_f_reliable[1] / hrms_tip_artefacts_removed[1] - 1) < 0.2

    # Makes sure there is a significant difference by removing tip artefacts, so we are doing a meaningful test
    assert hrms_r[2] / hrms_tip_artefacts_removed[2] > 4
    # now we test that we indeed removed the tip artefacts when integrating the PSD
    assert abs(hrms_f_reliable[2] / hrms_tip_artefacts_removed[2] - 1) < 0.2


def test_integrate_psd_from_profile_remove_tip_artefacts_areal_scan():
    data_container = read_published_container("https://doi.org/10.57703/ce-v9qwe")[0].read_all()

    for i in [0, 1]:  # workaround for wrong height scale factor
        data_container._topographies[i] = data_container._topographies[i].scale(1e-3).squeeze()
        assert data_container._topographies[i].rms_height_from_area() < 0.01
        assert data_container._topographies[i].rms_height_from_area() > 0.0005

    # workaround for tip radius not specified
    data_container._topographies[1]._info.update(dict(instrument=dict(name="Scanning rigid sphere simulation",
                                                                      parameters=dict(
                                                                          tip_radius=dict(value=40, unit="nm")))))

    # %% tags=[]
    plot = False
    if plot:
        import matplotlib.pyplot as plt

        data_container._topographies[0].plot()

        # %%
        data_container._topographies[1].plot()

        # %%

        plt.plot(np.concatenate([data_container._topographies[1].heights()[0, :]] * 2))
        plt.plot(np.concatenate([data_container._topographies[0].heights()[0, :]] * 2))
        ax = plt.gca()

        # %%

        ax.set_xlim(1000, 1100)
        ax.figure

        # %% [markdown]
        # The two plot seem to be indeed perfectly periodic

    # %%
    t = data_container._topographies[0]
    t_artefacted = data_container._topographies[1]

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        def func(dx, dy=None):
            return -np.min(dx)

        l, c = t.scale_dependent_statistical_property(
            n=2, func=func, reliable=True)
        ax.loglog(l, c, "+k", label="original")
        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            l, c = t_artefacted.scale_dependent_statistical_property(
                n=2, func=func, reliable=reliable)
            ax.loglog(l, c, ".", c=color, label=label)

        ax.legend()
        # When the topographies are well interpolated
        # below the short cutoff the scanned topography
        # has a higher rms height because of the
        # cusps introdyced by the tip artefacts

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*t.power_spectrum_from_profile(resampling_method=None), "+k", label="original")
        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.power_spectrum_from_profile(reliable=reliable, resampling_method=None), ".",
                      c=color,
                      label=label)
        ax.legend()

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*t.scale_dependent_slope_from_profile(resampling_method=None), "+k", label="original")

        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.scale_dependent_slope_from_profile(reliable=reliable, resampling_method=None), ".",
                      c=color, label=label)
        ax.legend()

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*t.scale_dependent_curvature_from_profile(resampling_method=None), "+k", label="original")

        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.scale_dependent_curvature_from_profile(reliable=reliable, resampling_method=None),
                      ".",
                      c=color, label=label)
        ax.legend()

    # %%

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*t.autocorrelation_from_profile(resampling_method=None), "+k", label="original")

        for reliable, color, label in [[False, "cyan", "scanned all"], [True, "k", "scann reliable"]]:
            ax.loglog(*t_artefacted.autocorrelation_from_profile(reliable=reliable, resampling_method=None), ".",
                      c=color,
                      label=label)
        ax.legend()

    if plot:
        plt.show()
    # %%
    hrms_r = [t.rms_height_from_area(), t.rms_slope_from_profile(), t.rms_curvature_from_area()]

    # %%
    # hrms_artefacted = [t_artefacted.rms_height_from_area(),
    #                    t_artefacted.rms_gradient() / np.sqrt(2),
    #                    t_artefacted.rms_curvature_from_area()]

    hrms_tip_artefacts_removed = [
        np.max(t_artefacted.autocorrelation_from_profile(reliable=True, resampling_method=None)[1]),
        np.max(t_artefacted.scale_dependent_slope_from_profile(reliable=True, resampling_method=None)[1]),
        np.max(t_artefacted.scale_dependent_curvature_from_profile(reliable=True, resampling_method=None)[1]),
    ]

    hrms_f_unreliable, hrms_f_reliable = [
        [
            np.sqrt(t_artefacted.integrate_psd_from_profile(fun, reliable=reliable)) for fun in [
                lambda qx: 1,
                lambda qx: qx ** 2,
                lambda qx: qx ** 4
            ]
        ] for reliable in [False, True]
    ]

    for derivative in [0, 1]:
        assert hrms_f_unreliable[derivative] < hrms_r[derivative]

    for derivative in [0, 1]:
        assert hrms_f_reliable[derivative] < hrms_f_unreliable[derivative]

    # Because of the well resolved cusps introduced by the tip artefacts
    assert hrms_f_reliable[2] < hrms_f_unreliable[2]

    # %%
    # Makes sure there is a significant difference by removing tip artefacts, so we are doing a meaningful test
    assert hrms_r[1] / hrms_tip_artefacts_removed[1] > 1.5
    # now we test that we indeed removed the tip artefacts when integrating the PSD
    assert abs(hrms_f_reliable[1] / hrms_tip_artefacts_removed[1] - 1) < 0.2

    # Makes sure there is a significant difference by removing tip artefacts, so we are doing a meaningful test
    assert hrms_r[2] / hrms_tip_artefacts_removed[2] > 4
    # now we test that we indeed removed the tip artefacts when integrating the PSD,
    # i.e. that reliable PSD integration is approx equivalent to reliable SDRPs
    assert abs(hrms_f_reliable[2] / hrms_tip_artefacts_removed[2] - 1) < 0.2

    # Assert the container gives the same results:
    c_artefacted = InMemorySurfaceContainer([t_artefacted, ])
    c_hrms_f_unreliable, c_hrms_f_reliable = [
        [
            np.sqrt(c_artefacted.integrate_psd_from_profile(fun, reliable=reliable, unit=t_artefacted.unit)) for fun in
            [
                lambda qx: 1,
                lambda qx: qx ** 2,
                lambda qx: qx ** 4
            ]
        ] for reliable in [False, True]
    ]

    np.testing.assert_allclose(c_hrms_f_unreliable, hrms_f_unreliable)
    np.testing.assert_allclose(c_hrms_f_reliable, hrms_f_reliable)
