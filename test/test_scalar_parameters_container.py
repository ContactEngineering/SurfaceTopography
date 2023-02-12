import numpy as np
import pytest

from SurfaceTopography import read_container, SurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Container.Moments import _bandwidth_count_from_profile


def test_ciso_moment_from_container(file_format_examples):
    c, = read_container(f'{file_format_examples}/container1.zip')

    unit = "m"
    varh_ciso = c.ciso_moment(order=0, unit=unit)
    varhp_ciso = c.ciso_moment(order=2, unit=unit)
    varhpp_ciso = c.ciso_moment(order=4, unit=unit)
    l, vbm = c.variable_bandwidth(unit=unit)

    error_hrms = (abs(vbm[-1] - np.sqrt(varh_ciso)) / vbm[-1])
    assert error_hrms < 0.1

    # topography with the smallest pixel size
    pixel_sizes = [min(t.pixel_size) for t in c._topographies]
    small_scale_topo = c._topographies[np.argmin(pixel_sizes)]

    hprms = small_scale_topo.to_unit(unit).rms_gradient()
    hpprms = small_scale_topo.to_unit(unit).rms_curvature_from_area()

    error_hprms = (abs(hprms - np.sqrt(varhp_ciso)) / hprms)
    assert error_hprms < 0.1

    # TODO : there is a lot of uncertainty here !
    # However that might because of the difference between Fourier derivative and finite difference derivative
    error_hpprms = (abs(hpprms - np.sqrt(varhpp_ciso)) / hpprms)
    assert error_hpprms < 1


@pytest.mark.parametrize("seed", range(3))
def test_ciso_moment_container_vs_topography(seed):
    np.random.seed(seed)
    sx, sy = physical_sizes = 2, 3
    nx, ny = nb_grid_pts = 1024, 1023

    unit = "m"
    t = fourier_synthesis(nb_grid_pts, physical_sizes=physical_sizes, hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center")

    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t, ])
    c_varh_ciso = c.ciso_moment(order=0, unit=unit, nb_points_per_decade=20)
    c_varhp_ciso = c.ciso_moment(order=2, unit=unit, nb_points_per_decade=20)
    c_varhpp_ciso = c.ciso_moment(order=4, unit=unit, nb_points_per_decade=20)

    # Moments from full integration of the 2D spectrum of the topopography
    t_varh_ciso = t.moment_power_spectrum(order=0, )
    t_varhp_ciso = t.moment_power_spectrum(order=2, )
    t_varhpp_ciso = t.moment_power_spectrum(order=4, )

    # TODO : there is a lot of uncertainty here !
    # some of the discrepancies might come from the conversion C1D to Ciso
    assert abs(1 - c_varhp_ciso / t_varhp_ciso) < 0.5
    assert abs(1 - c_varhpp_ciso / t_varhpp_ciso) < 0.5
    assert abs(1 - c_varh_ciso / t_varh_ciso) < 0.5


@pytest.mark.parametrize("seed", range(3))
def test_1d_moment_container_vs_linescan(seed):
    np.random.seed(seed)
    sx = 2
    nx = 1024

    unit = "m"
    t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center")

    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t, ])
    c_varh_ciso = c.c1d_moment(order=0, unit=unit, nb_points_per_decade=20)
    c_varhp_ciso = c.c1d_moment(order=2, unit=unit, nb_points_per_decade=20)
    c_varhpp_ciso = c.c1d_moment(order=4, unit=unit, nb_points_per_decade=20)

    # Moments from full integration of the 2D spectrum of the topopography
    t_varh_ciso = t.moment_power_spectrum(order=0, )
    t_varhp_ciso = t.moment_power_spectrum(order=2, )
    t_varhpp_ciso = t.moment_power_spectrum(order=4, )

    assert abs(1 - c_varhp_ciso / t_varhp_ciso) < 0.05
    assert abs(1 - c_varhpp_ciso / t_varhpp_ciso) < 0.05
    assert abs(1 - c_varh_ciso / t_varh_ciso) < 0.15
    # This means that the resampling procedure is not super precise for the integration


@pytest.mark.parametrize("seed", range(3))
def test_1d_moment_container_vs_linescan_integrate_psd(seed):
    np.random.seed(seed)
    sx = 2
    nx = 1024

    unit = "m"
    t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center")

    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t, ])
    c_varh_ciso = c.integrate_psd_from_profile(factor=lambda q: 1, unit=unit)
    c_varhp_ciso = c.integrate_psd_from_profile(factor=lambda q: q ** 2, unit=unit, )
    c_varhpp_ciso = c.integrate_psd_from_profile(factor=lambda q: q ** 4, unit=unit, )

    # Moments from full integration of the 2D spectrum of the topopography
    t_varh_ciso = t.moment_power_spectrum(order=0, )
    t_varhp_ciso = t.moment_power_spectrum(order=2, )
    t_varhpp_ciso = t.moment_power_spectrum(order=4, )

    assert abs(1 - c_varhp_ciso / t_varhp_ciso) < 1e-10
    assert abs(1 - c_varhpp_ciso / t_varhpp_ciso) < 1e-10
    assert abs(1 - c_varh_ciso / t_varh_ciso) < 1e-10
    # This means that the resampling procedure is not super precise for the integration


@pytest.mark.parametrize("n_topographies", [1, 3])
@pytest.mark.parametrize("seed", range(3))
def test_1d_moment_container_vs_linescan_integrate_psd_q0_mode(seed, n_topographies):
    np.random.seed(seed)
    sx = 2
    nx = 1024

    unit = "m"
    t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center").squeeze()
    # Introduce a significant offset in the heights that will affect the rms heights
    t._heights += t.rms_height_from_profile()
    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t, ] * n_topographies)
    c_varh_ciso = c.integrate_psd_from_profile(factor=lambda q: 1, unit=unit)

    # Moments from full integration of the 2D spectrum of the topopography
    t_varh_ciso = t.moment_power_spectrum(order=0, )
    assert abs(1 - c_varh_ciso / t_varh_ciso) < 1e-10
    # This means that the resampling procedure is not super precise for the integration


@pytest.mark.parametrize("seed", range(3))
def test_integrate_psd_nb_topographies(seed):
    np.random.seed(seed)
    sx = 2
    nx = 1024

    unit = "m"

    t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center")

    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t] * 3)
    c_varh_ciso = c.integrate_psd_from_profile(factor=lambda q: 1, unit=unit)
    c_varhp_ciso = c.integrate_psd_from_profile(factor=lambda q: q ** 2, unit=unit, )
    c_varhpp_ciso = c.integrate_psd_from_profile(factor=lambda q: q ** 4, unit=unit, )

    # Moments from full integration of the 2D spectrum of the topopography
    t_varh_ciso = t.moment_power_spectrum(order=0, )
    t_varhp_ciso = t.moment_power_spectrum(order=2, )
    t_varhpp_ciso = t.moment_power_spectrum(order=4, )

    assert abs(1 - c_varhp_ciso / t_varhp_ciso) < 1e-10
    assert abs(1 - c_varhpp_ciso / t_varhpp_ciso) < 1e-10
    assert abs(1 - c_varh_ciso / t_varh_ciso) < 1e-10
    # This means that the resampling procedure is not super precise for the integration


def test_bandwidth_count():
    sx = 2
    nx = 1024

    unit = "m"

    t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, rms_height=1,
                          short_cutoff=4 * (sx / nx),
                          long_cutoff=sx / 8, unit=unit).detrend(detrend_mode="center")

    # Moment of the isotropic PSD computed from the 1D power spectrum
    c = SurfaceContainer([t] * 3)

    qmin = 2 * np.pi / sx
    assert _bandwidth_count_from_profile(c, qmin, unit=unit) == 3

    assert _bandwidth_count_from_profile(c, qmin * 2, unit=unit) == 3

    qmax = np.pi / (sx / nx)
    assert _bandwidth_count_from_profile(c, qmax, unit=unit) == 3
    assert _bandwidth_count_from_profile(c, 2 * qmax, unit=unit) == 0
    assert _bandwidth_count_from_profile(c, 0.5 * qmin, unit=unit) == 0


@pytest.mark.parametrize("seed", range(10))
def test_integrate_psd_different_bandwidths(seed):
    np.random.seed(seed)
    sx = 2
    nx = 1024 * 16
    long_cutoff = sx / 64
    unit = "m"

    t_varh_c1d = []
    t_varhp_c1d = []
    t_varhpp_c1d = []

    # reference a large scan encompassing the whole PSD
    # need to average the results to get rid of the fluctuations
    n_av = 30
    for i in range(n_av):
        t = fourier_synthesis((nx,), physical_sizes=(sx,), hurst=0.8, c0=1,
                              short_cutoff=4 * (sx / nx),
                              long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center")

        # Moments from full integration of the 2D spectrum of the large topopography
        t_varh_c1d += [t.moment_power_spectrum(order=0, )]
        t_varhp_c1d += [t.moment_power_spectrum(order=2, )]
        t_varhpp_c1d += [t.moment_power_spectrum(order=4, )]

    t_varh_c1d_var = np.var(t_varh_c1d)
    t_varhp_c1d_var = np.var(t_varhp_c1d)
    t_varhpp_c1d_var = np.var(t_varhpp_c1d)

    t_varh_c1d_mean = np.mean(t_varh_c1d)
    t_varhp_c1d_mean = np.mean(t_varhp_c1d)
    t_varhpp_c1d_mean = np.mean(t_varhpp_c1d)

    print("relative fluctuations of var h", np.sqrt(t_varh_c1d_var) / t_varh_c1d_mean)
    print("relative fluctuations of var hp", np.sqrt(t_varhp_c1d_var) / t_varhp_c1d_mean)
    print("relative fluctuations of var hpp", np.sqrt(t_varhpp_c1d_var) / t_varhpp_c1d_mean)
    # create smaller topographies with same nominal PSD

    ts = []
    for i in range(10):
        ts += [fourier_synthesis((1024,), physical_sizes=(sx,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               fourier_synthesis((512,), physical_sizes=(sx,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               fourier_synthesis((2048,), physical_sizes=(sx,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               fourier_synthesis((1024,), physical_sizes=(sx / 4,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               fourier_synthesis((2048,), physical_sizes=(sx / 8,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               fourier_synthesis((512,), physical_sizes=(sx / 32,), hurst=0.8, c0=1,
                                 short_cutoff=4 * (sx / nx),
                                 long_cutoff=long_cutoff, unit=unit).detrend(detrend_mode="center"),
               ]

    c = SurfaceContainer(ts)
    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(*t.power_spectrum_from_profile(0), "k")

        for _t in c._topographies:
            ax.loglog(*_t.power_spectrum_from_profile(0))

        fig, ax = plt.subplots()
        q = np.logspace(np.log(2 * np.pi / sx), np.log(np.pi / (sx / nx)))
        ax.plot(q, _bandwidth_count_from_profile(c, q, unit, ))
        ax.set_xscale("log")
        plt.show(block=True)
    # Moment of the isotropic PSD computed from the 1D power spectrum

    c_varh_c1d = c.integrate_psd_from_profile(factor=lambda q: 1, unit=unit)
    c_varhp_c1d = c.integrate_psd_from_profile(factor=lambda q: q ** 2, unit=unit, )
    c_varhpp_c1d = c.integrate_psd_from_profile(factor=lambda q: q ** 4, unit=unit, )

    assert abs(1 - c_varhp_c1d / t_varhp_c1d_mean) < 0.1
    assert abs(1 - c_varhpp_c1d / t_varhpp_c1d_mean) < 0.1
    assert abs(1 - c_varh_c1d / t_varh_c1d_mean) < 0.1
    # The tolerance is not that small but it is understandable since t_varhp_c1d still has a nonneglidgible variance.
