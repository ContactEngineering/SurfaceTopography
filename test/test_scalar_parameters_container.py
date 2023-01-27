import numpy as np
import pytest

from SurfaceTopography import read_container, SurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis


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
