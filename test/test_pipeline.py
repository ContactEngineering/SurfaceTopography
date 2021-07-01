"""
Tests of the filter and modification pipeline
"""

import os
import pytest

import numpy as np

from NuMPI.Tools import Reduction

from SurfaceTopography import read_topography
from SurfaceTopography.UniformLineScanAndTopography import Topography, \
    DetrendedUniformTopography, UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.IO.Text import read_xyz


def test_translate(comm_self):
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.))

    assert (topography.translate(offset=(1, 0)).heights()
            ==
            np.array([[0, 0, 0],
                      [0, 1, 0]])).all()

    assert (topography.translate(offset=(2, 0)).heights()
            ==
            np.array([[0, 1, 0],
                      [0, 0, 0]])).all()

    assert (topography.translate(offset=(0, -1)).heights()
            ==
            np.array([[1, 0, 0],
                      [0, 0, 0]])).all()


def test_pipeline():
    t1 = fourier_synthesis((511, 511), (1., 1.), 0.8, rms_height=1)
    t2 = t1.detrend()
    p = t2.pipeline()
    assert isinstance(p[0], Topography)
    assert isinstance(p[1], DetrendedUniformTopography)


def test_uniform_detrended_periodicity():
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.), periodic=True)
    assert topography.detrend("center").is_periodic
    assert not topography.detrend("height").is_periodic
    assert not topography.detrend("curvature").is_periodic


def test_passing_of_docstring():
    from SurfaceTopography.Uniform.PowerSpectrum import power_spectrum_from_profile
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.), periodic=True)
    assert topography.power_spectrum_from_profile.__doc__ == power_spectrum_from_profile.__doc__


@pytest.mark.parametrize("periodic", (True, False))
def test_fill_undefined_data_linescan(periodic):
    topography = UniformLineScan(np.array([1, np.nan, 4.]),
                                 (1.,),
                                 info=dict(test=1),
                                 periodic=periodic,
                                 )
    assert topography.has_undefined_data

    filled_topography = topography.fill_undefined_data(fill_value=-np.infty)
    assert not filled_topography.has_undefined_data

    assert filled_topography.physical_sizes == topography.physical_sizes
    assert filled_topography.is_periodic == topography.is_periodic
    assert filled_topography.info["test"] == 1


@pytest.mark.parametrize("periodic", (True, False))
def test_fill_undefined_data(periodic):
    topography = Topography(np.array([[1, np.nan, 4.],
                                      [3, 4, 5]]),
                            (1., 1.),
                            info=dict(test=1),
                            periodic=periodic,
                            )
    assert topography.has_undefined_data

    filled_topography = topography.fill_undefined_data(fill_value=-np.infty)
    mask = np.ma.getmask(topography.heights())
    nmask = np.logical_not(mask)
    assert (filled_topography[nmask] == topography[nmask]).all()
    assert (filled_topography[mask] == - np.infty).all()
    assert not filled_topography.has_undefined_data

    assert filled_topography.physical_sizes == topography.physical_sizes
    assert filled_topography.is_periodic == topography.is_periodic
    assert filled_topography.info["test"] == 1


def test_fill_undefined_data_parallel(comm):
    np.random.seed(comm.rank)
    local_data = np.random.uniform(size=(3, 1))
    local_data[local_data > 0.9] = np.nan
    if comm.rank == 0:  # make sure we always have undefined data
        local_data[0, 0] = np.nan
    topography = Topography(local_data,
                            (1., 1.),
                            info=dict(test=1),
                            communicator=comm,
                            decomposition="subdomain",
                            nb_grid_pts=(3, comm.size),
                            subdomain_locations=(0, comm.rank)
                            )

    filled_topography = topography.fill_undefined_data(fill_value=-np.infty)
    assert topography.has_undefined_data
    assert not filled_topography.has_undefined_data

    mask = np.ma.getmask(topography.heights())
    nmask = np.logical_not(mask)

    reduction = Reduction(comm)

    assert reduction.all(filled_topography[nmask] == topography[nmask])
    assert reduction.all(filled_topography[mask] == - np.infty)
    assert not filled_topography.has_undefined_data


def test_uniform_scaled_topography():
    surf = fourier_synthesis((5, 7), (1.2, 1.1), 0.8, rms_height=1)
    sx, sy = surf.physical_sizes
    for fac in [1.0, 2.0, np.pi]:
        surf2 = surf.scale(fac)
        np.testing.assert_almost_equal(fac * surf.rms_height_from_profile(),
                                       surf2.rms_height_from_profile())
        np.testing.assert_almost_equal(surf.positions(), surf2.positions())

        surf2 = surf.scale(fac, 2 * fac)
        np.testing.assert_almost_equal(fac * surf.rms_height_from_profile(),
                                       surf2.rms_height_from_profile())

        sx2, sy2 = surf2.physical_sizes
        np.testing.assert_almost_equal(2 * fac * sx, sx2)
        np.testing.assert_almost_equal(2 * fac * sy, sy2)

        x, y = surf.positions()
        x2, y2 = surf2.positions()
        np.testing.assert_almost_equal(2 * fac * x, x2)
        np.testing.assert_almost_equal(2 * fac * y, y2)

        x2, y2, h2 = surf2.positions_and_heights()
        np.testing.assert_almost_equal(2 * fac * x, x2)
        np.testing.assert_almost_equal(2 * fac * y, y2)


def test_uniform_unit_conversion():
    surf = fourier_synthesis((5, 7), (1.2, 1.1), 0.8, rms_height=1, unit='um')
    assert surf.unit == 'um'

    surf2 = surf.to_unit('mm')
    assert surf2.info['unit'] == 'mm'
    assert surf2.unit == 'mm'

    np.testing.assert_almost_equal(tuple(p / 1000 for p in surf.physical_sizes), surf2.physical_sizes)
    np.testing.assert_almost_equal(tuple(p / 1000 for p in surf.pixel_size), surf2.pixel_size)
    np.testing.assert_almost_equal(tuple(p / 1000 for p in surf.positions()), surf2.positions())
    np.testing.assert_almost_equal(surf.rms_height_from_area() / 1000, surf2.rms_height_from_area())
    np.testing.assert_almost_equal(surf.rms_gradient(), surf2.rms_gradient())
    np.testing.assert_almost_equal(surf.rms_laplacian() * 1000, surf2.rms_laplacian())


def test_nonuniform_scaled_topography(file_format_examples):
    surf = read_xyz(os.path.join(file_format_examples, 'example.xyz'))
    sx, = surf.physical_sizes
    for fac in [1.0, 2.0, np.pi]:
        surf2 = surf.scale(fac)
        np.testing.assert_almost_equal(fac * surf.rms_height_from_profile(),
                                       surf2.rms_height_from_profile())
        np.testing.assert_almost_equal(surf.positions(), surf2.positions())

        surf2 = surf.scale(fac, 2*fac)
        np.testing.assert_almost_equal(fac * surf.rms_height_from_profile(),
                                       surf2.rms_height_from_profile())
        np.testing.assert_almost_equal(2 * fac * surf.positions(), surf2.positions())

        sx2, = surf2.physical_sizes
        np.testing.assert_almost_equal(2 * fac * sx, sx2)

        x = surf.positions()
        x2 = surf2.positions()
        np.testing.assert_almost_equal(2 * fac * x, x2)

        x2, h2 = surf2.positions_and_heights()
        np.testing.assert_almost_equal(2 * fac * x, x2)


def test_nonuniform_unit_conversion(file_format_examples):
    surf = read_xyz(os.path.join(file_format_examples, 'example.xyz'), unit='um')
    assert surf.unit == 'um'

    surf2 = surf.to_unit('mm')
    assert surf2.info['unit'] == 'mm'
    assert surf2.unit == 'mm'

    np.testing.assert_almost_equal(tuple(p / 1000 for p in surf.physical_sizes), surf2.physical_sizes)
    np.testing.assert_almost_equal(tuple(p / 1000 for p in surf.positions()), surf2.positions())
    np.testing.assert_almost_equal(surf.rms_height_from_profile() / 1000, surf2.rms_height_from_profile())
    np.testing.assert_almost_equal(surf.rms_slope_from_profile(), surf2.rms_slope_from_profile())
    np.testing.assert_almost_equal(surf.rms_curvature_from_profile() * 1000, surf2.rms_curvature_from_profile())


def test_transposed_topography():
    surf = fourier_synthesis([124, 368], [6, 3], 0.8, rms_slope=0.1)
    nx, ny = surf.nb_grid_pts
    sx, sy = surf.physical_sizes
    surf2 = surf.transpose()
    nx2, ny2 = surf2.nb_grid_pts
    sx2, sy2 = surf2.physical_sizes
    assert nx == ny2
    assert ny == nx2
    assert sx == sy2
    assert sy == sx2
    assert (surf.heights() == surf2.heights().T).all()


def test_scanning_probe_reliability_cutoff(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di1.di'))
    np.testing.assert_allclose(surf.scanning_probe_reliability_cutoff(40), 91.79698634551458)
