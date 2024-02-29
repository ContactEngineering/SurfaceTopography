#
# Copyright 2020-2021, 2023-2024 Lars Pastewka
#           2023-2024 Antoine Sanner
#           2021 Michael Röttger
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
Tests of the filter and modification pipeline
"""

import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.IO import XYZReader
from SurfaceTopography.Pipeline import pipeline_function
from SurfaceTopography.Uniform.Detrending import DetrendedUniformTopography
from SurfaceTopography.UniformLineScanAndTopography import (
    DecoratedUniformTopography, Topography, UniformLineScan,
    UniformTopographyInterface)

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_translate():
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.), periodic=True)

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


def test_nonperiodic_not_allowed():
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.),)
    with pytest.raises(ValueError):
        topography.translate(offset=(1, 0))


def test_translate_setter():
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.), periodic=True)

    translated_t = topography.translate()

    translated_t.offset = (1, 0)
    assert (translated_t.heights()
            ==
            np.array([[0, 0, 0],
                      [0, 1, 0]])).all()
    topography = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                            physical_sizes=(4., 3.), periodic=True)
    assert topography.translate(offset=(1, 0)).is_periodic


def test_superpose():
    topography_a = Topography(np.array([[0, 1, 0], [0, 0, 0]]),
                              physical_sizes=(4., 3.))

    topography_b = Topography(np.array([[1, 1, 0], [0, 0, 1]]),
                              physical_sizes=(4., 3.))

    topography_c = topography_a.superpose(topography_b)

    assert (topography_c.heights() == np.array([[1, 2, 0], [0, 0, 1]])).all()


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
    from SurfaceTopography.Uniform.PowerSpectrum import \
        power_spectrum_from_profile
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
    np.testing.assert_almost_equal(surf.area_per_pt / 1000 ** 2, surf2.area_per_pt)
    np.testing.assert_almost_equal(surf.rms_height_from_area() / 1000, surf2.rms_height_from_area())
    np.testing.assert_almost_equal(surf.rms_gradient(), surf2.rms_gradient())
    np.testing.assert_almost_equal(surf.rms_laplacian() * 1000, surf2.rms_laplacian())


def test_nonuniform_scaled_topography(file_format_examples):
    surf = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography()
    sx, = surf.physical_sizes
    for fac in [1.0, 2.0, np.pi]:
        surf2 = surf.scale(fac)
        np.testing.assert_almost_equal(fac * surf.rms_height_from_profile(),
                                       surf2.rms_height_from_profile())
        np.testing.assert_almost_equal(surf.positions(), surf2.positions())

        surf2 = surf.scale(fac, 2 * fac)
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
    surf = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography(unit='um')
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


def test_undefined_data_and_squeeze():
    nx, ny = 128, 128
    sx, sy = 5.0, 5.0
    rx = np.linspace(-sx / 2, sx / 2, nx)
    ry = np.linspace(-sy / 2, sy / 2, ny)
    rsq = rx.reshape((nx, -1)) ** 2 + ry.reshape((-1, ny)) ** 2
    rs = 1.0
    t = Topography(np.ma.masked_where(rsq > rs ** 2, np.zeros([nx, ny])), (sx, sy))
    assert t.has_undefined_data
    assert t.squeeze().has_undefined_data
    t2 = t.fill_undefined_data(1.0)
    assert not t2.has_undefined_data
    assert not t2.squeeze().has_undefined_data


@pipeline_function(DecoratedUniformTopography)
def scale_by_x(self, factor):
    return self.heights() * factor


def test_pipeline_decorators():
    t = fourier_synthesis((128, 128), (1, 1), 0.8, rms_height=1)
    t2 = t.scale_by_x(2)
    t3 = t.scale_by_x(3)
    np.testing.assert_almost_equal(2 * t.heights(), t2.heights())
    np.testing.assert_almost_equal(3 * t.heights(), t3.heights())
    assert len(t2.pipeline()) == 2
    assert t2.dim == t.dim
    assert t2.nb_grid_pts == t.nb_grid_pts
    np.testing.assert_almost_equal(t2.physical_sizes, t.physical_sizes)


UniformTopographyInterface.register_function('scale_by_x', scale_by_x)
