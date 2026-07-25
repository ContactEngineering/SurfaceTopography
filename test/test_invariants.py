#
# Copyright 2026 Lars Pastewka
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
Property-based invariant tests: analysis results must be unchanged (or
transform in a known way) under transposition, translation, scaling and
shifts of the coordinate origin, and pipeline decorators must report
consistent geometry.
"""

import numpy as np
import pytest

from SurfaceTopography import NonuniformLineScan, Topography, UniformLineScan


@pytest.fixture
def anisotropic_topography():
    """Periodic topography with anisotropic pixels (px = 4 py)."""
    rng = np.random.RandomState(42)
    nx, ny = 64, 32
    heights = rng.uniform(-1, 1, (nx, ny))
    return Topography(heights, (8.0, 1.0), periodic=True)


def test_transpose_geometry(anisotropic_topography):
    t = anisotropic_topography
    tt = t.transpose()
    nx, ny = t.nb_grid_pts
    sx, sy = t.physical_sizes
    px, py = t.pixel_size
    assert tt.nb_grid_pts == (ny, nx)
    assert tt.physical_sizes == (sy, sx)
    np.testing.assert_allclose(tt.pixel_size, (py, px))
    # Geometry consistency of the decorator itself
    np.testing.assert_allclose(
        tt.pixel_size, np.asarray(tt.physical_sizes) / np.asarray(tt.nb_grid_pts))
    np.testing.assert_allclose(tt.heights(), t.heights().T)


def test_transpose_invariance_of_scalar_parameters(anisotropic_topography):
    """Scalar roughness parameters that do not single out a scan direction
    must be invariant under transposition, also for anisotropic pixels."""
    t = anisotropic_topography
    tt = t.transpose()
    np.testing.assert_allclose(tt.rms_height_from_area(), t.rms_height_from_area())
    np.testing.assert_allclose(tt.rms_gradient(), t.rms_gradient())
    np.testing.assert_allclose(tt.rms_laplacian(), t.rms_laplacian())
    np.testing.assert_allclose(tt.bandwidth(), t.bandwidth())


def test_translate_invariance(anisotropic_topography):
    """Integer-pixel translations of a periodic topography change nothing
    but the position of the heights."""
    t = anisotropic_topography
    tt = t.translate(offset=(5, 7))
    np.testing.assert_allclose(
        tt.heights(), np.roll(np.roll(t.heights(), 5, axis=0), 7, axis=1))
    assert tt.physical_sizes == t.physical_sizes
    assert tt.nb_grid_pts == t.nb_grid_pts
    np.testing.assert_allclose(tt.rms_height_from_area(), t.rms_height_from_area())
    np.testing.assert_allclose(tt.rms_gradient(), t.rms_gradient())


def test_scale_invariants(anisotropic_topography):
    """Linear analysis functions scale linearly with a static height scale."""
    t = anisotropic_topography
    factor = 2.5
    ts = t.scale(factor)
    np.testing.assert_allclose(ts.rms_height_from_area(),
                               factor * t.rms_height_from_area())
    np.testing.assert_allclose(ts.rms_gradient(), factor * t.rms_gradient())
    # The bearing area transforms with the height scale
    c = 0.3
    np.testing.assert_allclose(ts.bearing_area(factor * c), t.bearing_area(c))


def test_decorator_geometry_consistency(anisotropic_topography):
    """All decorators must report pixel_size == physical_sizes / nb_grid_pts."""
    t = anisotropic_topography
    for decorated in [t.scale(2.0),
                      t.detrend('center'),
                      t.transpose(),
                      t.translate(offset=(1, 2)),
                      t.transpose().detrend('center'),
                      t.scale(3.0).transpose()]:
        np.testing.assert_allclose(
            decorated.pixel_size,
            np.asarray(decorated.physical_sizes) / np.asarray(decorated.nb_grid_pts),
            err_msg=f'inconsistent geometry for {decorated}')


def test_bearing_area_bounds_and_complement():
    """The bearing area is a CDF: it is 1 below the minimum, 0 above the
    maximum, and the bearing area of the inverted surface complements it."""
    rng = np.random.RandomState(0)
    heights = rng.uniform(-1, 1, (32, 33))
    for periodic in [False, True]:
        t = Topography(heights, (2.0, 1.0), periodic=periodic)
        ti = Topography(-heights, (2.0, 1.0), periodic=periodic)
        assert t.bearing_area(heights.min() - 0.1) == pytest.approx(1.0)
        assert t.bearing_area(heights.max() + 0.1) == pytest.approx(0.0)
        for c in [-0.5, -0.1, 0.0, 0.2, 0.7]:
            np.testing.assert_allclose(ti.bearing_area(-c),
                                       1 - t.bearing_area(c), atol=1e-12)


def test_uniform_line_scan_scale_and_transpose_roundtrip():
    rng = np.random.RandomState(1)
    h = rng.uniform(-1, 1, 128)
    t = UniformLineScan(h, 4.0, periodic=True)
    np.testing.assert_allclose(t.scale(3.0).rms_height_from_profile(),
                               3.0 * t.rms_height_from_profile())
    tt = t.translate(offset=17)
    np.testing.assert_allclose(tt.heights(), np.roll(h, 17))
    np.testing.assert_allclose(tt.rms_height_from_profile(),
                               t.rms_height_from_profile())


def test_nonuniform_detrend_origin_invariance():
    """Detrending must not depend on where the scan sits on the x-axis.

    This is a regression test for the ill-conditioned normal equations in
    the nonuniform `polyfit`, which are now solved in centered coordinates.
    """
    rng = np.random.RandomState(2)
    x = np.sort(rng.uniform(0, 1, 100))
    x[0] = 0
    h = 0.1 * rng.randn(100) + 0.5 * x - 0.8 * x * x
    t = NonuniformLineScan(x, h)
    for offset in [-1e3, 1e3, 1e5]:
        t_shifted = NonuniformLineScan(x + offset, h)
        # Evaluating the detrending polynomial a0 + a1 x + a2 x^2 at
        # x ~ offset is limited by double precision to an absolute error
        # of order eps * offset^2, even for exact coefficients
        atol = max(1e-9, 100 * np.finfo(float).eps * offset ** 2)
        for mode in ['mean', 'median', 'rms-tilt', 'slope', 'rms-curvature']:
            np.testing.assert_allclose(
                t_shifted.detrend(mode).heights(),
                t.detrend(mode).heights(), atol=atol,
                err_msg=f'detrend mode {mode} is not origin invariant '
                        f'for offset {offset}')


def test_nonuniform_polyfit_recovers_exact_polynomial():
    rng = np.random.RandomState(3)
    x = np.sort(rng.uniform(0, 1, 50))
    x[0] = 0
    h = 2.0 + 3.0 * x - 4.0 * x * x
    t = NonuniformLineScan(x, h)
    np.testing.assert_allclose(t.polyfit(2), [2.0, 3.0, -4.0], atol=1e-9)
