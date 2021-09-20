#
# Copyright 2019-2020 Lars Pastewka
#           2019 Antoine Sanner
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
Tests for autocorrelation function analysis
"""

import os

import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.interpolate import interp1d

from SurfaceTopography import read_container, read_topography, Topography, UniformLineScan, NonuniformLineScan
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Nonuniform.Autocorrelation import height_height_autocorrelation

DATADIR = os.path.join(os.path.dirname(__file__), 'file_format_examples')


###

def py_autocorrelation_from_profile(line_scan, distances=None):
    r"""
    Compute the one-dimensional height-difference autocorrelation function
    (ACF).

    This function treats the nonuniform line scan as a piece-wise function of
    straight lines between the data points. The ACF is computed exactly for
    this piece-wise linear interpolation of the data.

    Parameters
    ----------
    line_scan : :class:`SurfaceTopography.NonuniformLineScan`
        Container storing the nonuniform line scan.
    r : array_like
        Array containing distances for which to compute the ACF. If no array
        is given, the function will automatically construct an array with
        equally spaced distances. (Default: None)

    Returns
    -------
    distances : array
        Distances. (Units: length)
    A : array
        Autocorrelation function. (Units: length**2)
    """
    size, = line_scan.physical_sizes
    if distances is None:
        # FIXME!!! We need a better heuristics to decide on the distances
        res, = line_scan.nb_grid_pts
        distances = np.arange(res) * size / res
    else:
        distances = np.asarray(distances, dtype=float)
    A = np.zeros_like(distances)

    x, h = line_scan.positions_and_heights()
    s = line_scan.derivative(1)
    # FIXME!!! This is slow
    for i in range(len(x) - 1):
        for j in range(len(x) - 1):
            # Determine lower and upper distance between segment i, i+1 and
            # segment j, j+1
            x1 = x[i]
            x2 = x[j]
            h1 = h[i]
            h2 = h[j]
            s1 = s[i]
            s2 = s[j]
            b1 = np.maximum(x1, x2 - distances)
            b2 = np.minimum(x[i + 1], x[j + 1] - distances)
            b = (b1 + b2) / 2
            db = (b2 - b1) / 2
            m = db > 0
            if m.sum() > 0:
                b = b[m]
                db = db[m]
                # f1[x_] := (h1 + s1*(x - x1))
                # f2[x_] := (h2 + s2*(x - x2))
                # FullSimplify[Integrate[f1[x]*f2[x + d], {x, b - db, b + db}]]
                #   = 2 * f1[b] * f2[b + d] * db + 2 * s1 * s2 * db ** 3 / 3
                A[m] += (db * (3 * (h2 - s2 * x2 + (b + distances[
                    m]) * s2 - h1 + s1 * x1 - b * s1) ** 2 + (
                                       s1 - s2) ** 2 * db ** 2)) / 3
    return distances, A / (size - distances)


###

def test_uniform_impulse_autocorrelation():
    nx = 16
    for x, w, h, p in [(nx // 2, 3, 1, True), (nx // 3, 2, 2, True),
                       (nx // 2, 5, 1, False), (nx // 3, 6, 2.5, False)]:
        y = np.zeros(nx)
        y[x - w // 2:x + (w + 1) // 2] = h
        r, A = UniformLineScan(y, nx, periodic=True).autocorrelation_from_profile(resampling_method=None)

        A_ana = np.zeros_like(A)
        A_ana[:w] = h ** 2 * np.linspace(w / nx, 1 / nx, w)
        A_ana = A_ana[0] - A_ana
        assert_allclose(A, A_ana)


def test_uniform_brute_force_autocorrelation_from_profile():
    n = 10
    for surf in [UniformLineScan(np.ones(n), n, periodic=False),
                 UniformLineScan(np.arange(n), n, periodic=False),
                 Topography(np.random.random(n).reshape(n, 1), (n, 1),
                            periodic=False)]:
        r, A = surf.autocorrelation_from_profile(resampling_method=None)

        n = len(A)
        dir_A = np.zeros(n)
        for d in range(n):
            for i in range(n - d):
                dir_A[d] += (surf.heights()[i] - surf.heights()[
                    i + d]) ** 2 / 2
            dir_A[d] /= (n - d)
        assert_allclose(A, dir_A, atol=1e-12)


@pytest.mark.parametrize("surf,tol_kwargs",
                         [(Topography(np.ones([10, 11]), (10, 11), periodic=False), dict(atol=1e-12)),
                          (Topography(np.random.random([10, 11]), (10, 11), periodic=False), dict(atol=0.5))])
def test_uniform_brute_force_autocorrelation_from_area(surf, tol_kwargs):
    r, A, A_xy = surf.autocorrelation_from_area(nb_points=100, return_map=True)

    nx, ny = surf.nb_grid_pts
    dir_A_xy = np.zeros([nx, ny])
    dir_A = np.zeros_like(A)
    dir_n = np.zeros_like(A)
    for dx in range(nx):
        for dy in range(nx):
            for i in range(nx - dx):
                for j in range(ny - dy):
                    dir_A_xy[dx, dy] += (surf.heights()[i, j] - surf.heights()[i + dx, j + dy]) ** 2 / 2
            dir_A_xy[dx, dy] /= (nx - dx) * (ny - dy)
            d = np.sqrt(dx ** 2 + dy ** 2)
            i = np.argmin(np.abs(r - d))
            dir_A[i] += dir_A_xy[dx, dy]
            dir_n[i] += 1
    dir_n[dir_n == 0] = 1
    dir_A /= dir_n
    assert_allclose(A_xy, dir_A_xy, **tol_kwargs)
    m = np.isfinite(A)
    assert_allclose(A[m], dir_A[m], **tol_kwargs)


def test_nonuniform_impulse_autocorrelation():
    a = 3
    b = 2
    x = np.array([0, a])
    t = NonuniformLineScan(x, b * np.ones_like(x))
    r, A = height_height_autocorrelation(t, distances=np.linspace(-4, 4, 101))

    A_ref = b ** 2 * (a - np.abs(r))
    A_ref[A_ref < 0] = 0

    assert_allclose(A, A_ref)

    a = 3
    b = 2
    x = np.array([-a, 0, 1e-9, a - 1e-9, a, 2 * a])
    y = np.zeros_like(x)
    y[2] = b
    y[3] = b
    t = NonuniformLineScan(x, y)
    r, A = height_height_autocorrelation(t,
                                         distances=np.linspace(-4, 4, 101))

    A_ref = b ** 2 * (a - np.abs(r))
    A_ref[A_ref < 0] = 0

    assert_allclose(A, A_ref)

    t = t.detrend(detrend_mode='center')
    r, A = height_height_autocorrelation(t,
                                         distances=np.linspace(0, 10, 201))

    s, = t.physical_sizes
    assert_almost_equal(A[0], t.rms_height_from_profile() ** 2 * s)


def test_nonuniform_triangle_autocorrelation():
    a = 0.7
    b = 3
    x = np.array([0, b])
    t = NonuniformLineScan(x, a * x)
    r, A = height_height_autocorrelation(t,
                                         distances=np.linspace(-4, 4, 101))

    assert_almost_equal(A[np.abs(r) < 1e-6][0], a ** 2 * b ** 3 / 3)

    r3, A3 = height_height_autocorrelation(t.detrend(detrend_mode='center'),
                                           distances=[0])
    s, = t.physical_sizes
    assert_almost_equal(A3[0], t.rms_height_from_profile() ** 2 * s)

    x = np.array([0, 1., 1.3, 1.7, 2.0, 2.5, 3.0])
    t = NonuniformLineScan(x, a * x)
    r2, A2 = height_height_autocorrelation(t, distances=np.linspace(-4, 4,
                                                                    101))

    assert_allclose(A, A2)

    r, A = height_height_autocorrelation(t.detrend(detrend_mode='center'),
                                         distances=[0])
    s, = t.physical_sizes
    assert_almost_equal(A[0], t.rms_height_from_profile() ** 2 * s)


def test_self_affine_uniform_autocorrelation():
    r = 2048
    s = 1
    H = 0.8
    slope = 0.1
    t = fourier_synthesis((r,), (s,), H, rms_slope=slope,
                          amplitude_distribution=lambda n: 1.0)

    r, A = t.autocorrelation_from_profile()

    m = np.logical_and(np.logical_and(r > 1e-3, r < 10 ** (-1.5)), np.isfinite(A))
    b, a = np.polyfit(np.log(r[m]), np.log(A[m]), 1)
    assert abs(b / 2 - H) < 0.1


def test_c_vs_py_reference():
    from _SurfaceTopography import nonuniform_autocorrelation
    r = 16
    s = 1
    H = 0.8
    slope = 0.1
    t = fourier_synthesis((r,), (s,), H, rms_slope=slope,
                          amplitude_distribution=lambda n: 1.0,
                          periodic=False).to_nonuniform()

    r1, A1 = py_autocorrelation_from_profile(t)

    s, = t.physical_sizes
    r2, A2 = nonuniform_autocorrelation(*t.positions_and_heights(), s)

    assert_allclose(r1, r2)
    assert_allclose(A1, A2)


def test_nonuniform_rms_height():
    r = 128
    s = 1.3
    H = 0.8
    slope = 0.1
    t = fourier_synthesis((r,), (s,), H, rms_slope=slope, amplitude_distribution=lambda n: 1.0, periodic=False) \
        .to_nonuniform().detrend(detrend_mode='center')
    assert_almost_equal(t.mean(), 0)

    r, A = height_height_autocorrelation(t, distances=[0])
    s, = t.physical_sizes
    assert_almost_equal(t.rms_height_from_profile() ** 2 * s, A[0])


def test_self_affine_nonuniform_autocorrelation():
    r = 128
    s = 1.3
    H = 0.8
    slope = 0.1
    t = fourier_synthesis((r,), (s,), H, rms_slope=slope, short_cutoff=s / 20,
                          amplitude_distribution=lambda n: 1.0)
    t._periodic = False
    r, A = t.detrend(detrend_mode='center').autocorrelation_from_profile(resampling_method=None)
    # Need to exclude final point because we cannot compute nonuniform ACF at
    # that point
    r = r[1:-1]
    A = A[1:-1]
    r2, A2 = t.detrend(detrend_mode='center').to_nonuniform() \
        .autocorrelation_from_profile(algorithm='brute-force', distances=r, reliable=False, resampling_method=None,
                                      short_cutoff=None)

    assert_allclose(A, A2, atol=1e-5)


def test_brute_force_vs_fft():
    t = read_topography(os.path.join(DATADIR, 'example.xyz'))
    r, A = t.detrend().autocorrelation_from_profile()
    m = np.isfinite(A)
    r = r[m]
    A = A[m]
    r2, A2 = t.detrend().autocorrelation_from_profile(algorithm='brute-force', distances=r, nb_interpolate=5,
                                                      reliable=False, resampling_method=None, short_cutoff=None)
    x = A[1:] / A2[1:]
    assert np.alltrue(np.logical_and(x > 0.9, x < 1.1))


@pytest.mark.parametrize('nb_grid_pts,physical_sizes', [((128,), (1.3,)), ((128, 128), (2.3, 3.1))])
def test_resampling(nb_grid_pts, physical_sizes, plot=False):
    H = 0.8
    slope = 0.1
    t = fourier_synthesis(nb_grid_pts, physical_sizes, H, rms_slope=slope, short_cutoff=np.mean(physical_sizes) / 20,
                          amplitude_distribution=lambda n: 1.0)
    r1, A1 = t.autocorrelation_from_profile(resampling_method=None)
    r2, A2 = t.autocorrelation_from_profile(resampling_method='bin-average')
    # r3, A3 = t.autocorrelation_from_profile(resampling_method='gaussian-process')

    assert len(r1) == len(A1)
    assert len(r2) == len(A2)
    # assert len(r3) == len(A3)

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(r1, A1, 'x-', label='native')
        plt.loglog(r2, A2, 'o-', label='bin-average')
        # plt.loglog(r3, A3, 's-', label='gaussian-process')
        plt.legend(loc='best')
        plt.show()

    f = interp1d(r1, A1)
    assert_allclose(A2[np.isfinite(A2)], f(r2[np.isfinite(A2)]), atol=1e-6)
    # assert_allclose(A3, f(r3), atol=1e-4)


def test_container_uniform(file_format_examples, plot=False):
    """This container has just topography maps"""
    c, = read_container(f'{file_format_examples}/container1.zip')
    d, s = c.autocorrelation(unit='um', nb_points_per_decade=2)

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(d, s, 'o-')
        for t in c:
            plt.loglog(*t.to_unit('um').autocorrelation_from_profile(), 'x-')
        plt.show()

    assert_allclose(s, [4.76306017e-09, 5.42493630e-08, 3.44862254e-07, 2.09698128e-06, 1.16854493e-05, 6.11236431e-05,
                        4.07759215e-04, 1.41787788e-03, 7.73657754e-03, 1.39632798e-02])


# This test is just supposed to finish without an exception
def test_container_mixed(file_format_examples, plot=False):
    """This container has a mixture of maps and line scans"""
    c, = read_container(f'{file_format_examples}/container2.zip')
    d, s = c.autocorrelation(unit='um')

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(d, s, 'o-')
        for t in c:
            plt.loglog(*t.to_unit('um').autocorrelation_from_profile(), 'x-')
        plt.show()


@pytest.mark.skip('Run this if you have a one of the big diamond containers downloaded from contact.engineering')
def test_large_container_mixed(plot=True):
    c, = read_container('/home/pastewka/Downloads/surface.zip')
    d, s = c.autocorrelation(unit='um')

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(d, s, 'kx-')
        plt.show()
