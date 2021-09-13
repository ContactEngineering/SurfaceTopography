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

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from SurfaceTopography import read_topography, Topography, UniformLineScan, \
    NonuniformLineScan
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Nonuniform.Autocorrelation import \
    height_height_autocorrelation

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
        r, A = UniformLineScan(y, nx, periodic=True).autocorrelation_from_profile()

        A_ana = np.zeros_like(A)
        A_ana[:w] = h ** 2 * np.linspace(w / nx, 1 / nx, w)
        A_ana = A_ana[0] - A_ana
        assert_array_almost_equal(A, A_ana)


def test_uniform_brute_force_autocorrelation_from_profile():
    n = 10
    for surf in [UniformLineScan(np.ones(n), n, periodic=False),
                 UniformLineScan(np.arange(n), n, periodic=False),
                 Topography(np.random.random(n).reshape(n, 1), (n, 1),
                            periodic=False)]:
        r, A = surf.autocorrelation_from_profile()

        n = len(A)
        dir_A = np.zeros(n)
        for d in range(n):
            for i in range(n - d):
                dir_A[d] += (surf.heights()[i] - surf.heights()[
                    i + d]) ** 2 / 2
            dir_A[d] /= (n - d)
        assert_array_almost_equal(A, dir_A)


def test_uniform_brute_force_autocorrelation_from_area():
    n = 10
    m = 11
    for surf in [Topography(np.ones([n, m]), (n, m), periodic=False),
                 Topography(np.random.random([n, m]), (n, m), periodic=False)]:
        r, A, A_xy = surf.autocorrelation_from_area(nbins=100, return_map=True)

        nx, ny = surf.nb_grid_pts
        dir_A_xy = np.zeros([n, m])
        dir_A = np.zeros_like(A)
        dir_n = np.zeros_like(A)
        for dx in range(n):
            for dy in range(m):
                for i in range(nx - dx):
                    for j in range(ny - dy):
                        dir_A_xy[dx, dy] += (surf.heights()[i, j] -
                                             surf.heights()[
                                                 i + dx, j + dy]) ** 2 / 2
                dir_A_xy[dx, dy] /= (nx - dx) * (ny - dy)
                d = np.sqrt(dx ** 2 + dy ** 2)
                i = np.argmin(np.abs(r - d))
                dir_A[i] += dir_A_xy[dx, dy]
                dir_n[i] += 1
        dir_n[dir_n == 0] = 1
        dir_A /= dir_n
        assert_array_almost_equal(A_xy, dir_A_xy)
        assert_array_almost_equal(A[:-2], dir_A[:-2])


def test_nonuniform_impulse_autocorrelation():
    a = 3
    b = 2
    x = np.array([0, a])
    t = NonuniformLineScan(x, b * np.ones_like(x))
    r, A = height_height_autocorrelation(t,
                                         distances=np.linspace(-4, 4, 101))

    A_ref = b ** 2 * (a - np.abs(r))
    A_ref[A_ref < 0] = 0

    assert_array_almost_equal(A, A_ref)

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

    assert_array_almost_equal(A, A_ref)

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

    assert_array_almost_equal(A, A2)

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

    m = np.logical_and(r > 1e-3, r < 10 ** (-1.5))
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

    assert_array_almost_equal(r1, r2)
    assert_array_almost_equal(A1, A2)


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
    r, A = t.detrend(detrend_mode='center').autocorrelation_from_profile()
    # Need to exclude final point because we cannot compute nonuniform ACF at
    # that point
    r = r[1:-1]
    A = A[1:-1]
    r2, A2 = t.detrend(
        detrend_mode='center').to_nonuniform().autocorrelation_from_profile(
        algorithm='brute-force', distances=r)

    assert_array_almost_equal(A, A2, decimal=5)


def test_brute_force_vs_fft():
    t = read_topography(os.path.join(DATADIR, 'example.xyz'))
    r, A = t.detrend().autocorrelation_from_profile()
    r2, A2 = t.detrend().autocorrelation_from_profile(algorithm='brute-force', distances=r, nb_interpolate=5)
    x = A[1:] / A2[1:]
    print(x.min(), x.max())
    assert np.alltrue(np.logical_and(x > 0.98, x < 1.02))
