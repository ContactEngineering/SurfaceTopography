#
# Copyright 2021 Lars Pastewka
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

import numpy as np
import scipy.interpolate

from SurfaceTopography import read_container, read_topography
from SurfaceTopography.Generation import fourier_synthesis


def test_scale_dependent_rms_slope_from_profile():
    t = fourier_synthesis((1024, 1024), (1, 1), 0.8, rms_slope=0.1)

    x, A = t.autocorrelation_from_profile(resampling_method=None)

    _, s = t.scale_dependent_statistical_property(lambda x, y: np.var(x), distance=x[1::20])

    np.testing.assert_allclose(2 * A[1::20] / x[1::20] ** 2, s)


def test_scalar_input():
    t = fourier_synthesis((1024,), (7,), 0.8, rms_slope=0.1)
    _, s = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[0.01, 0.1, 1])
    assert len(s) == 3
    _, s2 = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=0.1)
    with pytest.raises(TypeError):
        iter(s2)
    np.testing.assert_almost_equal(s[1], s2)
    _, s3 = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, scale_factor=2)
    with pytest.raises(TypeError):
        iter(s3)


def test_uniform_synthetic():
    t = fourier_synthesis((1024,), (7,), 0.8, rms_slope=0.1, periodic=False).detrend()

    L, u = t.bandwidth()

    r, A = t.autocorrelation_from_profile(resampling_method=None)
    r = r[1:]
    A = A[1:]
    assert abs(r.min()/L - 1) < 0.1  # Within 10% of bandwidth
    assert abs(r.max()/u - 1) < 0.1

    f = scipy.interpolate.interp1d(r, np.sqrt(2 * A) / r)
    r, s = t.scale_dependent_statistical_property(lambda x: np.mean(x * x), n=1)

    assert abs(r.min()/L - 1) < 0.1  # With 10% of bandwidth
    assert abs(r.max()/u - 1) < 0.1

    np.testing.assert_allclose(np.sqrt(s), f(r), atol=0.01)


def test_nonuniform_synthetic():
    t = fourier_synthesis((1024,), (7,), 0.8, rms_slope=0.1, periodic=False)
    p, = t.pixel_size
    s = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[p, 4 * p, 16 * p])
    t2 = t.to_nonuniform()
    t3 = t2.to_uniform(pixel_size=p)
    np.testing.assert_almost_equal(t3.pixel_size, p)
    np.testing.assert_almost_equal(t3.positions()[1] - t3.positions()[0], p)
    s2 = t2.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[p, 4 * p, 16 * p])
    np.testing.assert_allclose(s, s2, atol=1e-3)


def test_nonuniform_file(file_format_examples, plot=False):
    t = read_topography(f'{file_format_examples}/nonuniform.asc', unit='nm').detrend()

    L, u = t.bandwidth()

    q, C = t.power_spectrum_from_profile()

    assert abs(2 * np.pi / q.max() / L - 1) < 0.2
    assert abs(2 * np.pi / q.min() / u - 1) < 0.6

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(*t.positions_and_heights(), 'kx-')
        plt.show()

    r, A = t.autocorrelation_from_profile()
    assert abs(r.min() / L - 1) < 0.1
    assert abs(r.max() / u - 1) < 0.1

    if plot:
        plt.loglog(r, A, 'x-')
        plt.loglog(*t.to_uniform(nb_interpolate=5).autocorrelation_from_profile())
        plt.show()
        plt.loglog(r, np.sqrt(2 * A) / r, label='ACF')

    f = scipy.interpolate.interp1d(r, np.sqrt(2 * A) / r)

    r, s = t.scale_dependent_statistical_property(lambda x: np.mean(x * x))
    assert abs(r.min() / L - 1) < 0.1
    assert abs(r.max() / u - 1) < 0.1

    if plot:
        plt.loglog(r, np.sqrt(s), label='SDRP')
        plt.legend(loc='best')
        plt.show()

    np.testing.assert_allclose(np.sqrt(s[1:-1]), f(r[1:-1]), atol=0.005)


def test_container_uniform(file_format_examples, plot=False):
    """This container has just topography maps"""
    c, = read_container(f'{file_format_examples}/container1.zip')
    _, s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distances=[0.01, 0.1, 1.0, 10],
                                                  unit='um')
    assert (np.diff(s) < 0).all()
    np.testing.assert_almost_equal(s, [0.0018715281899762592, 0.0006849065620048571, 0.0002991781282532277,
                                       7.224607689277936e-05])

    # Test that specifying distances where no data exists does not raise an exception
    iterations = []
    _, s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distances=[0.00001, 1.0, 10000],
                                                  unit='um', progress_callback=lambda i, n: iterations.append(i))
    assert s[0] is None
    assert s[2] is None
    np.testing.assert_allclose(np.array(iterations), np.arange(len(c) + 1))

    # Test without specifying explicit distances
    d, s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, nb_points_per_decade=1, unit='um')

    np.testing.assert_allclose(d, [0.01, 0.1, 1, 10])
    np.testing.assert_allclose(s, [1.87152819e-03, 6.84906562e-04, 2.99178128e-04, 7.22460769e-05])

    # Test without specifying explicit distances and more points
    d, s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, nb_points_per_decade=5, unit='um')

    if plot:
        import matplotlib.pyplot as plt
        for t in c:
            plt.loglog(*t.to_unit('um').scale_dependent_statistical_property(lambda x, y: np.var(x), n=1,
                                                                             nb_points_per_decade=5), 'o-')
        plt.loglog(d, s, 'kx-')
        plt.show()

    # Make sure that we don't have two distances that are almost identical
    assert (np.diff(d) > 1e-3).all()


def test_container_mixed(file_format_examples):
    """This container has a mixture of maps and line scans"""
    c, = read_container(f'{file_format_examples}/container2.zip')
    _, s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, distances=[0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()

    # Test without specifying explicit distances
    d, s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, nb_points_per_decade=1, unit='um')
    np.testing.assert_allclose(d, [0.001, 0.01, 0.1, 1, 10, 100])
    np.testing.assert_allclose(s, [1.057908e-03, 4.016837e-02, 2.302750e-03, 2.058876e-04, 2.822253e-06, 2.650320e-08],
                               atol=1e-8)

    # Test without specifying explicit distances and more points
    d, s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, nb_points_per_decade=5, unit='um')

    # Make sure that we don't have two distances that are almost identical
    assert (np.diff(d) > 1e-4).all()


@pytest.mark.skip('Run this if you have a one of the big diamond containers downloaded from contact.engineering')
def test_large_container_mixed(plot=False):
    c, = read_container('/home/pastewka/Downloads/surface.zip')
    d, s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, unit='um', nb_points_per_decade=2)
    assert not np.any(np.isnan(s))

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(d, s, 'kx-')
        plt.show()


@pytest.mark.skip('Run this if you have a one of the big diamond containers downloaded from contact.engineering')
def test_large_container_power_spectrum():
    c, = read_container('/home/pastewka/Downloads/surface.zip')
    for t in c:
        q, C = t.power_spectrum_from_profile()
        print(t.info['datafile']['original'], len(q))
