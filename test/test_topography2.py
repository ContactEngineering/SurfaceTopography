#
# Copyright 2019-2020 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import pickle

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from muFFT import FFT
from NuMPI.Tools import Reduction

from SurfaceTopography import Topography


def test_positions(comm):
    nx, ny = (12 * comm.Get_size(), 10 * comm.Get_size() + 1)
    sx = 33.
    sy = 54.
    fftengine = FFT((nx, ny), fft='mpi', communicator=comm)

    surf = Topography(np.zeros(fftengine.nb_subdomain_grid_pts),
                      physical_sizes=(sx, sy),
                      decomposition='subdomain',
                      nb_grid_pts=(nx, ny),
                      subdomain_locations=fftengine.subdomain_locations,
                      communicator=comm)

    x, y = surf.positions()
    assert x.shape == fftengine.nb_subdomain_grid_pts
    assert y.shape == fftengine.nb_subdomain_grid_pts

    assert Reduction(comm).min(x) == 0
    assert abs(Reduction(comm).max(x) - sx * (1 - 1. / nx)) \
           < 1e-8 * sx / nx, "{}".format(x)
    assert Reduction(comm).min(y) == 0
    assert abs(Reduction(comm).max(y) - sy * (1 - 1. / ny)) < 1e-8


def test_positions_and_heights():
    X = np.arange(3).reshape(1, 3)
    Y = np.arange(4).reshape(4, 1)
    h = X + Y

    t = Topography(h, (8, 6))

    assert t.nb_grid_pts == (4, 3)

    assert_array_equal(t.heights(), h)
    X2, Y2, h2 = t.positions_and_heights()
    assert_array_equal(X2, [
        (0, 0, 0),
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
    ])
    assert_array_equal(Y2, [
        (0, 2, 4),
        (0, 2, 4),
        (0, 2, 4),
        (0, 2, 4),
    ])
    assert_array_equal(h2, [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5)])

    #
    # After detrending, the position and heights should have again
    # just 3 arrays and the third array should be the same as .heights()
    #
    dt = t.detrend(detrend_mode='slope')

    np.testing.assert_allclose(dt.heights(), [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0)], atol=1e-15)

    X2, Y2, h2 = dt.positions_and_heights()

    assert h2.shape == (4, 3)
    assert_array_equal(X2, [
        (0, 0, 0),
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
    ])
    assert_array_equal(Y2, [
        (0, 2, 4),
        (0, 2, 4),
        (0, 2, 4),
        (0, 2, 4),
    ])
    np.testing.assert_allclose(h2, [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0)], atol=1e-15)


def test_squeeze():
    x = np.linspace(0, 4 * np.pi, 101)
    y = np.linspace(0, 8 * np.pi, 103)
    h = np.sin(x.reshape(-1, 1)) + np.cos(y.reshape(1, -1))
    surface = Topography(h, (1.2, 3.2)).scale(2.0)
    surface2 = surface.squeeze()
    assert isinstance(surface2, Topography)
    np.testing.assert_allclose(surface.heights(), surface2.heights())


def test_attribute_error():
    X = np.arange(3).reshape(1, 3)
    Y = np.arange(4).reshape(4, 1)
    h = X + Y
    t = Topography(h, (8, 6))

    # nonsense attributes return attribute error
    with pytest.raises(AttributeError):
        t.ababababababababa

    #
    # only scaled topographies have coeff
    #
    with pytest.raises(AttributeError):
        t.coeff

    st = t.scale(1)

    assert st.scale_factor == 1

    #
    # only detrended topographies have detrend_mode
    #
    with pytest.raises(AttributeError):
        st.detrend_mode

    dm = st.detrend(detrend_mode='height').detrend_mode
    assert dm == 'height'

    #
    # this all should also work after pickling
    #
    t2 = pickle.loads(pickle.dumps(t))

    with pytest.raises(AttributeError):
        t2.scale_factor

    st2 = t2.scale(1)

    assert st2.scale_factor == 1

    with pytest.raises(AttributeError):
        st2.detrend_mode

    dm2 = st2.detrend(detrend_mode='height').detrend_mode
    assert dm2 == 'height'

    #
    # this all should also work after scaled+pickled
    #
    t3 = pickle.loads(pickle.dumps(st))

    with pytest.raises(AttributeError):
        t3.detrend_mode

    dm3 = t3.detrend(detrend_mode='height').detrend_mode
    assert dm3 == 'height'


def test_init_with_lists_calling_scale_and_detrend():
    t = Topography(np.array([[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]]), physical_sizes=(1, 1))

    # the following commands should be possible without errors
    st = t.scale(1)
    st.detrend(detrend_mode='center')


def test_power_spectrum_1D():
    X = np.arange(3).reshape(1, 3)
    Y = np.arange(4).reshape(4, 1)
    h = X + Y

    t = Topography(h, (8, 6))

    q1, C1 = t.power_spectrum_1D(window='hann')

    # TODO add check for values
