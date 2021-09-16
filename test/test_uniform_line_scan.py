#
# Copyright 2016, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2018, 2020 Michael Röttger
#           2015-2016 Till Junge
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
Tests surface classes
"""

import os
import pickle

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import UniformLineScan

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'file_format_examples')


def test_properties():
    x = np.array((0, 1, 2, 3, 4))
    h = 2 * x
    t = UniformLineScan(h, 5)
    assert t.dim == 1


def test_squeeze():
    x = np.linspace(0, 4 * np.pi, 101)
    h = np.sin(x)
    surface = UniformLineScan(h, 4 * np.pi).scale(2.0)
    surface2 = surface.squeeze()
    assert isinstance(surface2, UniformLineScan)
    np.testing.assert_allclose(surface.heights(), surface2.heights())


def test_positions_and_heights():
    h = np.array((0, 1, 2, 3, 4))

    t = UniformLineScan(h, 4)

    np.testing.assert_allclose(t.heights(), h)

    expected_x = np.array((0., 0.8, 1.6, 2.4, 3.2))
    np.testing.assert_allclose(t.positions(), expected_x)

    x2, h2 = t.positions_and_heights()
    np.testing.assert_allclose(x2, expected_x)
    np.testing.assert_allclose(h2, h)


def test_attribute_error():
    h = np.array((0, 1, 2, 3, 4))
    t = UniformLineScan(h, 4)

    with pytest.raises(AttributeError):
        t.height_scale_factor
    # a scaled line scan has a scale_factor
    assert t.scale(1).height_scale_factor == 1

    #
    # This should also work after the topography has been pickled
    #
    pt = pickle.dumps(t)
    t2 = pickle.loads(pt)

    with pytest.raises(AttributeError):
        t2.height_scale_factor
    # a scaled line scan has a scale_factor
    assert t2.scale(1).height_scale_factor == 1


def test_setting_info_dict():
    h = np.array((0, 1, 2, 3, 4))
    t = UniformLineScan(h, 4)

    assert t.info == {}

    with pytest.deprecated_call():
        t = UniformLineScan(h, 4, info=dict(unit='A'))
    t = UniformLineScan(h, 4, unit='A')
    with pytest.deprecated_call():
        assert t.info['unit'] == 'A'

    #
    # This info should be inherited in the pipeline
    #
    st = t.scale(2)
    with pytest.deprecated_call():
        assert st.info['unit'] == 'A'

    #
    # It should be also possible to set the info
    #
    with pytest.deprecated_call():
        st = t.scale(2, info=dict(unit='B'))
    st = t.scale(2, 2, unit='B')
    with pytest.deprecated_call():
        assert st.info['unit'] == 'B'

    #
    # Again the info should be passed
    #
    dt = st.detrend(detrend_mode='center')
    with pytest.deprecated_call():
        assert dt.info['unit'] == 'B'

    #
    # It can no longer be changed in detrend (you need to use scale)
    #
    with pytest.deprecated_call():
        dt = st.detrend(detrend_mode='center', info=dict(unit='C'))


def test_init_with_lists_calling_scale_and_detrend():
    t = UniformLineScan([2, 4, 6, 8],
                        4)  # initialize with list instead of arrays

    # the following commands should be possible without errors
    st = t.scale(1)
    st.detrend(detrend_mode='center')


def test_detrend_curvature():
    n = 10
    dx = 0.5
    x = np.arange(n) * dx

    R = 4.
    h = x ** 2 / R

    t = UniformLineScan(h, dx * n)

    detrended = t.detrend(detrend_mode="curvature")

    assert abs(detrended.coeffs[-1] / detrended.physical_sizes[0] ** 2 - 1 / R) < 1e-12


def test_detrend_same_positions():
    """asserts that the detrended topography has the same x
    """
    n = 10
    dx = 0.5
    h = np.random.normal(size=n)

    t = UniformLineScan(h, dx * n)

    for mode in ["curvature", "slope", "height"]:
        detrended = t.detrend(detrend_mode=mode)
        np.testing.assert_allclose(detrended.positions(), t.positions())
        np.testing.assert_allclose(detrended.positions_and_heights()[0],
                                   t.positions_and_heights()[0])


def test_detrend_heights_call():
    """ tests if the call of heights make no mistake
    """
    n = 10
    dx = 0.5
    h = np.random.normal(size=n)

    t = UniformLineScan(h, dx * n)
    for mode in ["height", "curvature", "slope"]:
        detrended = t.detrend(detrend_mode=mode)
        detrended.heights()


def test_power_spectrum_from_profile():
    #
    # this test was added, because there were issues calling
    # power spectrum 1D with a window given
    #
    t = UniformLineScan([2, 4, 6, 8], 4)
    t.power_spectrum_from_profile(window='hann', resampling_method=None)
