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

from SurfaceTopography import NonuniformLineScan
from SurfaceTopography.IO.Text import read_xyz

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'file_format_examples')


def test_properties():
    x = np.array((0, 1, 1.5, 2, 3))
    h = 2 * x
    t = NonuniformLineScan(x, h)
    assert t.dim == 1


def test_squeeze():
    x = np.linspace(0, 4 * np.pi, 101) ** (1.3)
    h = np.sin(x)
    surface = NonuniformLineScan(x, h).scale(2.0)
    surface2 = surface.squeeze()
    assert isinstance(surface2, NonuniformLineScan)
    np.testing.assert_allclose(surface.positions(), surface2.positions())
    np.testing.assert_allclose(surface.heights(), surface2.heights())


def test_positions_and_heights():
    x = np.array((0, 1, 1.5, 2, 3))
    h = 2 * x

    t = NonuniformLineScan(x, h)

    np.testing.assert_allclose(t.heights(), h)
    np.testing.assert_allclose(t.positions(), x)

    x2, h2 = t.positions_and_heights()
    np.testing.assert_allclose(x2, x)
    np.testing.assert_allclose(h2, h)


def test_attribute_error():
    t = NonuniformLineScan([1, 2, 4], [2, 4, 8])
    with pytest.raises(AttributeError):
        t.scale_factor
    # a scaled line scan has a scale_factor
    assert t.scale(1).scale_factor == 1

    #
    # This should also work after the topography has been pickled
    #
    pt = pickle.dumps(t)
    t2 = pickle.loads(pt)

    with pytest.raises(AttributeError):
        t2.scale_factor
    # a scaled line scan has a scale_factor
    assert t2.scale(1).scale_factor == 1


def test_setting_info_dict():
    x = np.array((0, 1, 1.5, 2, 3))
    h = 2 * x

    t = NonuniformLineScan(x, h)

    assert t.info == {}

    with pytest.deprecated_call():
        t = NonuniformLineScan(x, h, info=dict(unit='A'))
    t = NonuniformLineScan(x, h, unit='A')
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
    st = t.scale(2, 1, unit='B')
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
    # initialize with lists instead of arrays
    t = NonuniformLineScan(x=[1, 2, 3, 4], y=[2, 4, 6, 8])

    # the following commands should be possible without errors
    st = t.scale(1)
    st.detrend(detrend_mode='center')


def test_power_spectrum_from_profile():
    #
    # this test was added, because there were issues calling
    # power spectrum 1D with a window given
    #
    t = NonuniformLineScan(x=[1, 2, 3, 4], y=[2, 4, 6, 8])
    q1, C1 = t.power_spectrum_from_profile(window='hann')
    q1, C1 = t.detrend('center').power_spectrum_from_profile(window='hann')
    q1, C1 = t.detrend('center').power_spectrum_from_profile()
    q1, C1 = t.detrend('height').power_spectrum_from_profile(window='hann')
    q1, C1 = t.detrend('height').power_spectrum_from_profile()
    # ok can be called without errors


def test_detrend():
    t = read_xyz(os.path.join(DATADIR, 'example.xyz'))
    assert not t.detrend('center').is_periodic
    assert not t.detrend('height').is_periodic


def test_masked_input():
    with pytest.raises(ValueError):
        NonuniformLineScan(x=np.ma.array([1, 2, 3], mask=[0, 1, 0]), y=[4, 5, 6])

    with pytest.raises(ValueError):
        NonuniformLineScan(y=np.ma.array([1, 2, 3], mask=[0, 1, 0]), x=[4, 5, 6])
