#
# Copyright 2021-2023 Lars Pastewka
#           2022-2023 Antoine Sanner
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
from SurfaceTopography.IO import XYZReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


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


def test_squeeze_unit():
    x = np.linspace(0, 4 * np.pi, 101) ** (1.3)
    h = np.sin(x)
    surface = NonuniformLineScan(x, h, unit="m").scale(2.0)
    surface2 = surface.squeeze()
    assert surface2.unit == surface.unit

    surface = NonuniformLineScan(x, h, ).scale(2.0)
    surface2 = surface.squeeze()
    assert surface2.unit == surface.unit


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
    x = np.array((0, 1, 1.5, 2, 3))
    h = 2 * x

    t = NonuniformLineScan(x, h)

    assert t.info == {}

    t = NonuniformLineScan(x, h, unit='A')

    #
    # This info should be inherited in the pipeline
    #
    st = t.scale(2)
    with pytest.deprecated_call():
        assert st.info['unit'] == 'A'

    #
    # It should be also possible to set the info
    #
    st = t.scale(2, 1, unit='B')
    with pytest.deprecated_call():
        assert st.info['unit'] == 'B'


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


def test_detrend(file_format_examples):
    t = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography()
    assert not t.detrend('center').is_periodic
    assert not t.detrend('height').is_periodic


def test_detrend_slope(file_format_examples):
    t = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography()
    assert not t.detrend('slope').is_periodic


def test_detrend_curvature(file_format_examples):
    t = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography()
    assert not t.detrend('curvature').is_periodic


def test_masked_input():
    with pytest.raises(ValueError):
        NonuniformLineScan(x=np.ma.array([1, 2, 3], mask=[0, 1, 0]), y=[4, 5, 6])

    with pytest.raises(ValueError):
        NonuniformLineScan(y=np.ma.array([1, 2, 3], mask=[0, 1, 0]), x=[4, 5, 6])
