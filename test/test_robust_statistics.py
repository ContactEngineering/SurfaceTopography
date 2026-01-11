#
# Copyright 2020-2021, 2023-2024 Lars Pastewka
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
Tests robust statistics module.
"""

import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import Topography, read_topography
from SurfaceTopography.Generation import fourier_synthesis

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_median_and_mad(plot=False):
    t = fourier_synthesis((2048, 2048), (1, 1), 0.1, rms_slope=0.1, periodic=True)

    # This is almost Gaussian
    np.testing.assert_allclose(t.median(), t.mean(), atol=1e-5)
    np.testing.assert_allclose(t.mad_height(), t.rms_height_from_area(), atol=1e-5)

    # Skew the distribution
    t2 = Topography(np.exp(t.heights() / t.rms_height_from_area() / 2), t.physical_sizes, periodic=t.is_periodic)

    assert t2.mean() > t2.median()
    assert t2.rms_height_from_area() > t2.mad_height()

    if plot:
        import matplotlib.pyplot as plt
        plt.hist(t2.heights().ravel(), bins=100)
        plt.show()


def test_mad_detrending_nonuniform(file_format_examples, plot=False):
    file_path = os.path.join(file_format_examples, 'dektak-1.csv')
    t = read_topography(file_path)

    if plot:
        import matplotlib.pyplot as plt

        t2 = t.detrend('rms-tilt')
        x = np.linspace(t2.coeffs[1] - 0.001, t2.coeffs[1] + 0.001, 101)
        rms = [t.detrend(coeffs=[0, _x]).rms_height_from_profile() for _x in x]
        mad = [t.detrend(coeffs=[0, _x]).mad_height() for _x in x]

        plt.figure()
        plt.plot(x, rms, 'k-')
        plt.plot(x, mad, 'r-')
        plt.show()

        plt.figure()
        plt.plot(*t.detrend('rms-tilt').positions_and_heights(), 'k--')
        plt.plot(*t.detrend('rms-curvature').positions_and_heights(), 'k-')
        plt.plot(*t.detrend('mad-tilt').positions_and_heights(), 'r--')
        plt.plot(*t.detrend('mad-curvature').positions_and_heights(), 'r-')
        plt.show()

    assert t.detrend('mad-tilt').mad_height() < t.mad_height()
    assert t.detrend('mad-curvature').mad_height() < t.detrend('mad-tilt').mad_height()


# @pytest.mark.parametrize('fn', ['di-1.di', 'plux-1.plux'])
@pytest.mark.parametrize('fn', ['plux-1.plux'])
def test_mad_detrending_topography(file_format_examples, fn, plot=False):
    file_path = os.path.join(file_format_examples, fn)
    t = read_topography(file_path)

    if plot:
        import matplotlib.pyplot as plt

        coeffs = [0, 1.39556151, 2.29554222]

        x = np.linspace(coeffs[2], coeffs[2] + 0.0002, 2)
        # rms = [t.detrend(coeffs=[0, coeffs[1], _x]).rms_height_from_profile() for _x in x]
        dt = [t.detrend(coeffs=[0, coeffs[1], _x]) for _x in x]
        mad = [_dt.mad_height() for _dt in dt]

        plt.figure()
        # plt.plot(x, rms, 'k-')
        plt.plot(x, mad, 'rx-')
        plt.show()

        plt.figure()
        for t in dt:
            ba = t.bearing_area()
            x = np.linspace(ba.min, ba.max, 101)
            print(ba(x))
            plt.plot(x, ba(x), 'k-')
        plt.show()

    assert t.detrend('mad-tilt').mad_height() < t.mad_height()
    # mad-curvature is currently too slow to be useful
    # assert t.detrend('mad-curvature').mad_height() < t.detrend('mad-tilt').mad_height()


@pytest.mark.parametrize('fn,ref_mad_height', [['opd-2.opd', 0.013786078653533923],
                                               ['plux-1.plux', 1.709517749823731]])
def test_undefined_data(file_format_examples, fn, ref_mad_height):
    file_path = os.path.join(file_format_examples, fn)
    t = read_topography(file_path)
    assert t.has_undefined_data
    print(t.mad_height())
    np.testing.assert_allclose(t.mad_height(), ref_mad_height)
