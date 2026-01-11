#
# Copyright 2020, 2024 Lars Pastewka
#           2020 Antoine Sanner
#           2015 Till Junge
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
Robust statistics (median, median absolute deviation, etc.) for topography data.
"""

from functools import cached_property

import scipy

from SurfaceTopography.HeightContainer import (NonuniformLineScanInterface,
                                               UniformTopographyInterface)

_rms_percentile = scipy.stats.norm.cdf(-1)


def bisect(bearing_area, target_area, rtol=1e-6):
    """
    Find the root of a function using the bisection method.

    Parameters
    ----------
    bearing_area : :obj:`UniformBearingArea` or :obj:`NonuniformBearingArea`
        Instance of bearing area class.
    target_area : float
        Target bearing area.
    rtol : float
        Relative tolerance for the root with respect to the initial bounds.

    Returns
    -------
    root : float
        Height where the bearing area is equal to `target_area`.
    """
    a = bearing_area.min
    b = bearing_area.max
    initial_ab = b - a
    fa_lower, fa_upper = bearing_area.bounds(a)
    fb_lower, fb_upper = bearing_area.bounds(b)
    assert fa_upper >= target_area >= fb_lower
    while b - a > initial_ab * rtol:
        c = (a + b) / 2
        fc_lower, fc_upper = bearing_area.bounds(c)  # This is O(log(N))
        if fc_lower < target_area < fc_upper:
            fc_lower = fc_upper = bearing_area(c)  # We need the exact value of the bearing area (this is O(N))
        if fc_upper < target_area:
            b, fb_lower, fb_upper = c, fc_lower, fc_upper  # noqa: F841
        elif fc_lower > target_area:
            a, fa_lower, fa_upper = c, fc_lower, fc_upper  # noqa: F841
        else:
            return c
    return (a + b) / 2


def median(self):
    """
    Compute the median height of the topography.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.

    Returns
    -------
    median : float
        Median height.
    """
    # The median is where the fractional bearing area is 1/2
    return bisect(self.bearing_area(), 0.5)


def mad_height(self, percentile=_rms_percentile):
    """
    Compute the median-absolute-deviation of the height.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.
    percentile : float
        Fraction of the bearing area that should be considered for the
        deviation. (Default: One standard deviation)

    Returns
    -------
    mad_height : float
        Median absolute deviation of the height.
    """

    class MAD:
        def __init__(self, bearing_area):
            self._median = bisect(bearing_area, 0.5)
            self._bearing_area = bearing_area

        def __call__(self, h):
            return self._bearing_area(self._median + h) + (1 - self._bearing_area(self._median - h))

        def bounds(self, h):
            lp, up = self._bearing_area.bounds(self._median + h)
            lm, um = self._bearing_area.bounds(self._median - h)
            return lp + (1 - um), up + (1 - lm)

        @cached_property
        def min(self):
            return 0

        @cached_property
        def max(self):
            return max(bearing_area.max - self._median, self._median - bearing_area.min)

    bearing_area = self.bearing_area()
    # Compute median

    # The median absolute deviation is where the fractional bearing area is equal to `percentile`. We need to consider
    # fluctuation in positive and negative directions.
    return bisect(MAD(self.bearing_area()), 2 * percentile)


def mad_polyfit(self, nb_coeffs):
    """
    Find the polynomial that minimizes the median absolute deviation.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.
    nb_coeffs : int
        Number of coefficients to be fitted.
    """

    def mad(coeffs):
        return self.detrend(coeffs=[0, *coeffs]).mad_height()

    x0 = self.polyfit(nb_coeffs)
    coeffs = scipy.optimize.minimize(mad, x0=x0[1:], method='Nelder-Mead').x
    return [self.detrend(coeffs=[0, *coeffs]).median(), *coeffs]


UniformTopographyInterface.register_function('median', median)
UniformTopographyInterface.register_function('mad_height', mad_height)
UniformTopographyInterface.register_function('mad_polyfit', mad_polyfit)
NonuniformLineScanInterface.register_function('median', median)
NonuniformLineScanInterface.register_function('mad_height', mad_height)
NonuniformLineScanInterface.register_function('mad_polyfit', mad_polyfit)
