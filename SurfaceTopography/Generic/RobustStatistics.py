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
Robust statistics (median, median absolute deviation, etc.) for topography
data. All quantities are obtained by inverting the bearing area curve of the
piecewise-linear interpolation of the data.
"""

from functools import cached_property

import scipy

from SurfaceTopography.HeightContainer import (NonuniformLineScanInterface,
                                               UniformTopographyInterface)

_rms_percentile = scipy.stats.norm.cdf(-1)


def bisect(bearing_area, target_area, rtol=1e-6):
    """
    Find the height at which the bearing area equals a target value using the
    bisection method.

    Each bisection step first evaluates the O(log(N)) bounds of the bearing
    area and only evaluates the O(N) exact bearing area if the target lies
    between these bounds.

    Parameters
    ----------
    bearing_area : :obj:`UniformBearingArea` or :obj:`NonuniformBearingArea`
        Instance of bearing area class.
    target_area : float
        Target bearing area.
    rtol : float, optional
        Relative tolerance for the root with respect to the initial bracket
        (the height range of the data). (Default: 1e-6)

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
    Compute the median height of the topography, i.e. the height at which
    the fractional bearing area equals 1/2. The median is computed for the
    piecewise-linear interpolation of the data (not as the median of the
    data points).

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
    r"""
    Compute a robust estimate of the rms height from the distribution of
    absolute deviations from the median.

    The function returns the deviation :math:`\delta` for which a fraction
    :math:`2p` of the projected area deviates from the median height
    :math:`h_\text{med}` by more than :math:`\delta`, i.e. it solves

    .. math::

        \Phi(h_\text{med} + \delta) + 1 - \Phi(h_\text{med} - \delta) = 2p,

    where :math:`\Phi(h)` is the fractional bearing area (area above
    :math:`h`) and :math:`p` is `percentile`. The default
    :math:`p = \Phi_\mathcal{N}(-1) \approx 0.159` (with :math:`\Phi_\mathcal{N}`
    the cumulative distribution function of the standard normal distribution)
    makes :math:`\delta` equal to the standard deviation for Gaussian
    distributed heights. Setting :math:`p = 1/4` yields the classical median
    absolute deviation.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.
    percentile : float, optional
        Fraction :math:`p` of the bearing area on each side of the median
        that lies outside the deviation. (Default: scipy.stats.norm.cdf(-1),
        i.e. one standard deviation)

    Returns
    -------
    mad_height : float
        Deviation from the median height. (Units: length)
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
    Find the polynomial that minimizes the median absolute deviation
    (as computed by :func:`mad_height`) of the residual heights.

    The non-constant coefficients are optimized with the Nelder-Mead method,
    starting from the least-squares fit `polyfit`; the constant coefficient is
    the median of the resulting residual.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.
    nb_coeffs : int
        Degree of the polynomial; this is passed as `deg` to `polyfit`. (For
        a line scan, 1 fits a straight line, yielding two coefficients.)

    Returns
    -------
    coeffs : list of float
        Polynomial coefficients, in the order used by `polyfit` and
        `detrend(coeffs=...)`.
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
