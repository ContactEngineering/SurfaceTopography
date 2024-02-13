#
# Copyright 2024 Lars Pastewka
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

import numpy as np
import scipy

from SurfaceTopography.HeightContainer import (NonuniformLineScanInterface,
                                               UniformTopographyInterface)

_rms_percentile = scipy.stats.norm.cdf(-1)


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
    tmp = self.squeeze()  # Avoid evaluating pipeline during bisect
    # The median is where the fractional bearing area is 1/2
    return scipy.optimize.bisect(lambda h: tmp.bearing_area(h) - 0.5, tmp.min(), tmp.max())


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
    tmp = self.squeeze()  # Avoid evaluating pipeline during bisect
    # Compute median
    med = tmp.median()
    # The median absolute deviation is where the fractional bearing area is equal to `percentile`. We need to consider
    # fluctuation in positive and negative directions.
    maxdev = max(tmp.max() - med, med - tmp.min())
    return scipy.optimize.bisect(lambda h: tmp.bearing_area(med + h) + (1 - tmp.bearing_area(med - h)) - 2 * percentile,
                                 -maxdev, maxdev)


def polynomial_that_minimizes_mad_height(self, nb_coeffs):
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

    coeffs = scipy.optimize.minimize(mad, x0=np.zeros(nb_coeffs), method='Nelder-Mead').x
    return [self.detrend(coeffs=[0, *coeffs]).median(), *coeffs]


UniformTopographyInterface.register_function('median', median)
UniformTopographyInterface.register_function('mad_height', mad_height)
UniformTopographyInterface.register_function('mad_polyfit',
                                             polynomial_that_minimizes_mad_height)
NonuniformLineScanInterface.register_function('median', median)
NonuniformLineScanInterface.register_function('mad_height', mad_height)
NonuniformLineScanInterface.register_function('mad_polyfit',
                                              polynomial_that_minimizes_mad_height)
