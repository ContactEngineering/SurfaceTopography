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
Bearing area curve, also known as Abbott-Firestone curve or cumulative
distribution function of the surface heights.
"""

from functools import cached_property

import _SurfaceTopographyPP
import numpy as np

from ..HeightContainer import NonuniformLineScanInterface


class NonuniformBearingArea:
    """
    Accelerated bearing area calculation for nonuniform line scans.
    """

    def __init__(self, x, h):
        self._x = np.asanyarray(x, dtype=float)
        self._h = np.asanyarray(h, dtype=float)

        # Element indices, sorted by min height of element
        self._el_sort_by_min = np.argsort(np.minimum(self._h[:-1], self._h[1:]))
        # Element indices, sorted by max height of element
        self._el_sort_by_max = np.argsort(np.maximum(self._h[:-1], self._h[1:]))

        # Cumulative sum of element widths
        max_width = self._x[-1] - self._x[0]
        el_width = np.diff(self._x)
        self._cum_width_min = np.append(1, 1 - np.cumsum(el_width[self._el_sort_by_min]) / max_width)
        self._cum_width_max = np.append(1, 1 - np.cumsum(el_width[self._el_sort_by_max]) / max_width)

    def __call__(self, heights):
        """
        Compute the bearing area for a specific height.

        The bearing area as a function of height is also known as the
        Abbott-Firestone curve. If expressed as a fractional area, it is the
        cumulative distribution function of the surface heights. This function
        returns this fractional area.

        The function has complexity O(N) where N is the number of elements in


        Parameters
        ----------
        heights : float or np.ndarray
            Heights for which to compute the bearing area.

        Returns
        -------
        fractional_area : float or np.ndarray
            Fractional area above a the threshold height.
        """
        if np.isscalar(heights):
            return _SurfaceTopographyPP.nonuniform_bearing_area(self._x, self._h, self._el_sort_by_max,
                                                                np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.nonuniform_bearing_area(self._x, self._h, self._el_sort_by_max,
                                                                np.asanyarray(heights, dtype=float))

    def bounds(self, heights):
        """
        Compute bounds on the bearing area for a specific height. These bounds
        can be computed O(log(N)) time, where N is the number of elements in the
        line scan.

        Parameters
        ----------
        heights : float or np.ndarray
            Heights for which to compute the bearing area.

        Returns
        -------
        lower_bound : float or np.ndarray
            Lower bound on fractional area above a threshold height.
        upper_bound : float or np.ndarray
            Upper bound on fractional area above a threshold height.
        """
        el_min_heights = np.minimum(self._h[:-1], self._h[1:])
        el_max_heights = np.maximum(self._h[:-1], self._h[1:])
        el_min = np.searchsorted(el_min_heights[self._el_sort_by_min], heights)
        el_max = np.searchsorted(el_max_heights[self._el_sort_by_max], heights)
        return self._cum_width_min[el_min], self._cum_width_max[el_max]

    @cached_property
    def min(self):
        return self._h.min()

    @cached_property
    def max(self):
        return self._h.max()


def bearing_area(self, heights=None):
    """
    Compute the bearing area for a specific height.

    The bearing area as a function of height is also known as the
    Abbott-Firestone curve. If expressed as a fractional area, it is the
    cumulative distribution function of the surface heights. This function
    returns this fractional area.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Line scan container object.
    heights : float or np.ndarray, optional
        Heights for which to compute the bearing area. (Default: None)

    Returns
    -------
    fractional_area : float or np.ndarray
        Fractional area above the threshold height, if height is given.
    bearing_area : :obj:`NonuniformBearingArea`
        Instance of :obj:`NonuniformBearingArea` class that caches the bearing area
        calculation.
    """
    b = NonuniformBearingArea(*self.positions_and_heights())
    if heights is None:
        return b
    else:
        return b(heights)


NonuniformLineScanInterface.register_function('bearing_area', bearing_area)
