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

from ..HeightContainer import UniformTopographyInterface


class UniformBearingArea:
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
        el_min = np.searchsorted(self._el_min_heights, heights)
        el_max = np.searchsorted(self._el_max_heights, heights)
        return (self._nb_els - el_min) / self._nb_els, (self._nb_els - el_max) / self._nb_els

    @cached_property
    def min(self):
        return np.nanmin(self._h)  # There are NaNs if the original topography was masked

    @cached_property
    def max(self):
        return np.nanmax(self._h)  # There are NaNs if the original topography was masked


class Uniform1DBearingArea(UniformBearingArea):
    """
    Accelerated bearing area calculation for uniform line scans.
    """

    def __init__(self, h, is_periodic):
        self._is_periodic = is_periodic

        if self._is_periodic:
            self._el_min_heights = np.sort(np.ma.compressed(np.minimum(h, np.roll(h, -1))))
            self._el_max_heights = np.sort(np.ma.compressed(np.maximum(h, np.roll(h, -1))))
        else:
            self._el_min_heights = np.sort(np.ma.compressed(np.minimum(h[:-1], h[1:])))
            self._el_max_heights = np.sort(np.ma.compressed(np.maximum(h[:-1], h[1:])))
        self._nb_els = len(self._el_min_heights)  # This treats undefined data correctly
        self._h = np.ma.filled(h.astype(float), fill_value=np.nan)  # We pass the mask as NaNs to the C++ code

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
            return _SurfaceTopographyPP.uniform1d_bearing_area(self._h, self._is_periodic,
                                                               np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.uniform1d_bearing_area(self._h, self._is_periodic,
                                                               heights.astype(float))


class Uniform2DBearingArea(UniformBearingArea):
    """
    Accelerated bearing area calculation for topographies.
    """

    def __init__(self, h, is_periodic):
        self._is_periodic = is_periodic

        if self._is_periodic:
            self._el_min_heights = np.sort(np.ma.compressed([
                np.minimum(np.minimum(h, np.roll(h, (-1, 0))), np.roll(h, (0, -1))),
                np.minimum(np.minimum(np.roll(h, (-1, -1)), np.roll(h, (-1, 0))), np.roll(h, (0, -1)))
            ]))
            self._el_max_heights = np.sort(np.ma.compressed([
                np.maximum(np.maximum(h, np.roll(h, (-1, 0))), np.roll(h, (0, -1))),
                np.maximum(np.maximum(np.roll(h, (-1, -1)), np.roll(h, (-1, 0))), np.roll(h, (0, -1)))
            ]))
        else:
            self._el_min_heights = np.sort(np.ma.compressed([
                np.minimum(np.minimum(h[:-1, :-1], h[1:, :-1]), h[:-1, 1:]),
                np.minimum(np.minimum(h[1:, 1:], h[1:, :-1]), h[:-1, 1:])
            ]))
            self._el_max_heights = np.sort(np.ma.compressed([
                np.maximum(np.maximum(h[:-1, :-1], h[1:, :-1]), h[:-1, 1:]),
                np.maximum(np.maximum(h[1:, 1:], h[1:, :-1]), h[:-1, 1:])
            ]))
        self._nb_els = len(self._el_min_heights)  # This treats undefined data correctly
        # We pass the mask as NaNs to the C++ code
        self._h = np.ascontiguousarray(np.ma.filled(h, fill_value=np.nan), dtype=float)

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
            return _SurfaceTopographyPP.uniform2d_bearing_area(self._h, self._is_periodic,
                                                               np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.uniform2d_bearing_area(self._h, self._is_periodic,
                                                               heights.astype(float))


def bearing_area(self, heights=None):
    """
    Compute the bearing area for a specific height.

    The bearing area as a function of height is also known as the
    Abbott-Firestone curve. If expressed as a fractional area, it is the
    cumulative distribution function of the surface heights. This function
    returns this fractional area.

    Parameters
    ----------
    self : :obj:`UniformLineScan` or :obj:`Topography`
        Topography or line scan container object.
    heights : float or np.ndarray, optional
        Heights for which to compute the bearing area. (Default: None)

    Returns
    -------
    fractional_area : float or np.ndarray
        Fractional area above the threshold height, if height is given.
    bearing_area : :obj:`Uniform1DBearingArea` or :obj:`Uniform2DBearingArea`
        Instance of a class that caches the bearing area calculation.
    """
    if self.dim == 1:
        b = Uniform1DBearingArea(self.heights(), self.is_periodic)
    elif self.dim == 2:
        b = Uniform2DBearingArea(self.heights(), self.is_periodic)
    else:
        raise NotImplementedError('Bearing area is only implemented for 1D line scans and 2D topography maps.')

    if heights is None:
        return b
    else:
        return b(heights)


UniformTopographyInterface.register_function('bearing_area', bearing_area)
