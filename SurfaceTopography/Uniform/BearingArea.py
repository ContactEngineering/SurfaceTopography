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

import _SurfaceTopographyPP
import numpy as np

from ..HeightContainer import UniformTopographyInterface


class Uniform1DBearingArea:
    """
    Accelerated bearing area calculation for nonuniform line scans.
    """

    def __init__(self, dx, h, is_periodic):
        self._dx = dx
        self._h = np.asanyarray(h, dtype=float)
        self._is_periodic = is_periodic

        if is_periodic:
            # Element indices, sorted by min height of element
            self._el_sort_by_min = np.argsort(np.minimum(np.roll(self._h, -1), np.roll(self._h, 1)))
            # Element indices, sorted by max height of element
            self._el_sort_by_max = np.argsort(np.maximum(np.roll(self._h, -1), np.roll(self._h, 1)))
        else:
            # Element indices, sorted by min height of element
            self._el_sort_by_min = np.argsort(np.minimum(self._h[:-1], self._h[1:]))
            # Element indices, sorted by max height of element
            self._el_sort_by_max = np.argsort(np.maximum(self._h[:-1], self._h[1:]))

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
            return _SurfaceTopographyPP.uniform1d_bearing_area(self._dx, self._h, self._is_periodic,
                                                               np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.uniform1d_bearing_area(self._dx, self._h, self._is_periodic,
                                                               heights.astype(float))

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
        if self._is_periodic:
            el_min_heights = np.minimum(np.roll(self._h, -1), np.roll(self._h, 1))
            el_max_heights = np.maximum(np.roll(self._h, -1), np.roll(self._h, 1))
        else:
            el_min_heights = np.minimum(self._h[:-1], self._h[1:])
            el_max_heights = np.maximum(self._h[:-1], self._h[1:])
        el_min = np.searchsorted(el_min_heights[self._el_sort_by_min], heights)
        el_max = np.searchsorted(el_max_heights[self._el_sort_by_max], heights)
        if self._is_periodic:
            nb_els = len(self._h)
        else:
            nb_els = len(self._h) - 1
        return (nb_els - el_min) / nb_els, (nb_els - el_max) / nb_els


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
        b = Uniform1DBearingArea(self.pixel_size[0], self.heights(), self.is_periodic)
    elif self.dim == 2:
        h = self.heights()
        dx, dy = self.pixel_size
        if np.isscalar(heights):
            return _SurfaceTopographyPP.uniform2d_bearing_area(dx, dy, np.ascontiguousarray(h, dtype=float),
                                                               self.is_periodic,
                                                               np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.uniform2d_bearing_area(dx, dy, np.ascontiguousarray(h, dtype=float),
                                                               self.is_periodic,
                                                               heights.astype(float))
    else:
        raise NotImplementedError('Bearing area is only implemented for 1D line scans and 2D topography maps.')

    if heights is None:
        return b
    else:
        return b(heights)


UniformTopographyInterface.register_function('bearing_area', bearing_area)
