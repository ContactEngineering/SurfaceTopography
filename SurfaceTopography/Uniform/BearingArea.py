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


def bearing_area(self, heights):
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
    heights : float or np.ndarray
        Heights for which to compute the bearing area.

    Returns
    -------
    fractional_area : float or np.ndarray
        Fractional area above a the threshold height.
    """
    if self.dim == 1:
        h = self.heights()
        dx, = self.pixel_size
        if np.isscalar(heights):
            return _SurfaceTopographyPP.uniform1d_bearing_area(dx, h.astype(float), self.is_periodic,
                                                               np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopographyPP.uniform1d_bearing_area(dx, h.astype(float), self.is_periodic,
                                                               heights.astype(float))
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


UniformTopographyInterface.register_function('bearing_area', bearing_area)
