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

import numpy as np
from ..HeightContainer import NonuniformLineScanInterface


def bearing_area(self, height):
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
    height : float
        Height for which to compute the bearing area.

    Returns
    -------
    fractional_area : float
        Fractional area above a the threshold height.
    """
    # FIXME! This is likely not efficient and is a candidate to be implemented in C++
    x, h = self.positions_and_heights()
    dx = np.diff(x).reshape(-1, 1)
    if len(x) <= 1:
        return 0.0
    L = x[-1] - x[0]
    minh = np.minimum(h[:-1], h[1:]).reshape(-1, 1)
    maxh = np.maximum(h[:-1], h[1:]).reshape(-1, 1)
    h = height.reshape(1, -1)
    return np.sum(dx * ((h < minh) + (h - minh) * np.logical_and(h > minh, h < maxh)), axis=0) / L


NonuniformLineScanInterface.register_function('bearing_area', bearing_area)
