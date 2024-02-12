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

from scipy.optimize import bisect

from SurfaceTopography.HeightContainer import (NonuniformLineScanInterface,
                                               UniformTopographyInterface)


def median(self):
    """
    Compute the median height of the topography.

    Parameters
    ----------
    self : :obj:`HeightContainer`
        Topography or line scan container object.
    """
    # The median is where the fractional bearing area is 1/2
    return bisect(lambda h: self.bearing_area(h) - 0.5, self.min(), self.max())


UniformTopographyInterface.register_function('median', median)
NonuniformLineScanInterface.register_function('median', median)
