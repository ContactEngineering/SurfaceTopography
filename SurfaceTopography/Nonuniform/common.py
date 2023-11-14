#
# Copyright 2018-2021 Lars Pastewka
#           2019 Antoine Sanner
#           2019 Michael Röttger
#           2015-2016 Till Junge
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
Bin for small common helper function and classes for nonuniform
topographies.
"""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface


def bandwidth(self):
    """
    Computes lower and upper bound of bandwidth, i.e. of the wavelengths or
    length scales occurring on a topography. The lower end of the bandwidth is
    given by the mean of the spacing of the individual points on the line
    scan. The upper bound is given by the overall length of the line scan.

    Returns
    -------
    lower_bound : float
        Lower bound of the bandwidth.
    upper_bound : float
        Upper bound of the bandwidth.
    """
    x = self.positions()
    lower_bound = np.mean(np.diff(x))
    upper_bound, = self.physical_sizes

    return lower_bound, upper_bound


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('bandwidth', bandwidth)
