#
# Copyright 2020-2021 Lars Pastewka
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

import scipy.interpolate

from ..Exceptions import ReentrantDataError
from ..HeightContainer import NonuniformLineScanInterface


def interpolate_linear(self):
    r"""
    Returns a linear interpolation function based on the topography's heights.
    """
    if self.is_reentrant:
        raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). Interpolation is not '
                                 'possible for reentrant topographies.')
    return scipy.interpolate.interp1d(*self.positions_and_heights(), kind='linear')


def interpolate_cubic(self):
    r"""
    Returns a linear interpolation function based on the topography's heights.
    """
    if self.is_reentrant:
        raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). Interpolation is not '
                                 'possible for reentrant topographies.')
    return scipy.interpolate.interp1d(*self.positions_and_heights(), kind='cubic')


NonuniformLineScanInterface.register_function('interpolate_cubic', interpolate_cubic)
NonuniformLineScanInterface.register_function('interpolate_linear', interpolate_linear)
