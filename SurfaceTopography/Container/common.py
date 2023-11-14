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
Bin for small common helper function and classes for containers.
"""

import SurfaceTopography.Support.UnitConversion as UnitConversion
from .SurfaceContainer import SurfaceContainer


def bandwidth(self, unit):
    """
    Computes lower and upper bound of bandwidth, i.e. of the wavelengths or
    length scales occurring on any topography within the container. The lower
    end of the bandwidth is given by the mean of the spacing of the individual
    points on the line scan. The upper bound is given by the overall length of
    the line scan.

    Parameters
    ----------
    self : SurfaceContainer
        Container object containing the underlying data sets.
    unit : str
        Unit for reporting the bandwidths.

    Returns
    -------
    lower_bound : float
        Lower bound of the bandwidth.
    upper_bound : float
        Upper bound of the bandwidth.
    """
    global_lower = global_upper = None
    for topography in self:
        current_lower, current_upper = topography.bandwidth()
        fac = UnitConversion.get_unit_conversion_factor(topography.unit, unit)
        global_lower = current_lower * fac if global_lower is None else min(global_lower, current_lower * fac)
        global_upper = current_upper * fac if global_upper is None else max(global_upper, current_upper * fac)

    return global_lower, global_upper


def suggest_length_unit(self, scale):
    """
    Compute a suggestion for a unit to diplay container-wider information.
    The unit is chose to minimize number of digits to the left and right of
    the decimal point.

    Parameters
    ----------
    scale : str
        'linear': displaying data on a linear axis
        'log' displaying data on a log-space axis

    Returns
    -------
    unit : str
        Suggestion for the length unit
    """
    return UnitConversion.suggest_length_unit(scale, *bandwidth(self, unit='m'))


# Register analysis functions from this module
SurfaceContainer.register_function('bandwidth', bandwidth)
SurfaceContainer.register_function('suggest_length_unit', suggest_length_unit)
