#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file   Autocorrelation.py
#
# @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
#
# @date   26 Feb 2019
#
# @brief  Height-difference autocorrelation functions for nonuniform line scans
#
# @section LICENCE
#
# Copyright 2015-2017 Till Junge, Lars Pastewka
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

from ..NonuniformLineScan import NonuniformLineScan


def autocorrelation_1D(topography, distances=None):
    r"""
    Compute the one-dimensional height-difference autocorrelation function (ACF).

    Parameters
    ----------
    topography : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    r : array_like
        Array containing distances for which to compute the ACF. If no array
        is given, the function will automatically construct an array with
        equally spaced distances. (Default: None)

    Returns
    -------
    distances : array
        Distances. (Units: length)
    A : array
        Autocorrelation function. (Units: length**2)
    """
    if distances is None:
        # FIXME!!! We need a better heuristics to decide on the distances
        r = np.linspace(0, topography.x_range, topography.resolution)
    A = np.zeros_like(distances)

    x, y = topograhy.positions_and_heights()
    s = topography.derivative(1)
    for k in range(len(distances)):
        d = distances[k]
        for i in range(len(x)-1):
            for j in range(i+1, len(x)-1):
                # Determine lower and upper distance between segment i, i+1 and
                # segment j, j+1
                lower_d = max(x[i], x[j]-d)
                upper_d = min(x[i+1], x[j+1]-d)
                acf = ((y[j] + s[j] * (upper_d + d - x[j]) - x[i] - s[i] * (upper_d - x[i])) ** 3 - (
                            y[j] + s[j] * (lower_d + d - x[j]) - x[i] - s[i] * (lower_d - x[i])) ** 3) / (
                                  6 * (s[j] - s[i]))
                A[k] += acf
    A /= topography.x_range - distances
    return distances, A

### Register analysis functions from this module

NonuniformLineScan.register_function('autocorrelation_1D', autocorrelation_1D)
