#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   common.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   11 Dec 2018

@brief  Bin for small common helper function and classes for uniform
        topographies.

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from ..UniformLineScanAndTopography import Topography, UniformLineScan


def derivative(topography, n):
    """
    Compute derivative of topography or line scan stored on a uniform grid.

    Parameters
    ----------
    topography : Topography or UniformLineScan
        Topography object containing height information.
    n : int
        Number of times the derivative is taken.

    Returns
    -------
    derivative : array
        Array with derivative values. If dimension of the topography is
        unity (line scan), then an array of the same shape as the
        topography is returned. Otherwise, the first array index contains
        the direction of the derivative. If the topgography is nonperiodic,
        then all returning array with have shape one less than the input
        arrays.
    """
    grid_spacing = topography.pixel_size
    heights = topography.heights()
    if topography.is_periodic:
        if n != 1:
            raise ValueError('Only first derivatives are presently supported for periodic topographies.')
        d = np.array([(np.roll(heights, axis=d) - heights) / grid_spacing[d] ** n
                      for d in range(len(heights.shape))])
    else:
        d = np.array([np.diff(heights, n=n, axis=d) / grid_spacing[d] ** n
                      for d in range(len(heights.shape))])
    if d.shape[0] == 1:
        return d[0]
    else:
        return d


### Register analysis functions from this module

Topography.register_function('derivative', derivative)

UniformLineScan.register_function('derivative', derivative)
