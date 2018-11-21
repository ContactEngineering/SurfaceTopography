#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ScalarParameters.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Nov 2018

@brief  Functions computing scalar roughness parameters

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


def rms_height(x, h):
    """
    Computes root-mean square height fluctuation of the line scan:

    .. math:: h_\text{rms} = \left[ \frac{1}{L} \int_0^L dx\, h^2(x) \right]^{1/2}

    Function approximates topography between data points as piece-wise linear.
    The piece-wise linear section between point :math:`i` and point
    :math:`i+1` contributes

    .. math:: \int_{0}^{\Delta x_i} dx\, \left( h_i + \frac{\Delta h_i}{\Delta x_i} x \right)^2 = \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i

    where :math:`\Delta x_i=x_{i+1}-x_i` and :math:`\Delta h_i=h_{i+1}-h_i` to the above integral.

    Parameters
    ----------
    x : array_like
        Array containing positions. This function assumes that this array is
        sorted in ascending order.
    h : array_like
        Array containing heights

    Returns
    -------
    rms_height : float
        Root-mean square height
    """
    dx = np.diff(x)
    L = x[-1] - x[0]
    mean_h = np.trapz(h, x)/L
    h0 = h - mean_h

    return np.sqrt(np.sum((h0[:-1]**2 + h0[1:]**2 + h0[:-1]*h0[1:])*dx)/(3*L))