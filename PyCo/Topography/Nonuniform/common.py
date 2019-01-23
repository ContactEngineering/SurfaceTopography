#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   common.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   11 Dec 2018

@brief  Bin for small common helper function and classes for nonuniform
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

def _derivative(x, h, n=1):
    """
    Compute derivative of nonuniform line-scan. Function assumes nonperiodic
    topographies.

    Parameters
    ----------
    x : array
        X-coordinates.
    h : array
        Y- or height coordinates.
    n : int
        Order of derivative.

    Returns
    -------
    derivative : array
        Array with derivative values. Length of array is reduced by one with
        respect to the input array.
    """
    if n != 1:
        raise RuntimeError('Currently only first derivatives are supported for nonuniform topographies.')
    return np.diff(h) / np.diff(x)
