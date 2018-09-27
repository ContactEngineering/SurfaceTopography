#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ScalarParameters.py

@author Till Junge <till.junge@kit.edu>

@date   11 Feb 2015

@brief  Bin for small common helper function and classes

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

from .common import compute_derivative


def rms_height(profile, kind='Sq'):
    "computes the rms height fluctuation of the surface"
    if kind == 'Sq':
        return np.sqrt(((profile[...]-profile[...].mean())**2).mean())
    elif kind == 'Rq':
        return np.sqrt(((profile[...]-profile[...].mean(axis=0))**2).mean())
    else:
        raise RuntimeError("Unknown rms height kind '{}'.".format(kind))


def rms_slope(profile, size=None, dim=None):
    "computes the rms height gradient fluctuation of the surface"
    diff = compute_derivative(profile, size, dim)
    return np.sqrt((diff[0]**2).mean()+(diff[1]**2).mean())


def rms_curvature(profile, size=None, dim=None):
    """
    computes the rms Laplacian of the surface
    the rms mean-curvature would be half of this
    """
    curv = compute_derivative(profile, size, dim, n=2)
    return np.sqrt(((curv[0][:, 1:-1]+curv[1][1:-1, :])**2).mean())