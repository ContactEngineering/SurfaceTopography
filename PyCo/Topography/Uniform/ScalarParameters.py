#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ScalarParameters.py

@author Till Junge <till.junge@kit.edu>

@date   11 Feb 2015

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

from ..UniformLineScanAndTopography import Topography, UniformLineScan


def rms_height(topography, kind='Sq'):
    """
    Compute the root mean square height amplitude of a topography or
    line scan stored on a uniform grid.

    Parameters
    ----------
    topography : Topography or UniformLineScan
        Topography object containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value.
    """
    n = np.prod(topography.resolution)
    #if topography.is_MPI:
    pnp = topography.pnp
    profile = topography.heights()
    if kind == 'Sq':
        return np.sqrt(
            pnp.sum((profile - pnp.sum(profile) / n) ** 2) / n)
    elif kind == 'Rq':
        decomp_axis = [full != loc for full, loc in
                       zip(np.array(topography.resolution), profile.shape)]
        temppnp = pnp if decomp_axis[0] == True else np
        return np.sqrt(temppnp.sum(
            (profile - temppnp.sum(profile, axis=0)
                / topography.resolution[0]) ** 2
                                    ) / n)
    else:
        raise RuntimeError("Unknown rms height kind '{}'.".format(kind))


def rms_slope(topography):
    """
    Compute the root mean square amplitude of the height gradient of a
    topography or line scan stored on a uniform grid.

    Parameters
    ----------
    topography : Topography or UniformLineScan
        Topography object containing height information.

    Returns
    -------
    rms_slope : float
        Root mean square slope value.
    """
    if topography.is_MPI:
        raise NotImplementedError("rms_slope not implemented for parallelized topographies")
    diff = topography.derivative(1)
    return np.sqrt((diff[0]**2).mean()+(diff[1]**2).mean())


def rms_Laplacian(topography):
    """
    Compute the root mean square Laplacian of the height gradient of a
    topography or line scan stored on a uniform grid. The rms curvature
    is half of the value returned here.

    Parameters
    ----------
    topography : Topography or UniformLineScan
        Topography object containing height information.

    Returns
    -------
    rms_laplacian : float
        Root mean square Laplacian value.
    """
    if topography.is_MPI:
        raise NotImplementedError("rms_Laplacian not implemented for parallelized topographies")

    curv = topography.derivative(2)
    return np.sqrt(((curv[0][:, 1:-1]+curv[1][1:-1, :])**2).mean())


### Register analysis functions from this module

Topography.register_function('rms_height', rms_height)
Topography.register_function('rms_slope', rms_slope)
Topography.register_function('rms_curvature', rms_Laplacian)

UniformLineScan.register_function('rms_height', rms_height)
UniformLineScan.register_function('rms_slope', rms_slope)
UniformLineScan.register_function('rms_curvature', rms_Laplacian)
