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


def rms_height(profile, kind='Sq' , resolution = None,pnp = np):
    """computes the rms height fluctuation of the surface

    Parameters
    ----------
    profile: np.array, :obj: `UniformTopography`
    kind: {'Sq','Rq'} , optional
    resolution: tuple or list, optional
    pnp: module or :obj: ParallelNumpy, optional


    Returns
    -------
    float
    ..math \sqrt{\Sigma_i (h_i - <h_i>)^2}

    MPI Parallelisation:

    When profile represents only a part of the data (attributed to the processor), you should provide the resolution of
    the full data with `resolution` and `pnp`.

    """
    if hasattr(profile,"rms_height"):
        return profile.rms_height(kind=kind)

    if resolution == None: # then profile will be interpreted as the total data
        resolution = profile.shape

    n=np.prod(resolution)

    # check if data is decomposed between processors or not in each direction.
    # TODO: maybe handle this in pnp already ?
    decomp_axis = [full != loc for full, loc in zip(np.array(resolution),profile.shape)]

    if kind == 'Sq':
        return np.sqrt(pnp.sum((profile-pnp.sum(profile) / n )**2)/n)
    elif kind == 'Rq':
        temppnp = pnp if decomp_axis[0] == True else np
        return np.sqrt(temppnp.sum((profile-temppnp.sum(profile,axis=0) / resolution[0])**2) / n)
    else:
        raise RuntimeError("Unknown rms height kind '{}'.".format(kind))


def rms_slope(profile, size=None, dim=None):
    "computes the rms height gradient fluctuation of the surface"
    if hasattr(profile,"rms_slope"):
        return profile.rms_slope()

    diff = compute_derivative(profile, size, dim)
    return np.sqrt((diff[0]**2).mean()+(diff[1]**2).mean())


def rms_curvature(profile, size=None, dim=None):
    """
    computes the rms Laplacian of the surface
    the rms mean-curvature would be half of this
    """
    if hasattr(profile,"rms_curvature"):
        return profile.rms_curvature()

    curv = compute_derivative(profile, size, dim, n=2)
    return np.sqrt(((curv[0][:, 1:-1]+curv[1][1:-1, :])**2).mean())


def rms_height_nonuniform(x, y, kind='Sq'):
    raise NotImplementedError


def rms_slope_nonuniform(x, y):
    raise NotImplementedError


def rms_curvature_nonuniform(x, y):
    raise NotADirectoryError