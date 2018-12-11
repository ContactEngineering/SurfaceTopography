#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Detrending.py

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
import scipy

from .common import _get_size


def tilt_from_height(arr, size=None, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()
    """
    size = _get_size(arr, size)
    arr = arr[...]
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    #res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])*\
        np.array(list(arr.shape)+[1.])/np.array(list(size)+[1.])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def tilt_from_slope(arr, size=None):
    return [x.mean() for x in compute_derivative(arr, size)]


def tilt_and_curvature(arr, size=None, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()
    """
    size = _get_size(arr, size)
    arr = arr[...]
    nb_dim = len(arr.shape)
    assert nb_dim == 2
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    # Linear terms
    x_grids = np.meshgrid(*x_grids, indexing='ij')
    # Quadratic terms
    x, y = x_grids
    x_grids += [x*x, y*y, x*y]
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    #res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)

    nx, ny = arr.shape
    sx, sy = size

    x, y, xx, yy, xy, z = res[0]
    coeffs = np.array([x*nx/sx, y*ny/sy, xx*(nx/sx)**2, yy*(ny/sy)**2,
                       xy*nx/sx*ny/sy, z])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def shift_and_tilt(arr, full_output=False):
    """
    returns an array of same shape and size as arr, but shifted and tilted so
    that mean(arr) = 0 and mean(arr**2) is minimized
    """
    coeffs, location_matrix = tilt_from_height(arr, full_output=True)
    coeffs = np.array(coeffs)
    offsets = arr[...].reshape((-1,))
    if full_output:
        return ((offsets-location_matrix@coeffs).reshape(arr.shape),
                coeffs, res[1])
    else:
        return (offsets-location_matrix@coeffs).reshape(arr.shape)


def shift_and_tilt_approx(arr, full_output=False):
    """
    does the same as shift_and_tilt, but computes an iterative approximation.
    Use in case of large surfaces.
    """
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if nb_dim == 2:
        sx_ = x_grids[0].sum()
        sy_ = x_grids[1].sum()
        s__ = np.prod(arr.shape)
        sxx = (x_grids[0]**2).sum()
        sxy = (x_grids[0]*x_grids[1]).sum()
        syy = (x_grids[1]**2).sum()
        sh_ = arr.sum()
        shx = (arr*x_grids[0]).sum()
        shy = (arr*x_grids[1]).sum()
        location_matrix = np.array(((sxx, sxy, sx_),
                                    (sxy, syy, sy_),
                                    (sx_, sy_, s__)))
        offsets = np.array(((shx,),
                            (shy,),
                            (sh_, )))
        coeffs = scipy.linalg.solve(location_matrix, offsets)
        corrective = coeffs[0]*x_grids[0] + coeffs[1]*x_grids[1] + coeffs[2]
        if full_output:
            return arr - corrective, coeffs
        else:
            return arr - corrective


def shift_and_tilt_from_slope(arr, size=None):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface. returns an array of same shape and size, but shifted and tilted so
    that mean(arr) = 0 and mean(arr') = 0
    """
    nx, ny = arr.shape
    mean_slope = tilt_from_slope(arr, size)
    tilt_correction = sum([x*y for x, y in
                           zip(mean_slope,
                               np.meshgrid(np.arange(nx)-nx//2,
                                           np.arange(ny)-ny//2,
                                           indexing='ij'))])
    arr = arr - tilt_correction
    return arr - arr.mean()
