#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   VariableBandwidth.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   06 Sep 2018

@brief  Variable bandwidth analysis

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

from ..common import _get_size


def checkerboard_tilt_correction(arr, sd, size=None):
    """
    Perform tilt correction (and substract mean value) in each individual
    rectangle of a checkerboard decomposition of the surface. This is
    identical to subdividing the surface into individual, nonoverlapping
    rectangles and performing individual tilt corrections on them.

    The main application of this function is to carry out a variable
    bandwidth analysis of the surface.

    Parameters
    ----------
    arr : array, :obj:`Topography`
        Height information.
    sd : tuple
        Number of subdivision per dimension, i.e. size of the checkerboard.
    size : tuple, optional
        Size of the topography specified in `arr`. If `arr` is a
        :obj:`Topography` then size will be obtained automatically.
    Returns
    -------
    arr : array
        Array with height information, tilt-corrected within each
        checkerboard.
    """
    size = _get_size(arr, size)
    arr = arr[...]
    nb_dim = len(arr.shape)

    # compute unique consecutive index for each subdivided region
    region_coord = [np.arange(arr.shape[i])*sd[i]//arr.shape[i]
                    for i in range(nb_dim)]
    if nb_dim > 1:
        region_coord = np.meshgrid(*region_coord, indexing='ij')
    region_index = region_coord[0]
    for i in range(1, nb_dim):
        region_index = sd[i]*region_index + region_coord[i]

    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    full_columns = [x.reshape((-1, 1)) for x in x_grids]

    for i in range(np.max(region_index)+1):
        # tilt correction of each of these regions
        mask = region_index == i
        flat_mask = np.ravel(mask)
        columns = [c[flat_mask] for c in full_columns]
        columns.append(np.ones_like(columns[-1]))
        # linear regression model
        location_matrix = np.array(np.hstack(columns))
        offsets = np.ravel(arr)[flat_mask]
        res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
        # correct tilt
        arr[mask] -= location_matrix.dot(res[0])

    return arr
