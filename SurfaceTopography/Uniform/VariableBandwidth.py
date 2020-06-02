#
# Copyright 2018, 2020 Lars Pastewka
#           2019 Antoine Sanner
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
Variable bandwidth analysis for uniform topographies
"""

import numpy as np

from ..HeightContainer import UniformTopographyInterface


def checkerboard_detrend(topography, subdivisions):
    """
    Perform tilt correction (and substract mean value) in each individual
    rectangle of a checkerboard decomposition of the surface. This is
    identical to subdividing the surface into individual, nonoverlapping
    rectangles and performing individual tilt corrections on them.

    The main application of this function is to carry out a variable
    bandwidth analysis of the surface.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    subdivisions : tuple
        Number of subdivision per dimension, i.e. physical_sizes of the
        checkerboard.

    Returns
    -------
    arr : array
        Array with height information, tilt-corrected within each
        checkerboard.
    """
    arr = topography.heights().copy()
    nb_dim = topography.dim

    shape = arr.shape

    # compute unique consecutive index for each subdivided region
    region_coord = [np.arange(shape[i]) * subdivisions[i] // shape[i] for i in
                    range(nb_dim)]
    if nb_dim > 1:
        region_coord = np.meshgrid(*region_coord, indexing='ij')
    region_index = region_coord[0]
    for i in range(1, nb_dim):
        region_index = subdivisions[i] * region_index + region_coord[i]

    # compute x- and y-coordinate
    if nb_dim == 1:
        x = np.arange(shape[0])
    elif nb_dim == 2:
        x, y = np.meshgrid(*(np.arange(n) for n in shape), indexing='ij')
    else:
        raise ValueError(
            'Cannot handle {}-dimensional topographies.'.format(nb_dim))

    region_index.shape = (-1,)
    x.shape = (-1,)
    arr.shape = (-1,)
    sum_1 = np.bincount(region_index)
    sum_x = np.bincount(region_index, x)
    sum_xx = np.bincount(region_index, x * x)
    sum_h = np.bincount(region_index, arr)
    sum_xh = np.bincount(region_index, x * arr)
    if nb_dim == 2:
        y.shape = (-1,)
        sum_y = np.bincount(region_index, y)
        sum_yy = np.bincount(region_index, y * y)
        sum_xy = np.bincount(region_index, x * y)
        sum_yh = np.bincount(region_index, y * arr)

    if nb_dim == 1:
        # Calculated detrended plane. Detrended plane is given by h0 + mx*x.
        A = np.array([[sum_1, sum_x],
                      [sum_x, sum_xx]])
        b = np.array([sum_h, sum_xh])
        h0, mx = np.linalg.solve(A.T, b.T).T

        arr -= h0[region_index] + mx[region_index] * x
    else:
        # Calculated detrended plane. Detrended plane is given by
        # h0 + mx*x + my*y.
        A = np.array([[sum_1, sum_x, sum_y],
                      [sum_x, sum_xx, sum_xy],
                      [sum_y, sum_xy, sum_yy]])
        b = np.array([sum_h, sum_xh, sum_yh])
        h0, mx, my = np.linalg.solve(A.T, b.T).T

        arr -= h0[region_index] + mx[region_index] * x + my[region_index] * y

    arr.shape = shape

    return arr


def variable_bandwidth(topography, nb_grid_pts_cutoff=4):
    """
    Perform a variable bandwidth analysis by computing the mean
    root-mean-square height within increasingly finer subdivisions of the
    surface topography.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    nb_grid_pts_cutoff : int
        Minimum nb_grid_pts to allow for subdivision. The analysis will
        automatically analyze subdivision down to this nb_grid_pts.

    Returns
    -------
    magnifications : array
        Array containing the magnifications.
    bandwidths : array
        Array containing the bandwidths, here the physical sizes of the
        subdivided topography. For 2D topography maps, this is the mean of the
        two physical sizes of the subdivided section of the topography.
    rms_heights : array
        Array containing the rms height corresponding to the respective
        magnification.
    """
    magnification = 1
    physical_sizes = np.array(topography.physical_sizes)
    min_size = np.min(physical_sizes)
    subdivisions = np.round(topography.physical_sizes / min_size).astype(int)
    nb_grid_pts = np.array(topography.nb_grid_pts, dtype=int)
    magnifications = []
    bandwidths = []
    rms_heights = []
    while ((nb_grid_pts // subdivisions).min() >= nb_grid_pts_cutoff):
        magnifications += [magnification]
        bandwidths += [np.mean(physical_sizes / subdivisions)]
        rms_heights += [np.std(topography.checkerboard_detrend(subdivisions))]
        magnification *= 2
        subdivisions *= 2
    return np.array(magnifications), np.array(bandwidths), np.array(
        rms_heights)


# Register analysis functions from this module
UniformTopographyInterface.register_function('checkerboard_detrend',
                                             checkerboard_detrend)
UniformTopographyInterface.register_function('variable_bandwidth',
                                             variable_bandwidth)
