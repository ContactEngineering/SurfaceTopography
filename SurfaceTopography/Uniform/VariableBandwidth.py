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


def checkerboard_detrend_line_scan(topography, region_index, order, return_plane):
    x, h = topography.positions_and_heights()
    b = np.array([np.bincount(region_index, h * (x ** i)) for i in range(order + 1)])
    C = np.array([[np.bincount(region_index, x ** (k + i)) for i in range(order + 1)] for k in range(order + 1)])
    a = np.linalg.solve(C.T, b.T).T

    detrended_h = h - np.sum([a[i, region_index] * x ** i for i in range(order + 1)], axis=0)

    if return_plane:
        return detrended_h, a
    else:
        return detrended_h


def checkerboard_detrend_topography(topography, region_index, order, return_plane):
    x, h = topography.positions_and_heights()
    b = [(h * (x ** i)).sum() for i in range(order)]
    C = [[(x ** (k + i)).sum() for i in range(order)] for k in range(order)]
    a = np.linalg.solve(C, b)

    return h - np.sum([a[i] * x ** i for i in range(order)], axis=0)


def checkerboard_detrend(topography, subdivisions, order=1, return_plane=False):
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
    order : int, optional
        Order of the polynomial used for detrending. 0 = subtract mean,
        1 = remove mean and tilt, 2 = remove curvature. (Default: 1)
    return_plane : bool, optional
        Return parameters of the detrending plane. (Default: False)

    Returns
    -------
    arr : np.ndarray
        Array with height information, tilt-corrected within each
        checkerboard.

    if return_plane == True:
    parameters : np.ndarray
        Array of leading order `subdivisions` containing the fit
        parameters.
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

    if nb_dim == 1:
        return checkerboard_detrend_line_scan(topography, region_index, order, return_plane)
    else:
        return checkerboard_detrend_topography(topography, region_index, order, return_plane)


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
UniformTopographyInterface.register_function('checkerboard_detrend', checkerboard_detrend)
UniformTopographyInterface.register_function('variable_bandwidth', variable_bandwidth)
