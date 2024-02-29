#
# Copyright 2018-2021, 2024 Lars Pastewka
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

from ..Exceptions import NoReliableDataError, UndefinedDataError
from ..HeightContainer import UniformTopographyInterface
from ..Support import build_tuple


def checkerboard_detrend_profile(self, subdivisions, order=1, return_plane=False):
    """
    Perform tilt correction (and substract mean value) in each individual
    line section of a checkerboard decomposition of a profile. For topography
    maps, each horizontal slice is interpreted as a profile and this
    decomposition is carried out for each slice. The tilt correction minimizes
    the RMS height within each checkerboard.

    The main application of this function is to carry out a variable
    bandwidth analysis of the surface.

    Parameters
    ----------
    self : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    subdivisions : int
        Number of subdivision, i.e. physical_sizes of the
        checkerboard.
    order : int, optional
        Maximum order of the polynomial used for detrending. (Default: 1)
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
    # compute unique consecutive index for each subdivided region
    region_index = np.arange(self.nb_grid_pts[0]) * subdivisions // self.nb_grid_pts[0]

    if self.dim == 1:
        x, h = self.positions_and_heights()
    elif self.dim == 2:
        nx, ny = self.nb_grid_pts
        x, y, h = self.positions_and_heights()
        region_index = np.array([[region_index + i * subdivisions] for i in range(ny)]).T
    else:
        raise ValueError(f'Cannot perform checkerboard detrend on topographies of dimension {self.dim}')

    shape = h.shape
    h = h.reshape(-1)
    x = x.reshape(-1)
    region_index = region_index.reshape(-1)

    b = np.array([np.bincount(region_index, h * (x ** i)) for i in range(order + 1)])
    C = np.array([[np.bincount(region_index, x ** (k + i)) for i in range(order + 1)] for k in range(order + 1)])
    a = np.linalg.solve(C.T, b.T).T

    detrended_h = h - np.sum([a[i, region_index] * x ** i for i in range(order + 1)], axis=0)
    detrended_h.shape = shape

    if return_plane:
        return detrended_h, a
    else:
        return detrended_h


def checkerboard_detrend_area(self, subdivisions, order=1, return_plane=False):
    """
    Perform tilt correction (and substract mean value) in each individual
    rectangle of a checkerboard decomposition of the surface. This is
    identical to subdividing the surface into individual, nonoverlapping
    rectangles and performing individual tilt corrections on them.
    The tilt correction minimizes the RMS height within each checkerboard.

    The main application of this function is to carry out a variable
    bandwidth analysis of the surface.

    Parameters
    ----------
    self : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    subdivisions : tuple
        Number of subdivision per dimension, i.e. physical_sizes of the
        checkerboard.
    order : int, optional
        Maximum order of the polynomial used for detrending. (Default: 1)
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
    if self.dim != 2:
        raise ValueError('Areal checkerboard tilt correction can only performed on topography maps, not on profiles.')

    # compute unique consecutive index for each subdivided region
    region_coord = [np.arange(n) * s // n for n, s in zip(self.nb_grid_pts, subdivisions)]
    region_coord = np.meshgrid(*region_coord, indexing='ij')
    region_index = region_coord[0]
    for i in range(1, self.dim):
        region_index = subdivisions[i] * region_index + region_coord[i]

    # Number of polynomial coefficents
    nb_coeff = (order + 1) * (order + 2) // 2

    # Build list of possible exponents
    ij = []
    i = j = 0
    k = nb_coeff
    while k > 0:
        ij += [(i, j)]
        i += 1
        if i + j > order:
            i = 0
            j += 1
        k -= 1

    assert len(ij) == nb_coeff

    x, y, h = self.positions_and_heights()
    shape = h.shape
    h = h.reshape(-1)
    x = x.reshape(-1)
    y = y.reshape(-1)
    region_index = region_index.reshape(-1)

    b = np.array([np.bincount(region_index, h * (x ** i) * (y ** j)) for i, j in ij])
    C = np.array([[np.bincount(region_index, x ** (k + i) * y ** (l + j)) for i, j in ij] for k, l in ij])
    a = np.linalg.solve(C.T, b.T).T

    detrended_h = h - np.sum([a[k, region_index] * (x ** i) * (y ** j) for k, (i, j) in enumerate(ij)], axis=0)
    detrended_h.shape = shape

    if return_plane:
        return detrended_h, a
    else:
        return detrended_h


def variable_bandwidth_from_profile(self, quantities='bh', reliable=True, resampling_method=None, nb_grid_pts_cutoff=4,
                                    func=lambda h: np.sqrt(np.mean(h * h))):
    """
    Perform a variable bandwidth analysis by computing the mean
    root-mean-square height within increasingly finer subdivisions of the
    profiles (line scans).

    Parameters
    ----------
    self : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map or line scan.
    quantities : str, optional
        Specification of return tuple, where each string character
        stand for specific quantity. Possible quantities are
            - 'm': Magnification (Unit: dimensionless)
            - 'b': Bandwidth (Unit: length)
            - 'h': Statistical property (Unit: default length, see func)
            - 's': RMS detrending slope (Unit: dimensionless)
        For example, 'mbh' return a tuple with the three entries
        magnification, bandwidth, rms height.
        (Default: 'bh')
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    resampling_method : str, optional
        Can only be None for the variable bandwidth analysis.
        (Default: None)
    nb_grid_pts_cutoff : int, optional
        Minimum nb_grid_pts to allow for subdivision. The analysis will
        automatically analyze subdivision down to this nb_grid_pts.
        (Default: 4)
    func : callable, optional
        Function that computes a statistical property from an array of heights.
        (Default is the RMS height: np.sqrt(np.mean(h * h)))

    Returns
    -------
    magnifications : np.ndarray
        Array containing the magnifications.
    bandwidths : np.ndarray
        Array containing the bandwidths, here the physical sizes of the
        subdivided topography. For 2D topography maps, this is the mean of the
        two physical sizes of the subdivided section of the topography.
    statistical_properties : np.ndarray
        Array containing the rms height corresponding to the respective
        magnification.
    rms_slopes : np.ndarray
        Array containing the rms values of the detrending slopes corresponding
        to the respective magnification.
    """
    if resampling_method is not None:
        raise ValueError('`variable_bandwidth_from_profile` does not support resampling.')

    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Variable bandwidth '
                                 'analysis cannot be computed for topographies with missing data points.')

    magnification = 1
    subdivisions = 1
    sx = self.physical_sizes[0]
    nx = int(self.nb_grid_pts[0])
    magnifications = []
    bandwidths = []
    statistical_properties = []
    rms_slopes = []
    while nx // subdivisions >= nb_grid_pts_cutoff:
        magnifications += [magnification]
        bandwidths += [sx / subdivisions]
        h, coeffs = self.checkerboard_detrend_profile(subdivisions, return_plane=True)
        statistical_properties += [func(h)]
        rms_slopes += [np.sqrt(np.mean(coeffs[1] * coeffs[1]))]
        magnification *= 2
        subdivisions *= 2

    magnifications = np.array(magnifications)
    bandwidths = np.array(bandwidths)
    statistical_properties = np.array(statistical_properties)
    rms_slopes = np.array(rms_slopes)

    if reliable:
        short_cutoff = self.short_reliability_cutoff()
        if short_cutoff:
            m = bandwidths > short_cutoff
            if m.sum() == 0:
                raise NoReliableDataError('Dataset contains no reliable data.')
            magnifications = magnifications[m]
            bandwidths = bandwidths[m]
            statistical_properties = statistical_properties[m]
            rms_slopes = rms_slopes[m]

    return build_tuple(quantities, m=magnifications, b=bandwidths, h=statistical_properties, g=rms_slopes)


def variable_bandwidth_from_area(self, quantities='bh', reliable=True, resampling_method=None, nb_grid_pts_cutoff=4,
                                 func=lambda h: np.sqrt(np.mean(h * h))):
    """
    Perform a variable bandwidth analysis by computing the mean
    root-mean-square height within increasingly finer subdivisions of the
    surface topography.

    Parameters
    ----------
    self : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map.
    quantities : str, optional
        Specification of return tuple, where each string character
        stand for specific quantity. Possible quantities are
            - 'm': Magnification (Unit: dimensionless)
            - 'b': Bandwidth (Unit: length)
            - 'h': Statistical property (Unit: default length, see func)
            - 'g': RMS detrending gradient (Unit: dimensionless)
        For example, 'mbh' return a tuple with the three entries
        magnification, bandwidth, rms height.
        (Default: 'bh')
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    resampling_method : str, optional
        Can only be None for the variable bandwidth analysis.
        (Default: None)
    nb_grid_pts_cutoff : int, optional
        Minimum nb_grid_pts to allow for subdivision. The analysis will
        automatically analyze subdivision down to this nb_grid_pts.
        (Default: 4)
    func : callable, optional
        Function that computes a statistical property from an array of heights.
        (Default is the RMS height: np.sqrt(np.mean(h * h)))

    Returns
    -------
    magnifications : array
        Array containing the magnifications.
    bandwidths : array
        Array containing the bandwidths, here the physical sizes of the
        subdivided topography. For 2D topography maps, this is the mean of the
        two physical sizes of the subdivided section of the topography.
    statistical_properties : array
        Array containing the rms height corresponding to the respective
        magnification.
    rms_gradients : np.ndarray
        Array containing the rms values of the detrending gradients
        corresponding to the respective magnification.
    """
    if resampling_method is not None:
        raise ValueError('`variable_bandwidth_from_profile` does not support resampling.')

    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Variable bandwidth '
                                 'analysis cannot be computed for topographies with missing data points.')

    magnification = 1
    physical_sizes = np.array(self.physical_sizes)
    min_size = np.min(physical_sizes)
    subdivisions = np.round(self.physical_sizes / min_size).astype(int)
    nb_grid_pts = np.array(self.nb_grid_pts, dtype=int)
    magnifications = []
    bandwidths = []
    statistical_properties = []
    rms_gradients = []
    while ((nb_grid_pts // subdivisions).min() >= nb_grid_pts_cutoff):
        magnifications += [magnification]
        bandwidths += [np.mean(physical_sizes / subdivisions)]
        h, coeffs = self.checkerboard_detrend_area(subdivisions, return_plane=True)
        statistical_properties += [func(h)]
        rms_gradients += [np.sqrt(np.mean(coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2]))]
        magnification *= 2
        subdivisions *= 2

    magnifications = np.array(magnifications)
    bandwidths = np.array(bandwidths)
    statistical_properties = np.array(statistical_properties)
    rms_gradients = np.array(rms_gradients)

    if reliable:
        short_cutoff = self.short_reliability_cutoff()
        if short_cutoff:
            m = bandwidths > short_cutoff
            if m.sum() == 0:
                raise NoReliableDataError('Dataset contains no reliable data.')
            magnifications = magnifications[m]
            bandwidths = bandwidths[m]
            statistical_properties = statistical_properties[m]
            rms_gradients = rms_gradients[m]

    return build_tuple(quantities, m=magnifications, b=bandwidths, h=statistical_properties, g=rms_gradients)


# Register analysis functions from this module
UniformTopographyInterface.register_function('checkerboard_detrend_profile', checkerboard_detrend_profile)
UniformTopographyInterface.register_function('checkerboard_detrend_area', checkerboard_detrend_area)
UniformTopographyInterface.register_function('variable_bandwidth_from_profile', variable_bandwidth_from_profile)
UniformTopographyInterface.register_function('variable_bandwidth_from_area', variable_bandwidth_from_area)
