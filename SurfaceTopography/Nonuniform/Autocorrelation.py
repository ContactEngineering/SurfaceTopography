#
# Copyright 2019-2021 Lars Pastewka
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
Height-difference autocorrelation functions for nonuniform line scans
"""

import numpy as np

import _SurfaceTopography

from ..Exceptions import ReentrantDataError, NoReliableDataError
from ..HeightContainer import NonuniformLineScanInterface


def height_height_autocorrelation(self, distances=None):
    r"""
    Compute the one-dimensional height-height autocorrelation function
    (ACF).

    This function treats the nonuniform line scan as a piece-wise function of
    straight lines between the data points. The ACF is computed exactly for
    this piece-wise linear interpolation of the data.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    r : array_like
        Array containing distances for which to compute the ACF. If no array
        is given, the function will automatically construct an array with
        equally spaced distances. (Default: None)

    Returns
    -------
    distances : array
        Distances. (Units: length)
    A : array
        Autocorrelation function. (Units: length**2)
    """
    if self.is_reentrant:
        raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). The autocorrelation '
                                 'cannot be computed for reentrant topographies.')

    size, = self.physical_sizes
    if distances is None:
        # FIXME!!! We need a better heuristics to decide on the distances
        res, = self.nb_grid_pts
        distances = np.linspace(0, size, res)
    else:
        distances = np.asarray(distances, dtype=float)

    x, h = self.positions_and_heights()
    A = _SurfaceTopography.nonuniform_height_height_autocorrelation(
        np.asarray(x, dtype=float), np.asarray(h, dtype=float), distances)
    return distances, A


def height_difference_autocorrelation(self, reliable=True, algorithm='fft', distances=None, nb_interpolate=5,
                                      short_cutoff=np.mean, resampling_method='bin-average', collocation='log',
                                      nb_points=None, nb_points_per_decade=10):
    r"""
    Compute the one-dimensional height-difference autocorrelation function
    (ACF).

    This function treats the nonuniform line scan as a piece-wise function of
    straight lines between the data points. The ACF is computed exactly for
    this piece-wise linear interpolation of the data.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    algorithm : str
        Algorithm to compute autocorrelation.
        * 'fft': Interpolates the nonuniform line scan on a grid and then uses
        the FFT to compute the autocorrelation. Scales O(N log N)
        * 'brute-force': Brute-force computation using between line segements.
        Scale O(N^2 M) where M is the number of distance point.
        (Default: 'fft')
    distances : array_like
        Array containing distances for which to compute the ACF. If no array
        is given, the function will automatically construct an array with
        equally spaced distances. Can be used only if 'brute-force'
        algorithm is used. (Default: None)
    nb_interpolate : int
        Number of grid points to put between closest points on surface. Only
        used for 'fft' algorithm. (Default: 5)
    short_cutoff : function, optional
        Function that determines how the short cutoff for the returned PSD is
        computed. If set to None, the full PSD of the interpolated data is
        returned. If the user passes a function, that function is called with
        an array that contains the distances between the points of the uniform
        line scan. Pass `np.max`, `np.mean` or `np.min` to use the maximum,
        mean or minimum of that distance as the cutoff.
    resampling_method : str, optional
        Method can be None for no resampling (return on the grid of the
        data) 'bin-average' for simple bin averaging and 'gaussian-process'
        for Gaussian process regression.
        (Default: None)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)

    Returns
    -------
    distances : array
        Distances. (Units: length)
    acf : array
        Autocorrelation function. (Units: length**2)
    """
    if self.is_reentrant:
        raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). The autocorrelation '
                                 'cannot be computed for reentrant topographies.')

    s, = self.physical_sizes
    x, h = self.positions_and_heights()
    if algorithm == 'fft':
        if distances is not None:
            raise ValueError("`distances` can only be used with 'brute-force' algorithm.")
        distances, acf = self.to_uniform(nb_interpolate=nb_interpolate) \
            .autocorrelation_from_profile(reliable=reliable, resampling_method=resampling_method,
                                          collocation=collocation, nb_points=nb_points,
                                          nb_points_per_decade=nb_points_per_decade)
    elif algorithm == 'brute-force':
        if reliable:
            raise ValueError("'Brute-force' algorithm for nonuniform line scans does not support reliability analysis.")
        if resampling_method is not None:
            raise ValueError("'Brute-force' algorithm for nonuniform line scans does not support resampling.")
        if distances is None:
            raise ValueError("You need to specify `distances` for the 'brute-force' algorithm.")
        distances, acf = _SurfaceTopography.nonuniform_autocorrelation(x, h, s, distances)
    else:
        raise ValueError("Unknown algorithm '{}' specified.".format(algorithm))

    if short_cutoff is not None:
        mask = distances > short_cutoff(np.diff(x)) / 2
        if mask.sum() == 0:
            raise NoReliableDataError('Dataset contains no reliable data.')
        distances = distances[mask]
        acf = acf[mask]
    return distances, acf


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('autocorrelation_from_profile', height_difference_autocorrelation)
