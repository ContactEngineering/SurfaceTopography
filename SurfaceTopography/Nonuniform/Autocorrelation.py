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
    Compute the unnormalized height-height correlation

    .. math::

        \int dx\, h(x)\, h(x + r)

    of a nonuniform line scan, where the integral runs over the overlap of
    the scan with its shifted copy. The line scan is treated as a piecewise
    linear function and the integral is computed exactly. Scales as
    :math:`O(N^2 M)`, where :math:`M` is the number of distances.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    distances : array_like, optional
        Array containing distances for which to compute the correlation. If
        no array is given, :math:`N` equally spaced distances between
        :math:`0` and :math:`L` are used. (Default: None)

    Returns
    -------
    distances : np.ndarray
        Distances. (Units: length)
    A : np.ndarray
        Height-height correlation integral. (Units: length³)
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
    (ACF), :math:`A(r) = \frac{1}{2}\langle[h(x+r) - h(x)]^2\rangle`, of a
    nonuniform line scan. The line scan is treated as a piecewise linear
    function, i.e. the data points are connected by straight lines.

    The 'fft' algorithm interpolates this function linearly onto a uniform
    grid whose spacing is the smallest distance between data points divided
    by `nb_interpolate` and computes the ACF of the resulting (nonperiodic)
    uniform line scan, see
    :func:`SurfaceTopography.Uniform.Autocorrelation.autocorrelation_from_profile`.

    The 'brute-force' algorithm computes

    .. math::

        A(r) = \frac{1}{2(L - r)} \int_{x_0}^{x_{N-1} - r} dx\,
        \left[h(x+r) - h(x)\right]^2

    exactly: for each pair of segments, the integrand is a quadratic
    polynomial on the overlap of the first segment with the shifted second
    segment and is integrated analytically.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    reliable : bool, optional
        Only return data deemed reliable. Only supported by the 'fft'
        algorithm. (Default: True)
    algorithm : {'fft', 'brute-force'}, optional
        Algorithm to compute the autocorrelation.

        * 'fft': Interpolates the nonuniform line scan on a uniform grid
          and then uses the FFT to compute the autocorrelation. Scales as
          :math:`O(N_\text{int}\log N_\text{int})`, where :math:`N_\text{int}`
          is the number of interpolation points.
        * 'brute-force': Exact computation from pairs of line segments.
          Scales as :math:`O(N^2 M)`, where :math:`M` is the number of
          distances.

        (Default: 'fft')
    distances : array_like, optional
        Distances at which to compute the ACF; all must be smaller than
        :math:`L`. Required for, and only accepted by, the 'brute-force'
        algorithm. (Default: None)
    nb_interpolate : int, optional
        Number of grid points to put between the closest pair of data
        points. Only used for the 'fft' algorithm. (Default: 5)
    short_cutoff : callable, optional
        Function that determines the short-distance cutoff of the returned
        ACF. It is called with the array of distances :math:`\Delta x_i`
        between neighboring data points; distances
        :math:`r > \text{short\_cutoff}(\Delta x_i)/2` are returned. Pass
        `np.max`, `np.mean` or `np.min` to use the maximum, mean or minimum
        of the distances. If set to None, the full ACF is returned. This
        cutoff is applied in addition to the reliability cutoff.
        (Default: np.mean)
    resampling_method : {None, 'bin-average', 'gaussian-process'}, optional
        None returns the ACF on the distances of the interpolation grid (or
        the requested distances), 'bin-average' averages it within bins,
        'gaussian-process' resamples it by Gaussian process regression.
        The 'brute-force' algorithm requires None. (Default: 'bin-average')
    collocation : {'log', 'quadratic', 'linear'}, optional
        Resampling grid. 'log' yields collocation points equally spaced on
        a log scale, 'quadratic' yields bins equally spaced in :math:`r^2`
        and 'linear' yields linear bins. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. For 'log' collocation, the number of
        bins is determined from `nb_points_per_decade` if set to None;
        'linear' and 'quadratic' collocation require this argument.
        (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)

    Returns
    -------
    distances : np.ndarray or np.ma.MaskedArray
        Distances. (Units: length)
    acf : np.ndarray or np.ma.MaskedArray
        Height-difference autocorrelation function. (Units: length²)
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
