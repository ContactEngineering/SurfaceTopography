#
# Copyright 2018, 2020-2021 Lars Pastewka
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
Functions for regression of noisy data and resampling/averaging from one grid
to another. Currently implemented are simple bin averages and Gaussian process
regression.
"""

import logging

import numpy as np

_log = logging.Logger(__name__)


def make_grid(collocation, min_value, max_value, nb_points=None, nb_points_per_decade=10):
    """
    Create collocation points.

    Parameters
    ----------
    collocation : {'log', 'quadratic', 'linear', array_like}
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin edges are explicitly specified, then the other arguments
        to this function are ignored.
    min_value : float
        Minimum value. Note that for log-spaced collocation points, this
        is the first value - the leftmost bin edge with always be zero.
    max_value : float
        Maximum value.
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)

    Returns
    -------
    collocation_points : np.ndarray
        List of collocation points.
    bin_edges : np.ndarray
        List of bins edges; collocations points are located within the
        respective bin. This array contains one more data point than
        the `collocation_points` array.
    """
    if collocation == 'log':
        # Power law -> equally spaced on a log-log plot
        if nb_points is None:
            # The size of the first bin should be equal to min_radius,
            # which is the (minimal) size of a grid point
            # i.e. min_radius = np.exp(np.log(min_radius) + dl) - min_radius
            # => dl = np.log(2)
            nb_points = int(np.ceil((np.log10(max_value) - np.log10(min_value) + 1) * nb_points_per_decade))
        bin_edges = np.linspace(np.log10(min_value), np.log10(max_value), nb_points)
        collocation_points = 10 ** ((bin_edges[1:] + bin_edges[:-1]) / 2)
        bin_edges = 10 ** bin_edges
    elif collocation == 'quadratic':
        # Quadratic -> similar statistics for each data point on a 2D radial grid
        if nb_points is None:
            raise ValueError("Please specify number of bins for 'quadratic' bins.")
        bin_edges = np.sqrt(np.linspace(min_value ** 2, max_value ** 2, nb_points + 1))
        collocation_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    elif collocation == 'linear':
        # Linear
        if nb_points is None:
            raise ValueError("Please specify number of bins for 'linear' bins.")
        bin_edges = np.linspace(min_value, max_value, nb_points + 1)
        collocation_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    else:
        bin_edges = collocation
        collocation_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    return collocation_points, bin_edges


def gaussian_kernel(x1, x2, length_scale=1, signal_variance=1):
    return signal_variance * np.exp(-(x1 - x2) ** 2 / (2 * length_scale ** 2))


def gaussian_log_kernel(x1, x2, length_scale=1, signal_variance=1):
    return signal_variance * np.exp(-(np.log10(x1) - np.log10(x2)) ** 2 / (2 * length_scale ** 2))


def suggest_kernel_for_grid(collocation, nb_collocation_points, min_value, max_value):
    """
    Suggest a kernel function for Gaussian process regression with test
    locations on a specific grid. The length scale (smoothening scale)
    of the kernel is set to the characteristic spacing between the sampling
    output points.

    Parameters
    ----------
    collocation : {'log', 'quadratic', 'linear', array_like}
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin edges are explicitly specified, then the other arguments
        to this function are ignored.
    nb_collocation_points : int
        Number of collocation points.
    min_value : float
        Minimum value. Note that for log-spaced collocation points, this
        is the first value - the leftmost bin edge with always be zero.
    max_value : float
        Maximum value.

    Returns
    -------
    kernel : func
        Kernel function.
    """
    if collocation == 'log':
        length_scale = (np.log10(max_value) - np.log10(min_value)) / nb_collocation_points
        return lambda x1, x2: gaussian_log_kernel(x1, x2, length_scale=length_scale)
    else:
        length_scale = (max_value - min_value) / nb_collocation_points
        return lambda x1, x2: gaussian_kernel(x1, x2, length_scale=length_scale)


def bin_average(collocation_points, bin_edges, x, values):
    """
    Average values over bins. Returns NaN for bins without data.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points.
    bin_edges : array_like
        Edges of bins.
    x : array_like
        Collocation points for `values` array.
    values : array_like
        Function values/variates.

    Returns
    -------
    resampled_collocation_points : np.ndarray
        Collocation points, resampled as average over input `x` if data is present.
    resampled_values : np.ndarray
        Resampled/averaged values.
    resampled_variance : np.ndarray
        Variance of resampled data.
    """
    # Find bin index
    bin_index = np.searchsorted(bin_edges, x)

    # Compute averages within bins
    number_of_data_points = np.bincount(bin_index, minlength=len(bin_edges) + 1)
    number_of_data_points1 = np.where(number_of_data_points == 0, np.ones_like(number_of_data_points),
                                      number_of_data_points)
    resampled_values = np.bincount(bin_index, weights=values, minlength=len(bin_edges) + 1) / number_of_data_points1
    resampled_variance = \
        np.bincount(bin_index, weights=values ** 2,
                    minlength=len(bin_edges) + 1) / number_of_data_points1 - resampled_values ** 2

    # Resample collocation points as average of the distances in each bin
    resampled_collocation_points = np.bincount(bin_index, weights=x,
                                               minlength=len(bin_edges) + 1) / number_of_data_points1

    # We discard the final element as it contains data points outside our binned region
    resampled_collocation_points = resampled_collocation_points[1:-1]
    resampled_values = resampled_values[1:-1]
    resampled_variance = resampled_variance[1:-1]

    # Mark elements with no data
    mask = number_of_data_points[1:-1] == 0
    resampled_collocation_points[mask] = collocation_points[mask]
    resampled_values[mask] = np.nan
    resampled_variance[mask] = np.nan
    return resampled_collocation_points, resampled_values, resampled_variance


def gaussian_process_regression(output_x, x, values, kernel=gaussian_kernel, noise_variance=1e-6):
    """
    Gaussian process regression for resampling a simple function.

    Parameters
    ----------
    output_x : array_like
        Collocation points for resampled values.
    x : array_like
        Collocation points for `values` array.
    values : array_like
        Function values/variates.
    kernel : func, optional
        Kernel function/covariance model.
        (Default: gaussian_kernel)
    noise_variance : float, optional
        Noise variance.
        (Default: 1e-6)

    Returns
    -------
    resampled_values : np.ndarray
        Resampled/averaged values.
    """
    # Covariance between observations
    obs_cov = kernel(x.reshape(-1, 1), x.reshape(1, -1))

    # Add noise to observation covariance matrix
    obs_cov += noise_variance * np.identity(len(x))

    # Compute kernel coefficients
    coeff = np.linalg.solve(obs_cov, values)

    # Covariance between test outputs
    test_cov = kernel(output_x.reshape(-1, 1), output_x.reshape(1, -1))

    # Covariance between observation and test outputs
    obs_test_cov = kernel(x.reshape(-1, 1), output_x.reshape(1, -1))

    # Compute predictive mean
    pred_mean = coeff.dot(obs_test_cov)

    # Compute predictive covariance
    pred_cov = test_cov - obs_test_cov.T.dot(np.linalg.solve(obs_cov, obs_test_cov))

    # Return mean and variance
    return pred_mean, pred_cov.diagonal()


def resample(x, values, collocation='log', nb_points=None, min_value=None, max_value=None, nb_points_per_decade=10,
             method='bin-average'):
    """
    Resample noisy function data set onto a specific grid.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    values : array_like
        Function values.
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
    min_value : float, optional
        Minimum value, is automatically determined from the range of the data
        if not provided. (Default: None)
    max_value : float, optional
        Maximum value, is automatically determined from the range of the data
        if not provided. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)
    method : str, optional
        Method can be 'bin-average' for simple bin averaging and
        'gaussian-process' for Gaussian process regression.
        (Default: 'bin-average')

    Returns
    -------
    collocation_points : np.ndarray
        Points where the data has been collected.
    bin_edges : np.ndarray
        Bin edges.
    resampled_values : np.ndarray
        Resampled values.
    resampled_variance : np.ndarray
        Variance of resampled data.
    """
    # pylint: disable=invalid-name
    if max_value is None:
        max_value = np.max(x)
    if min_value is None:
        min_value = np.min(x)

    collocation_points, bin_edges = make_grid(collocation, min_value, max_value, nb_points=nb_points,
                                              nb_points_per_decade=nb_points_per_decade)

    if method == 'bin-average':
        collocation_points, resampled_values, resampled_variance = bin_average(collocation_points, bin_edges, x, values)
    elif method == 'gaussian-process':
        if collocation == 'log':
            # For log-spaced sampling points we fit in log-space... because
            # this is then most likely a power-law. Note that this is pure
            # function fitting - if we are collecting noisy data to compute
            # the ACF or PSD we should actually be averaging in linear space.
            # (This is because the arithmetic mean is the max likelihood
            # estimator for the underlying distribution of the data. The
            # difference can be clearly seen in numerical experiments.)
            resampled_values, resampled_variance = \
                gaussian_process_regression(np.log(collocation_points), np.log(x), np.log(values))
            resampled_values = np.exp(resampled_values)
        else:
            _log.warning('Gaussian process regression of log-spaced data should only be used for fitting functions, '
                         'not for inference from noisy data.')
            resampled_values, resampled_variance = \
                gaussian_process_regression(collocation_points, x, values)
    # FIXME: The variance is bogus when log-sampling and we should really use a different kernel function in that case.
    #        resampled_values, resampled_variance = \
    #            gaussian_process_regression(collocation_points, x, values,
    #                                        kernel=suggest_kernel_for_grid(collocation, len(collocation_points),
    #                                                                       min_value, max_value))
    else:
        raise ValueError(f"Unknown resampling method '{method}'.")

    return collocation_points, bin_edges, resampled_values, resampled_variance


def resample_radial(data, physical_sizes=None, collocation='log', nb_points=None, min_radius=0, max_radius=None,
                    nb_points_per_decade=5, full=True, method='bin-average'):
    """
    Compute radial average of quantities reported on a 2D grid and collect
    results of collocation points.

    Parameters
    ----------
    data : array_like
        2D-array of values to be averaged.
    physical_sizes : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of
        grid points.)
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
    min_radius : float, optional
        Minimum radius. (Default: 0)
    max_radius : float, optional
        Maximum radius, is automatically determined from the range of the data
        if not provided. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)
    full : bool, optional
        Number of quadrants contained in data.
        True: Full radial average from 0 to 2*pi.
        False: Only the one quarter of the full circle is present. Radial
        average from 0 to pi/2.
        (Default: True)
    method : str, optional
        Method can be 'bin-average' for simple bin averaging and
        'gaussian-process' for Gaussian process regression.
        (Default: 'bin-average')

    Returns
    -------
    collocation_points : np.ndarray
        Points where the data has been collected.
    bin_edges : np.ndarray
        Bin edges.
    resampled_values : np.ndarray
        Resampled values.
    resampled_variance : np.ndarray
        Variance of resampled data.
    """
    # pylint: disable=invalid-name
    nx, ny = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    if full:
        x = np.where(x > nx // 2, nx - x, x)
        y = np.where(y > ny // 2, ny - y, y)

    if physical_sizes is None:
        min_radius = max(min_radius, 1.0)
    else:
        sx, sy = physical_sizes
        x = sx / nx * x
        y = sy / ny * y
        min_radius = max(min(sx / nx, sy / ny), min_radius)
    radius = np.sqrt((x ** 2).reshape(-1, 1) + (y ** 2).reshape(1, -1))
    if max_radius is None:
        max_radius = np.max(radius)

    return resample(np.ravel(radius), np.ravel(data), collocation=collocation, nb_points=nb_points,
                    min_value=min_radius, max_value=max_radius, nb_points_per_decade=nb_points_per_decade,
                    method=method)
