#
# Copyright 2021 Lars Pastewka
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
Resampling/averaging of noisy data from one grid to another by bin
averages.
"""

import numpy as np

from ..Exceptions import NoReliableDataError


def make_grid(collocation, min_value, max_value, nb_points=None, nb_points_per_decade=10, dectol=0.01):
    """
    Create collocation points.

    Parameters
    ----------
    collocation : {'log', 'quadratic', 'linear', array_like}
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bin edges
        equally spaced in the square of the value (similar number of data
        points per bin on a 2D radial grid) and 'linear' yields linear bins.
        An array specifies the bin edges explicitly; `min_value`,
        `max_value`, `nb_points` and `nb_points_per_decade` are then ignored.
    min_value : float
        Minimum value. For log-spaced collocation points, bin edges extend
        `dectol` decades beyond `min_value` and `max_value`.
    max_value : float
        Maximum value.
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. Caution: for 'log' collocation, this is the number of bin
        *edges*, i.e. `nb_points` - 1 collocation points are returned, while
        for 'linear' and 'quadratic' collocation it is the number of
        collocation points (`nb_points` + 1 edges). (Default: None)
    nb_points_per_decade : int, optional
        Number of bin edges per decade for log-spaced collocation points,
        used if `nb_points` is None. (Default: 10)
    dectol : float, optional
        Tolerance for left and right bin edge in terms of decades added to
        the edges. (Default: 0.01)

    Returns
    -------
    collocation_points : np.ndarray
        List of collocation points.
    bin_edges : np.ndarray
        List of bins edges; collocations points are located within the
        respective bin. This array contains one more data point than
        the `collocation_points` array.
    """
    if not isinstance(collocation, str):
        # Explicit bin edges
        bin_edges = np.asarray(collocation, dtype=float)
        if bin_edges.ndim != 1 or len(bin_edges) < 2 or np.any(np.diff(bin_edges) <= 0):
            raise ValueError('Bin edges must be a one-dimensional, strictly increasing array with at least two '
                             'entries.')
        collocation_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    elif collocation == 'log':
        # Power law -> equally spaced on a log-log plot
        if nb_points is None:
            # The size of the first bin should be equal to min_radius,
            # which is the (minimal) size of a grid point
            # i.e. min_radius = np.exp(np.log(min_radius) + dl) - min_radius
            # => dl = np.log(2)
            nb_points = int(np.ceil((np.log10(max_value) - np.log10(min_value) + 1) * nb_points_per_decade))
        bin_edges = np.linspace(np.log10(min_value) - dectol, np.log10(max_value) + dectol, nb_points)
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
        raise ValueError(f"Unknown collocation '{collocation}'.")
    return collocation_points, bin_edges


def bin_average(collocation_points, bin_edges, x, values):
    """
    Average values over bins. Returns masked arrays with bins without data
    masked. Data points outside the outermost bin edges are discarded.

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
    resampled_collocation_points : np.ma.MaskedArray
        Collocation points, resampled as average over input `x` if data is
        present. Bins without data are masked.
    resampled_values : np.ma.MaskedArray
        Resampled/averaged values. Bins without data are masked.
    resampled_variance : np.ma.MaskedArray
        Variance of the data within each bin (not the variance of the
        mean). Bins without data are masked.
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

    # Mask elements with no data
    mask = number_of_data_points[1:-1] == 0
    resampled_collocation_points[mask] = collocation_points[mask]

    # Convert to masked arrays
    resampled_collocation_points = np.ma.array(resampled_collocation_points, mask=mask)
    resampled_values = np.ma.array(resampled_values, mask=mask)
    resampled_variance = np.ma.array(resampled_variance, mask=mask)

    return resampled_collocation_points, resampled_values, resampled_variance


def resample(x, values, collocation='log', nb_points=None, min_value=None, max_value=None, nb_points_per_decade=10,
             method='bin-average'):
    r"""
    Resample noisy function data set onto a specific grid.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    values : array_like
        Function values.
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bin edges
        equally spaced in the square of the value (similar number of data
        points per bin on a 2D radial grid) and 'linear' yields linear bins.
        An array specifies the bin edges explicitly.
        (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging, see `make_grid`. For 'log'
        collocation, it is determined from `nb_points_per_decade` if set to
        None; 'linear' and 'quadratic' collocation require it.
        (Default: None)
    min_value : float, optional
        Minimum value, is automatically determined from the range of the data
        if not provided. (Default: None)
    max_value : float, optional
        Maximum value, is automatically determined from the range of the data
        if not provided. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)
    method : str, optional
        Resampling method; only 'bin-average', the average of the data
        within each bin, is supported. (Default: 'bin-average')

    Returns
    -------
    collocation_points : np.ma.MaskedArray
        Points where the data has been collected, the mean of the data
        points in each bin. Bins without data are masked.
    bin_edges : np.ndarray
        Bin edges.
    resampled_values : np.ma.MaskedArray
        Resampled values. Bins without data are masked.
    resampled_variance : np.ma.MaskedArray
        Variance of the data within each bin. Bins without data are
        masked.
    """
    # pylint: disable=invalid-name
    if max_value is None:
        max_value = np.max(x)
    if min_value is None:
        min_value = np.min(x)

    if min_value >= max_value:
        raise NoReliableDataError('Dataset contains no reliable data.')

    collocation_points, bin_edges = make_grid(collocation, min_value, max_value, nb_points=nb_points,
                                              nb_points_per_decade=nb_points_per_decade)

    if method == 'bin-average':
        collocation_points, resampled_values, resampled_variance = bin_average(collocation_points, bin_edges, x, values)
    else:
        raise ValueError(f"Unknown resampling method '{method}'.")

    return collocation_points, bin_edges, resampled_values, resampled_variance


def resample_radial(data, physical_sizes=None, collocation='log', nb_points=None, min_radius=0, max_radius=None,
                    nb_points_per_decade=5, full=True, method='bin-average'):
    r"""
    Compute radial average of quantities reported on a 2D grid and collect
    results at collocation points. The radius of grid point :math:`(i, j)`
    is :math:`\sqrt{(i L_x/N_x)^2 + (j L_y/N_y)^2}`; for `full=True`,
    indices beyond the middle of the grid are wrapped to
    :math:`N - i` (the layout of FFT output).

    Parameters
    ----------
    data : array_like
        2D-array of values to be averaged.
    physical_sizes : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of
        grid points.)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bin edges
        equally spaced in the square of the value (similar number of data
        points per bin on a 2D radial grid) and 'linear' yields linear bins.
        An array specifies the bin edges explicitly.
        (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging, see `make_grid`. For 'log'
        collocation, it is determined from `nb_points_per_decade` if set to
        None; 'linear' and 'quadratic' collocation require it.
        (Default: None)
    min_radius : float, optional
        Minimum radius. It is raised to at least the smaller grid spacing.
        (Default: 0)
    max_radius : float, optional
        Maximum radius, is automatically determined from the range of the data
        if not provided. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 5)
    full : bool, optional
        Number of quadrants contained in data.
        True: Full radial average from 0 to 2*pi; the data is in the
        wrap-around layout of FFT output.
        False: Only one quarter of the full circle is present, with the
        origin at index (0, 0). Radial average from 0 to pi/2.
        (Default: True)
    method : str, optional
        Resampling method; only 'bin-average', the average of the data
        within each bin, is supported. (Default: 'bin-average')

    Returns
    -------
    collocation_points : np.ma.MaskedArray
        Points where the data has been collected, the mean of the data
        points in each bin. Bins without data are masked.
    bin_edges : np.ndarray
        Bin edges.
    resampled_values : np.ma.MaskedArray
        Resampled values. Bins without data are masked.
    resampled_variance : np.ma.MaskedArray
        Variance of the data within each bin. Bins without data are
        masked.
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
