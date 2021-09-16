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
to another.
"""

import numpy as np


def make_grid(collocation, min_value, max_value, nb_points=None, nb_points_per_decade=5):
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
            nb_points = int(np.ceil((np.log10(max_value) - np.log10(min_value) + 1) * nb_points_per_decade)) + 1
        bin_edges = np.linspace(np.log10(min_value), np.log10(max_value), nb_points - 1)
        collocation_points = np.append([0], 10 ** ((bin_edges[1:] + bin_edges[:-1]) / 2))
        bin_edges = np.append([0], 10 ** bin_edges)
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


def resample_radial(data, physical_sizes=None, collocation='log', nb_points=None, max_radius=None,
                    nb_points_per_decade=5, full=True):
    """
    Compute radial average of quantities reported on a 2D grid and collect
    results of collocation points.

    Either `nb_bins` or `collocation` must be present.

    Parameters
    ----------
    data : array_like
        2D-array of values to be averaged.
    max_radius : float, optional
        Maximum radius, is automatically determined from the range of the data
        if not provided. (Default: None)
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
    physical_sizes : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of
        grid points.)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)
    full : bool, optional
        Number of quadrants contained in data.
        True: Full radial average from 0 to 2*pi.
        False: Only the one quarter of the full circle is present. Radial
        average from 0 to pi/2.
        (Default: True)

    Returns
    -------
    collocation_points : np.ndarray
        Points where the data has been collected.
    bin_edges : np.ndarray
        Bin edges.
    number_of_data_points : np.ndarray
        Number of data points per radial bin.
    resampled_values : np.ndarray
        Resampled values.
    """
    # pylint: disable=invalid-name
    nx, ny = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    if full:
        x = np.where(x > nx // 2, nx - x, x)
        y = np.where(y > ny // 2, ny - y, y)

    min_value = 1.0
    if physical_sizes is not None:
        sx, sy = physical_sizes
        x = sx / nx * x
        y = sy / ny * y
        min_value = min(sx / nx, sy / ny)
    radius = np.sqrt((x ** 2).reshape(-1, 1) + (y ** 2).reshape(1, -1))
    if max_radius is None:
        max_radius = np.max(radius)

    collocation_points, bin_edges = make_grid(collocation, min_radius, max_radius, nb_points=nb_points,
                                              nb_points_per_decade=nb_points_per_decade)

    # Flatten array and find bin index of flattened arrays
    radius = np.ravel(radius)
    data = np.ravel(data)
    bin_index = np.searchsorted(collocation_points, bin_edges, radius)

    # Compute averages within bins
    number_of_data_points = np.bincount(bin_index, minlength=len(collocation_points) + 1)
    resampled_values = np.bincount(bin_index, weights=data, minlength=len(collocation_points) + 1) / \
                       np.where(number_of_data_points == 0, np.ones_like(number_of_data_points), number_of_data_points)

    # We discard the final element as it contains data points outside our binned region
    return collocation_points, bin_edges, number_of_data_points[:-1], resampled_values[:-1]
