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
Bin for small common helper function and classes
"""

import numpy as np


def radial_average(C_xy, rmax=None, nbins=None, bin_edges='log',
                   physical_sizes=None, log_factor=1.1, full=True):
    """
    Compute radial average of quantities reported on a 2D grid.

    Either `nbins` or `bin_edges` must be present.

    Parameters
    ----------
    C_xy : array_like
        2D-array of values to be averaged.
    rmax : float, optional
        Maximum radius, is automatically determined from the range of the data
        if not provided. (Default: None)
    nbins : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    bin_edges : {'log', 'quadratic', 'linear', array_like}, optional
        Edges used for binning the average. Specifying 'log' yields bins
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
    physical_sizes : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of
        grid points.)
    log_factor : float, optional
        Factor between consecutive bin edges for log-spaced bins.
        (Default: 1.1)
    full : bool
        Number of quadrants contained in data. (Default: True)
        True: Full radial average from 0 to 2*pi.
        False: Only the one quarter of the full circle is present. Radial
        average from 0 to pi/2.

    Returns
    -------
    r_edges : array
        Bin edges.
    r_averages : array
        Bin centers, obtained by averaging actual distance values.
        (This is *not* the average of adjacent `bin_edges`!)
    n : array
        Number of data points per radial bin.
    C_r : array
        Averaged values.
    """
    # pylint: disable=invalid-name
    nx, ny = C_xy.shape
    x = np.arange(nx)
    y = np.arange(ny)
    if full:
        x = np.where(x > nx // 2, nx - x, x)
        y = np.where(y > ny // 2, ny - y, y)

    rmin = 1.0
    if physical_sizes is not None:
        sx, sy = physical_sizes
        x = sx / nx * x
        y = sy / ny * y
        rmin = min(sx / nx, sy / ny)
    dr_xy = np.sqrt((x ** 2).reshape(-1, 1) + (y ** 2).reshape(1, -1))
    if rmax is None:
        rmax = np.max(dr_xy)

    if bin_edges == 'log':
        # Power law -> equally spaced on a log-log plot
        if nbins is None:
            # The size of the first bin should be equal to rmin,
            # which is the (minimal) size of a grid point
            # i.e. rmin = np.exp(np.log(rmin) + dl) - rmin
            # => dl = np.log(2)
            nbins = int((np.log(rmax) - np.log(rmin)) / np.log(log_factor)) + 1
        dr_r = np.exp(np.linspace(np.log(rmin), np.log(rmax), nbins - 1))
    elif bin_edges == 'quadratic':
        # Quadratic -> similar statistics for each data point
        if nbins is None:
            raise ValueError("Please specify number of bins for 'quadratic' "
                             "bins.")
        dr_r = np.sqrt(np.linspace(0, rmax ** 2, nbins)[1:])
    elif bin_edges == 'linear':
        # Linear
        if nbins is None:
            nbins = int(rmax / rmin) + 1
        dr_r = np.linspace(0, rmax, nbins)[1:]
    else:
        dr_r = bin_edges

    # Linear interpolation
    dr_xy = np.ravel(dr_xy)
    C_xy = np.ravel(C_xy)
    i_xy = np.searchsorted(dr_r, dr_xy)

    n_r = np.bincount(i_xy, minlength=len(dr_r))
    dravg_r = np.bincount(i_xy, weights=dr_xy, minlength=len(dr_r))
    C_r = np.bincount(i_xy, weights=C_xy, minlength=len(dr_r))

    nreg_r = np.where(n_r == 0, np.ones_like(n_r), n_r)
    dravg_r /= nreg_r
    C_r /= nreg_r

    return np.append([0.0], dr_r), n_r, dravg_r, C_r
