#
# Copyright 2016, 2018-2020 Lars Pastewka
#           2019 Antoine Sanner
#           2019 Michael Röttger
#           2015-2016 Till Junge
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
Helper functions to compute trends of surfaces
"""

import numpy as np


def tilt_from_height(topography, full_output=False):
    """
    Compute the tilt plane that if subtracted minimizes the rms height of the
    surface. The tilt plane is parameterized as:

    .. math::

        p(x, y) = h_0 + m x + n y

    The values of :math:`m`, :math:`n` and :math:`h0` are return by this
    function.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()

    Parameters
    ----------
    arr : UniformTopography
        Height information.

    Returns
    -------
    m : float
        Slope in x-direction.
    n : float
        Slope in y-direction.
    h0 : float
        Mean value.
    """
    arr = topography.heights()
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) / arr.shape[i] for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def tilt_and_curvature(arr, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()

    Returns:
    ---------
    coeffs [, location_matrix (if full_output)]

    coeffs ordered as follows

    {5} + {0} x + {1} y + {2} x^2 + {3} y^2 + {4} xy

    """
    arr = arr[...]
    nb_dim = len(arr.shape)
    assert nb_dim == 2
    x_grids = (np.arange(arr.shape[i]) / arr.shape[i] for i in range(nb_dim))
    # Linear terms
    x_grids = np.meshgrid(*x_grids, indexing='ij')
    # Quadratic terms
    x, y = x_grids
    x_grids += [x * x, y * y, x * y]
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def shift_and_tilt(topography):
    """
    returns an array of same shape and physical_sizes as arr, but shifted and
    tilted so that mean(arr) = 0 and mean(arr**2) is minimized
    """
    arr = topography.heights()
    coeffs, location_matrix = tilt_from_height(topography, full_output=True)
    coeffs = np.array(coeffs)
    offsets = arr.reshape((-1,))

    return (offsets - location_matrix @ coeffs).reshape(arr.shape)
