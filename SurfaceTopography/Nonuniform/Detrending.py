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
Helper functions to compute trends of surfaces
"""

import numpy as np


def polyfit(x, h, deg):
    r"""
    Compute the detrending plane that, if subtracted, minimizes the rms height
    of the surface. The detrending plane is parameterized as a polynomial:

    .. math::

        p(x) = \sum_{k=0}^n a_k x^k

    The values of :math:`a_k` are returned by this function.

    The rms height of the surface is given by (see `rms_height`)

    .. math::

        h_\text{rms}^2 = \frac{1}{3L} \sum_{i=0}^{N-2} \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i

    where :math:`N` is the total number of data points. Hence we need to solve the following minimization problem:

    .. math::

        \min_{\{a_k\}} \left\{ \frac{1}{3L} \sum_{i=0}^{N-2} \left[ (h_i - p(x_i))^2 + (h_{i+1} - p(x_{i+1}))^2 + (h_i - p(x_i))(h_{i+1} - p(x_{i+1})) \right] \Delta x_i \right\}

    This gives the system of linear equations (one for each :math:`k`)

    .. math::

        \sum_{i=0}^{N-2} \left( \left[ 2h_i + h_{i+1} \right] x_i^k + \left[ 2h_{i+1} + h_i \right] x_{i+1}^k \right) \Delta x_i = \sum_{l=0}^n a_l \sum_{i=0}^{N-2} \left( 2x_i^{k+l} + 2x_{i+1}^{k+l} + x_i^k x_{i+1}^l + x_{i+1}^k x_i^l \right) \Delta x_i

    Parameters
    ----------
    x : array_like
        Array containing positions. This function assumes that this array is
        sorted in ascending order.
    h : array_like
        Array containing heights.
    deg : int
        Degree of polynomial :math:`n`.

    Returns
    -------
    a : array
        Array with coefficients :math:`a_k`.
    """  # noqa: E501
    dx = np.diff(x)
    k = np.arange(deg + 1).reshape(-1, 1)
    b = np.sum(((2 * h[:-1] + h[1:]) * x[:-1] ** k +
                (2 * h[1:] + h[:-1]) * x[1:] ** k) * dx,
               axis=1)
    L = k.reshape(1, -1, 1)
    k = k.reshape(-1, 1, 1)
    A = np.sum((2 * x[:-1] ** (k + L) + 2 * x[1:] ** (k + L) +
                x[:-1] ** k * x[1:] ** L + x[1:] ** k * x[:-1] ** L) * dx,
               axis=2)
    return np.linalg.solve(A, b)
