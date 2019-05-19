#
# Copyright 2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
Power-spectral density for nonuniform topographies.
"""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface


def sinc(x):
    """sinc function"""
    return np.sinc(x/np.pi)


def dsinc(x):
    """Derivative of the sinc function"""
    tol = 1e-6
    x = np.asarray(x)
    small_values = np.abs(x) < tol
    if small_values.sum() > 0:
        ret = np.zeros_like(x)

        # For small values, use the Taylor series expansion (accurate to O(x^^))
        ret[small_values] = -x[small_values] / 3 + x[small_values] ** 3 / 30

        # For large values, use the normal expression
        large_values = np.logical_not(small_values)
        ret[large_values] = (np.cos(x[large_values]) - sinc(x[large_values])) / x[large_values]
        return ret
    else:
        return (np.cos(x) - sinc(x)) / x


def ft_rectangle(q):
    r"""
    Fourier transform of a rectangle, :math:`f(x) = 1` for :math:`|x| < 1/2`,

    .. math ::

        \tilde{f}(q) = 2\sin(q/2)/q = sinc(q/2)
    """
    # np.sinc is sin(pi*x)/(pi*x), sinc is sin(x)/xCo
    return sinc(q / 2)


def ft_one_sided_triangle(q):
    r"""
    Fourier transform of the one-sided triangle, :math:`f(x) = x` for
    :math:`|x| < 1/2`,

    .. math ::

        \tilde{f}(q) = i [\cos(q/2)/q - 2\sin(q/2)/q^2] = d sinc(q/2)/dq

    Returned value does not contain the factor of :math:`i`, i.e. this
    function is real-valued.
    """
    return dsinc(q / 2) / 2


def apply_window(x, y, window=None):
    if window == 'hann':
        l = x.max() - x.min()
        return (2 / 3) ** (1 / 2) * (1 - np.cos(2 * np.pi * x / l)) * y
    elif window is None or window == 'None':
        return y
    else:
        raise ValueError('Unknown window {}'.format(window))


def power_spectrum_1D(topography, q=None, window=None):
    r"""
    Compute power-spectral density (PSD) for a nonuniform topography. The
    topography is assumed to be given by a series of points connected by
    straight lines.

    Parameters
    ----------
    x : array
        x-coordinates of the points.
    y : array
        y-coordinates of the points.
    q : array, optional
        Wavevectors at which to compute the PSD. If omitted, wavevectors are
        equally spaced with a spacing that corresponds to :math:`2\pi/\lambda`
        where :math:`\lambda` is the shortest distance between two points in
        the `x`-array. (Default: None)
    window : str, optional
        Name of the window function to apply before computing the PSD.
        Presently only supports Hann window ('hann') or no window (None or
        'None').
        (Default: None)

    Returns
    -------
    q : array
        Wavevector array at which the PSD has been computed.
    C : array
        PSD values.
    """
    x, y = topography.positions_and_heights()
    y = apply_window(x, y, window=window)
    L = x[-1] - x[0]
    if q is None:
        q = 2 * np.pi * np.arange(int(L / np.diff(x).min())) / L
    y_q = np.zeros_like(q, dtype=np.complex128)
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        dx = x2 - x1
        if dx > 0:
            dy = y2 - y1
            x0 = (x1 + x2) / 2
            y0 = (y1 + y2) / 2
            y_q += dx * (y0 * ft_rectangle(q * dx) + 1j * dy * ft_one_sided_triangle(q * dx)) * np.exp(-1j * x0 * q)
        else:
            raise ValueError('Nonuniform data points must be sorted in order of ascending x-values.')
    return q, np.abs(y_q) ** 2 / L


### Register analysis functions from this module

NonuniformLineScanInterface.register_function('power_spectrum_1D', power_spectrum_1D)
