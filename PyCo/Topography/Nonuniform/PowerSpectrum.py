#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   PowerSpectrum.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   17 Dec 2018

@brief  Power-spectral density for nonuniform topographies.

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def dsinc(x):
    """Derivative of the numpy.sinc function"""
    tol = 1e-6
    x = np.asarray(x)
    small_values = np.abs(x) < tol
    if small_values.sum() > 0:
        ret = np.zeros_like(x)

        # For small values, use the Taylor series expansion (accurate to O(x^^))
        ret[small_values] = -x[small_values] / 3 + x[small_values] ** 3 / 30

        # For large values, use the normal expression
        large_values = np.logical_not(small_values)
        ret[large_values] = (np.cos(x[large_values]) - np.sinc(x[large_values])) / x[large_values]
        return ret
    else:
        return (np.cos(np.pi * x) - np.sinc(x)) / x


def ft_rectangle(a, q):
    """
    Fourier transform of a rectangle, :math:`f(x) = 1` for :math:``|x| < a`,

    ..math ::

        \\tilde{f}(q) = 2a \\sin(aq)/aq = 2a \\sinc(aq)
    """
    # np.sinc is sin(pi*x)/(pi*x)
    return 2 * a * np.sinc(a * q / np.pi)


def ft_one_sided_triangle(a, q):
    """
    Fourier transform of the one-sided triangle, :math:`f(x) = x` for
    :math:`|x| < a`,

    ..math ::

        \\tilde{f}(q) = 2ia^2 [\\cos(aq)/(aq) - \\sin(aq)/(aq)^2]

    Returned value does not contain the factor of :math:`i`, i.e. this
    function is real-valued.
    """
    return 2 * a * a * dsinc(a * q / np.pi) / np.pi


def apply_window(x, y, window=None):
    if window == 'hann':
        l = x.max() - x.min()
        return (2 / 3) ** (1 / 2) * (1 - np.cos(2 * np.pi * x / l)) * y
    elif window is None or window == 'None':
        return y
    else:
        raise ValueError('Unknown window {}'.format(window))


def power_spectrum(x, y, q=None, window=None):
    """
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
        equally spaced with a spacing that corresponds to :math:`2\\pi/\lambda`
        where :math:`\\lambda` is the shortest distance between two points in
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
    y = apply_window(x, y, window=window)
    if q is None:
        L = x[-1] - x[0]
        q = 2 * np.pi * np.arange(int(L / np.diff(x).min())) / L
    y_q = np.zeros_like(q, dtype=np.complex128)
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        a = x2 - x1
        if a > 0:
            slope = (y2 - y1) / a
            a /= 2
            x0 = (x1 + x2) / 2
            y0 = (y1 + y2) / 2
            y_q += (y0 * ft_rectangle(a, q) + 1j * slope * ft_one_sided_triangle(a, q)) * np.exp(-1j * x0 * q) / (2 * a)
    return q, (x.max() - x.min()) * (np.abs(y_q) / len(x)) ** 2
