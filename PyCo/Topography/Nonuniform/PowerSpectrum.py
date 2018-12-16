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


def ft_rectangle(a, q):
    """
    Fourier transform of a rectangle, f(x) = 1 for |x| < a:
      2a sin(aq)/aq = 2a sinc(aq)
    """
    return 2 * a * np.sin(a * q) / (a * q)


def ft_one_sided_triangle(a, q):
    """
    Fourier transform of the one-sided triangle, f(x) = x for |x| < a:
      2ia^2 [cos(aq)/(aq) - sin(aq)/(aq)^2]
    """
    return 2j * a * a * (np.cos(a * q) / (a * q) - np.sin(a * q) / ((a * q) ** 2))


def apply_window(x, y, window):
    if window == 'hann':
        l = x.max() - x.min()
        return (2 / 3) ** (1 / 2) * (1 - np.cos(2 * np.pi * x / l)) * y
    elif window is None or window == 'None':
        return y
    else:
        raise ValueError('Unknown window {}'.format(window))


def power_spectrum(x, y, q=None, window=None):
    y = apply_window(x, y, window)
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
            y_q += (y0 * ft_rectangle(a, q) + slope * ft_one_sided_triangle(a, q)) * np.exp(-1j * x0 * q) / (2 * a)
    return (x.max() - x.min()) * (np.abs(y_q) / len(x)) ** 2
