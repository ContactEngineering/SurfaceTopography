#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   PowerSpectrum.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 May 2018

@brief  Power-spectral density for uniform topographies.

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
from scipy.signal import get_window

from .common import _get_size
from ..common import radial_average


def power_spectrum_1D(topography,  # pylint: disable=invalid-name
                      size=None, window=None):
    """
    Compute power spectrum from 1D FFT.

    Parameters
    ----------
    topography : array_like
        2D-array of surface topography
    size : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of grid
        points.)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: None)

    Returns
    -------
    q : array_like
        Reciprocal space vectors.
    C_all : array_like
        Power spectrum. (Units: length**3)
    """
    # pylint: disable=invalid-name
    if hasattr(topography, "power_spectrum_1D"):
        return topography.power_spectrum_1D(window=window)

    n = topography.shape
    s = _get_size(topography, size)

    try:
        nx, ny = n
        sx, sy = s
    except:
        nx, = n
        sx, = s

    # Construct and apply window
    if window is not None:
        win = get_window(window, nx)
        # Normalize window
        win *= np.sqrt(nx/(win**2).sum())
        topography = (win * topography.T).T

    # Pixel size
    len0 = sx/nx

    # Compute FFT and normalize
    fourier_topography = len0*np.fft.fft(topography, axis=0)
    dq = 2*np.pi/sx
    q = dq*np.arange(nx//2)

    # This is the raw power spectral density
    C_raw = (np.abs(fourier_topography)**2)/sx

    # Fold +q and -q branches. Note: Entry q=0 appears just once, hence exclude
    # from average!
    C_all = C_raw[:nx//2, ...]
    C_all[1:nx//2, ...] += C_raw[nx-1:(nx+1)//2:-1, ...]
    C_all /= 2

    if len(topography.shape) == 1:
        return q, C_all
    else:
        return q, C_all.mean(axis=1)


def get_window_2D(window, nx, ny, size=None):
    if isinstance(window, np.ndarray):
        if window.shape != (nx, ny):
            raise TypeError('Window size (= {2}x{3}) must match signal size '
                            '(={0}x{1})'.format(nx, ny, *window.shape))
        return window

    if size is None:
        sx, sy = nx, ny
    else:
        sx, sy = size
    if window == 'hann':
        maxr = min(sx, sy)/2
        r = np.sqrt((sx*(np.arange(nx).reshape(-1,1)-nx//2)/nx)**2 +
                    (sy*(np.arange(ny).reshape(1,-1)-ny//2)/ny)**2)
        win = 0.5+0.5*np.cos(np.pi*r/maxr)
        win[r>maxr] = 0.0
        return win
    else:
        raise ValueError("Unknown window type '{}'".format(window))


def power_spectrum_2D(topography, nbins=100,  # pylint: disable=invalid-name
                      size=None, window=None, normalize_window=True,
                      return_map=False):
    """
    Compute power spectrum from 2D FFT and radial average.

    Parameters
    ----------
    topography : array_like
        2D-array of surface topography
    nbins : int
        Number of bins for radial average. Note: Returned array can be smaller
        than this because bins without data point are discarded.
    size : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of grid
        points.)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: None)
    normalize_window : bool, optional
        Normalize window to unit mean. (Default: True)
    return_map : bool, optional
        Return full 2D power spectrum map. (Default: False)

    Returns
    -------
    q : array_like
        Reciprocal space vectors.
    C_all : array_like
        Power spectrum. (Units: length**4)
    """
    nx, ny = topography.shape
    sx, sy = _get_size(topography, size)

    # Construct and apply window
    if window is not None:
        win = get_window_2D(window, nx, ny, size)
        # Normalize window
        if normalize_window:
            win *= np.sqrt(nx*ny/(win**2).sum())
        topography = win * topography[:, :]

    # Pixel size
    area0 = (sx/nx)*(sy/ny)

    # Compute FFT and normalize
    surface_qk = area0*np.fft.fft2(topography[:, :])
    C_qk = abs(surface_qk)**2/(sx*sy)  # pylint: disable=invalid-name

    if nbins is None:
        return C_qk

    # Radial average
    q_edges, n, q_val, C_val = radial_average(  # pylint: disable=invalid-name
        C_qk, 2*np.pi*nx/(2*sx), nbins, size=(2*np.pi*nx/sx, 2*np.pi*ny/sy))

    if return_map:
        return q_val[n>0], C_val[n>0], C_qk
    else:
        return q_val[n>0], C_val[n>0]
