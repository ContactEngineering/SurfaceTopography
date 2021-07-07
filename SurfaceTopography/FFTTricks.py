#
# Copyright 2020 Lars Pastewka
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
import muFFT

import numpy as np


def make_fft(topography, fft='mpi'):
    """
    Instantiate a muFFT object that can compute the Fourier transform of the
    topography and has the same decomposition layout (or raise an error if
    this is not the case).

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        Container storing the topography map.
    """

    # We only initialize this once and attach it to the topography object
    if hasattr(topography, '_mufft'):
        return topography._mufft

    if topography.is_domain_decomposed:
        fft = muFFT.FFT(topography.nb_grid_pts, fft=fft, communicator=topography.communicator)
        if fft.subdomain_locations != topography.subdomain_locations or \
                fft.nb_subdomain_grid_pts != topography.nb_subdomain_grid_pts:
            raise RuntimeError('muFFT suggested a domain decomposition that '
                               'differs from the decomposition of the topography.')
    else:
        fft = muFFT.FFT(topography.nb_grid_pts)
    topography._mufft = fft
    return fft


def get_window_2D(window, nx, ny, physical_sizes=None):
    """
    Construct a rotationally symmetric window for windowing two-dimensional
    signals (fields).

    Parameters
    ----------
    window : str or np.ndarray
        If window is an np.ndarray, it is just passed through (i.e.
        immediately returned) from this function. If window is a string,
        then a corresponding window will be constructed. Currently the only
        support window is 'hann'.
    nx : int
        Number of grid points of the signal in x-direction
    ny : int
        Number of grid points of the signal in y-direction
    physical_sizes : tuple of float
        The physical size of the underlying signal. If None, then the number
        of grid points nx, ny are used as the physical size. This parameter
        is used to adjust the aspect ratio of the signal. (Default: None)

    Returns
    -------
    window : np.ndarray
        Numerical window, multiply the signal with this window
    """

    if isinstance(window, np.ndarray):
        if window.shape != (nx, ny):
            raise TypeError('Window shape (= {0}x{1}) must match the number of points of the signal (={2}x{3})'
                            .format(*window.shape, nx, ny))
        return window

    if physical_sizes is None:
        sx, sy = nx, ny
    else:
        sx, sy = physical_sizes
    if window == 'hann':
        maxr = min(sx, sy) / 2
        r = np.sqrt((sx * (np.arange(nx).reshape(-1, 1) - nx // 2) / nx) ** 2 +
                    (sy * (np.arange(ny).reshape(1, -1) - ny // 2) / ny) ** 2)
        win = 0.5 + 0.5 * np.cos(np.pi * r / maxr)
        win[r > maxr] = 0.0
        return win
    else:
        raise ValueError("Unknown window type '{}'".format(window))
