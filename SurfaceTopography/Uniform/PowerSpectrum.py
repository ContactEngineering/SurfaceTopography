#
# Copyright 2018, 2020 Lars Pastewka
#           2019 Antoine Sanner
#           2019 Michael Röttger
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
Power-spectral density for uniform topographies.
"""

import numpy as np
from scipy.signal import get_window

from ..common import radial_average
from ..FFTTricks import get_window_2D
from ..HeightContainer import UniformTopographyInterface


def power_spectrum_1D(topography,  # pylint: disable=invalid-name
                      window=None):
    """
    Compute power spectrum from 1D FFT(s) of a topography or line scan
    stored on a uniform grid.

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
        Container with height information.
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies

    Returns
    -------
    q : array_like
        Reciprocal space vectors.
    C_all : array_like
        Power spectrum. (Units: length**3)
    """
    n = topography.nb_grid_pts
    s = topography.physical_sizes

    try:
        nx, ny = n
        sx, sy = s
    except ValueError:
        nx, = n
        sx, = s

    h = topography.heights()

    if not topography.is_periodic and window is None:
        window = "hann"

    # Construct and apply window
    if window is not None and window != 'None':
        win = get_window(window, nx)
        # Normalize window
        win *= np.sqrt(nx / (win ** 2).sum())
        h = (win * h.T).T

    # Pixel physical_sizes
    len0 = sx / nx

    # Compute FFT and normalize
    fourier_topography = len0 * np.fft.fft(h, axis=0)
    dq = 2 * np.pi / sx
    q = dq * np.arange(nx // 2)

    # This is the raw power spectral density
    C_raw = (np.abs(fourier_topography) ** 2) / sx

    # Fold +q and -q branches. Note: Entry q=0 appears just once, hence
    # exclude from average!
    C_all = C_raw[:nx // 2, ...]
    C_all[1:nx // 2, ...] += C_raw[nx - 1:(nx + 1) // 2:-1, ...]
    C_all /= 2

    if topography.dim == 1:
        return q, C_all
    else:
        return q, C_all.mean(axis=1)


def power_spectrum_2D(topography, nbins=100,  # pylint: disable=invalid-name
                      window=None, normalize_window=True,
                      return_map=False):
    """
    Compute power spectrum from 2D FFT and radial average of a topography
    stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        Container storing the (two-dimensional) topography map.
    nbins : int
        Number of bins for radial average. Note: Returned array can be smaller
        than this because bins without data point are discarded.
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
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    if not topography.is_periodic and window is None:
        window = "hann"

    # Construct and apply window
    if window is not None:
        win = get_window_2D(window, nx, ny, topography.physical_sizes)
        # Normalize window
        if normalize_window:
            win *= np.sqrt(nx * ny / (win ** 2).sum())
        topography = win * topography[:, :]

    # Pixel physical_sizes
    area0 = (sx / nx) * (sy / ny)

    # Compute FFT and normalize
    surface_qk = area0 * np.fft.fft2(topography[:, :])
    C_qk = abs(surface_qk) ** 2 / (sx * sy)  # pylint: disable=invalid-name

    if nbins is None:
        return C_qk

    # Radial average
    q_edges, n, q_val, C_val = radial_average(  # pylint: disable=invalid-name
        C_qk, 2 * np.pi * nx / (2 * sx), nbins,
        physical_sizes=(2 * np.pi * nx / sx, 2 * np.pi * ny / sy))

    if return_map:
        return q_val[n > 0], C_val[n > 0], C_qk
    else:
        return q_val[n > 0], C_val[n > 0]


# Register analysis functions from this module
UniformTopographyInterface.register_function('power_spectrum_1D',
                                             power_spectrum_1D)
UniformTopographyInterface.register_function('power_spectrum_2D',
                                             power_spectrum_2D)
