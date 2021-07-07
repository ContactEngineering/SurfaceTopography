#
# Copyright 2018-2021 Lars Pastewka
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

from ..common import radial_average
from ..HeightContainer import UniformTopographyInterface


def power_spectrum_from_profile(topography,  # pylint: disable=invalid-name
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

    h = topography.window(window).heights()

    # Compute FFT and normalize
    fourier_topography = sx/nx * np.fft.fft(h, axis=0)
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


def power_spectrum_from_area(topography, nbins=None,  # pylint: disable=invalid-name
                             bin_edges='log',
                             window=None, normalize_window=True,
                             return_map=False):
    """
    Compute power spectrum from 2D FFT and radial average of a topography
    stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        Container storing the (two-dimensional) topography map.
    nbins : int, optional
        Number of bins for radial average. Bins are automatically determined
        if set to None. (Default: None) Note: Returned array can be smaller
        than this because bins without data points are discarded.
        (Default: None)
    bin_edges : {'log', 'quadratic', 'linear', array_like}, optional
        Edges used for binning the average. Specifying 'log' yields bins
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
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

    h = topography.window(window=window, direction='radial').heights()

    # Compute FFT and normalize
    surface_qk = topography.area_per_pt * np.fft.fft2(h)
    C_qk = abs(surface_qk) ** 2 / (sx * sy)  # pylint: disable=invalid-name

    # Radial average
    q_edges, n, q_val, C_val = radial_average(  # pylint: disable=invalid-name
        C_qk, rmax=2 * np.pi * nx / (2 * sx),
        nbins=nbins, bin_edges=bin_edges,
        physical_sizes=(2 * np.pi * nx / sx, 2 * np.pi * ny / sy))

    if return_map:
        return q_val[n > 0], C_val[n > 0], C_qk
    else:
        return q_val[n > 0], C_val[n > 0]


# Register analysis functions from this module
UniformTopographyInterface.register_function('power_spectrum_from_profile',
                                             power_spectrum_from_profile)
UniformTopographyInterface.register_function('power_spectrum_from_area',
                                             power_spectrum_from_area)
