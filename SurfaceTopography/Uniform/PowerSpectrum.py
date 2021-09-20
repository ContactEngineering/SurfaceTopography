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

from SurfaceTopography.Support.Regression import resample, resample_radial
from ..HeightContainer import UniformTopographyInterface


def power_spectrum_from_profile(self, window=None, reliable=True, resampling_method='bin-average',
                                collocation='log', nb_points=None, nb_points_per_decade=10):
    """
    Compute power spectrum from 1D FFT(s) of a topography or line scan
    stored on a uniform grid.

    Parameters
    ----------
    self : SurfaceTopography or UniformLineScan
        Container with height information.
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    resampling_method : str, optional
        Method can be None for no resampling (return on the grid of the
        data) 'bin-average' for simple bin averaging and 'gaussian-process'
        for Gaussian process regression.
        (Default: 'bin-average')
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)

    Returns
    -------
    q : array_like
        Reciprocal space vectors.
    C_all : array_like
        Power spectrum. (Units: length**3)
    """
    try:
        nx, ny = self.nb_grid_pts
        sx, sy = self.physical_sizes
    except ValueError:
        nx, = self.nb_grid_pts
        sx, = self.physical_sizes

    h = self.window(window).heights()

    # Compute FFT and normalize
    fourier_topography = sx / nx * np.fft.fft(h, axis=0)
    dq = 2 * np.pi / sx
    q = dq * np.arange(nx // 2)

    # This is the raw power spectral density
    C_raw = (np.abs(fourier_topography) ** 2) / sx

    # Fold +q and -q branches. Note: Entry q=0 appears just once, hence
    # exclude from average!
    C_all = C_raw[:nx // 2, ...]
    C_all[1:nx // 2, ...] += C_raw[nx - 1:(nx + 1) // 2:-1, ...]
    C_all /= 2

    if resampling_method is None:
        # Only keep reliable data
        short_cutoff = self.short_reliability_cutoff() if reliable else None
        if short_cutoff is not None:
            mask = q < 2 * np.pi / short_cutoff
            q = q[mask]
            C_all = C_all[mask]
        if self.dim == 1:
            return q, C_all
        else:
            return q, C_all.mean(axis=1)
    else:
        short_cutoff = 2 * np.pi / self.short_reliability_cutoff(2 * sx / nx) if reliable else np.pi * nx / sx
        if collocation == 'log':
            # Exclude zero distance because that does not work on a log-scale
            q = q[1:]
            C_all = C_all[1:]
        if self.dim == 2:
            q = np.resize(q, (C_all.shape[1], q.shape[0])).T.ravel()
            C_all = np.ravel(C_all)
        q, _, C, _ = resample(q, C_all, min_value=q[1], max_value=short_cutoff, collocation=collocation,
                              nb_points=nb_points, nb_points_per_decade=nb_points_per_decade, method=resampling_method)
        return q, C


def power_spectrum_from_area(self, window=None, reliable=True, collocation='log', nb_points=None,
                             nb_points_per_decade=10, return_map=False, resampling_method='bin-average'):
    """
    Compute power spectrum from 2D FFT and radial average of a topography
    stored on a uniform grid.

    Parameters
    ----------
    self : :obj:`SurfaceTopography`
        Container storing the (two-dimensional) topography map.
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: None)
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. Bins are automatically determined if set
        to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)
    return_map : bool, optional
        Return full 2D power spectrum map. (Default: False)
    resampling_method : str, optional
        Method can be 'bin-average' for simple bin averaging and
        'gaussian-process' for Gaussian process regression.
        (Default: 'bin-average')

    Returns
    -------
    q : array_like
        Reciprocal space vectors.
    C_all : array_like
        Power spectrum. (Units: length**4)
    """
    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes

    qmax = 2 * np.pi * nx / (2 * sx)

    if reliable:
        # Update qmax
        short_cutoff = self.short_reliability_cutoff()
        if short_cutoff is not None:
            qmax = 2 * np.pi / short_cutoff

    h = self.window(window=window, direction='radial').heights()

    # Compute FFT and normalize
    surface_qk = self.area_per_pt * np.fft.fft2(h)
    C_qk = abs(surface_qk) ** 2 / (sx * sy)  # pylint: disable=invalid-name

    # Radial average
    q_val, q_edges, C_val, _ = resample_radial(C_qk, physical_sizes=(2 * np.pi * nx / sx, 2 * np.pi * ny / sy),
                                               collocation=collocation, nb_points=nb_points,
                                               nb_points_per_decade=nb_points_per_decade, max_radius=qmax,
                                               method=resampling_method)

    if return_map:
        return q_val, C_val, C_qk
    else:
        return q_val, C_val


# Register analysis functions from this module
UniformTopographyInterface.register_function('power_spectrum_from_profile', power_spectrum_from_profile)
UniformTopographyInterface.register_function('power_spectrum_from_area', power_spectrum_from_area)
