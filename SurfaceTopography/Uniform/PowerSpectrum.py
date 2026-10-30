#
# Copyright 2019, 2023 Antoine Sanner
#           2018-2021 Lars Pastewka
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

from ..Exceptions import NoReliableDataError, UndefinedDataError
from ..HeightContainer import UniformTopographyInterface
from ..Support import doi, fold_fft_half
from ..Support.Regression import resample, resample_radial


@doi('10.1088/2051-672X/aa51f8')
def power_spectrum_from_profile(self, window=None, reliable=True, resampling_method='bin-average',
                                collocation='log', nb_points=None, nb_points_per_decade=10):
    r"""
    Compute the one-dimensional power spectral density (PSD) of a line scan,
    or the average of the 1D PSDs of all profiles (along :math:`x`) of a
    topography map, from the FFT of the heights on a uniform grid.

    The PSD is

    .. math::

        C^\text{1D}(q_k) = \frac{1}{L}\,|\tilde h(q_k)|^2, \qquad
        \tilde h(q_k) = \Delta x \sum_{i=0}^{N-1} h_i e^{-i q_k x_i},

    with :math:`q_k = 2\pi k/L`. The values at :math:`+q_k` and
    :math:`-q_k` are averaged; wavevectors :math:`0 \le q_k < \pi/\Delta x`
    are returned. For topography maps, the PSDs of all profiles are averaged.

    Parameters
    ----------
    self : :obj:`Topography` or :obj:`UniformLineScan`
        Container with height information.
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        The window is normalized such that the mean of its square is one.
        For line scans, it is applied along :math:`x`; for topography maps,
        the rotationally symmetric 2D window of
        :func:`SurfaceTopography.FFTTricks.get_window_2D` is applied.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only return data deemed reliable, i.e. wavevectors
        :math:`q < 2\pi/\lambda_c`, where :math:`\lambda_c` is the short
        reliability cutoff. (Default: True)
    resampling_method : {None, 'bin-average', 'gaussian-process'}, optional
        None returns the PSD on the wavevectors of the FFT, 'bin-average'
        averages it within bins, 'gaussian-process' resamples it by
        Gaussian process regression. (Default: 'bin-average')
    collocation : {'log', 'quadratic', 'linear'}, optional
        Resampling grid. 'log' yields collocation points equally spaced on
        a log scale, 'quadratic' yields bins equally spaced in :math:`q^2`
        and 'linear' yields linear bins. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. For 'log' collocation, the number of
        bins is determined from `nb_points_per_decade` if set to None;
        'linear' and 'quadratic' collocation require this argument.
        (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)

    Returns
    -------
    q : np.ndarray or np.ma.MaskedArray
        Wavevectors. Bins without data are masked for 'bin-average'.
        (Units: 1/length)
    C : np.ndarray or np.ma.MaskedArray
        Power spectral density. (Units: length³)

    References
    ----------
    T. D. B. Jacobs, T. Junge, L. Pastewka, Surf. Topogr. Metrol. Prop. 5,
    013001 (2017), https://doi.org/10.1088/2051-672X/aa51f8
    """
    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Power-spectrum cannot be '
                                 'computed for topographies with missing data points.')

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
    C_all = fold_fft_half(C_raw, nx)

    if resampling_method is None:
        # Only keep reliable data
        short_cutoff = self.short_reliability_cutoff(2 * sx / nx) if reliable else None
        if short_cutoff is not None:
            mask = q < 2 * np.pi / short_cutoff
            if mask.sum() <= 1:  # There is always q=0 in there
                raise NoReliableDataError('Dataset contains no reliable data.')
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
        # The smallest positive wavevector limits the resampling range; note
        # that q = 0 has either been stripped above (log collocation) or
        # carries the (uninteresting) mean of the topography. Using q[1]
        # here would discard the fundamental mode when q = 0 has already
        # been stripped.
        q, _, C, _ = resample(q, C_all, min_value=np.min(q[q > 0]), max_value=short_cutoff, collocation=collocation,
                              nb_points=nb_points, nb_points_per_decade=nb_points_per_decade, method=resampling_method)
        return q, C


@doi('10.1088/2051-672X/aa51f8')
def power_spectrum_from_area(self, window=None, reliable=True, collocation='log', nb_points=None,
                             nb_points_per_decade=10, return_map=False, resampling_method='bin-average'):
    r"""
    Compute the two-dimensional power spectral density (PSD) of a
    topography map from its 2D FFT and average it radially.

    The PSD is

    .. math::

        C^\text{2D}(\mathbf q) = \frac{1}{L_x L_y}\,|\tilde h(\mathbf q)|^2,
        \qquad
        \tilde h(\mathbf q) = \Delta x \Delta y \sum_{i,j} h_{ij}
        e^{-i(q_x x_i + q_y y_j)}.

    The radial average :math:`C^\text{iso}(q)` is the average over all
    wavevectors :math:`\mathbf q` whose magnitude falls into a bin. Bins
    extend up to :math:`\pi\min(1/\Delta x, 1/\Delta y)`, or up to
    :math:`2\pi/\lambda_c` if a short reliability cutoff :math:`\lambda_c`
    is known and `reliable` is True.

    Parameters
    ----------
    self : :obj:`Topography`
        Container storing the (two-dimensional) topography map.
    window : str, optional
        Window for eliminating edge effects. The rotationally symmetric 2D
        window of :func:`SurfaceTopography.FFTTricks.get_window_2D` is
        applied, normalized such that the mean of its square is one.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    collocation : {'log', 'quadratic', 'linear'}, optional
        Resampling grid. 'log' yields collocation points equally spaced on
        a log scale, 'quadratic' yields bins equally spaced in :math:`q^2`
        (similar number of wavevectors per bin) and 'linear' yields linear
        bins. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. For 'log' collocation, the number of
        bins is determined from `nb_points_per_decade` if set to None;
        'linear' and 'quadratic' collocation require this argument.
        (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)
    return_map : bool, optional
        Return full 2D power spectrum map. (Default: False)
    resampling_method : {'bin-average', 'gaussian-process'}, optional
        'bin-average' for averaging within bins and 'gaussian-process' for
        Gaussian process regression. (Default: 'bin-average')

    Returns
    -------
    q : np.ndarray or np.ma.MaskedArray
        Wavevectors. Bins without data are masked for 'bin-average'.
        (Units: 1/length)
    C : np.ndarray or np.ma.MaskedArray
        Radially averaged power spectral density. (Units: length⁴)
    C_qk : np.ndarray, optional
        Full 2D power spectral density in the layout of `np.fft.fft2`,
        only returned if `return_map` is True. (Units: length⁴)

    References
    ----------
    T. D. B. Jacobs, T. Junge, L. Pastewka, Surf. Topogr. Metrol. Prop. 5,
    013001 (2017), https://doi.org/10.1088/2051-672X/aa51f8
    """
    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Power-spectrum cannot be '
                                 'computed for topographies with missing data points.')

    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes

    # Radial averaging is meaningful up to the smaller of the two Nyquist
    # frequencies; using only the x-Nyquist would make the result depend on
    # the orientation of the map for anisotropic pixels
    qmax = np.pi * min(nx / sx, ny / sy)

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


def fftfreq(t, meshgrid=True):
    r"""
    Compute and return the grid of wavevectors :math:`q_k = 2\pi k/L`
    corresponding to `np.fft.fft` (2D: `np.fft.fft2`).

    Parameters
    ----------
    t : UniformLineScan or Topography
        Container with height information.
    meshgrid : bool, optional
        If True, return wavevectors as a meshgrid for 2D topographies.
        If False, return separate 1D arrays. (Default: True)

    Returns
    -------
    qx : array_like
        Wavevectors in the x-direction. (Units: 1/length)
    qy : array_like, optional
        Wavevectors in the y-direction, only returned for 2D topographies.
        (Units: 1/length)
    """

    qx = 2 * np.pi * np.fft.fftfreq(
        t.nb_grid_pts[0],
        t.physical_sizes[0] / t.nb_grid_pts[0])
    if t.dim == 1:
        return qx
    elif t.dim == 2:
        qy = 2 * np.pi * np.fft.fftfreq(
            t.nb_grid_pts[1],
            t.physical_sizes[1] / t.nb_grid_pts[1])

        if meshgrid:
            qx, qy = np.meshgrid(qx, qy, indexing='ij')
        return qx, qy


def wavevectors_norm2(t, meshgrid=True):
    r"""
    Compute the norm (magnitude) :math:`|\mathbf q|` of the wavevectors
    returned by `fftfreq`. (Despite the name, this is the 2-norm, not its
    square.)

    Parameters
    ----------
    t : UniformLineScan or Topography
        Container with height information.
    meshgrid : bool, optional
        If True, return wavevector norms as a meshgrid for 2D topographies.
        If False, the wavevectors are computed without meshgrid before taking
        the norm. (Default: True)

    Returns
    -------
    q : array_like
        Norm of the wavevectors. (Units: 1/length)
    """
    if t.dim == 1:
        return np.abs(t.fftfreq(meshgrid=meshgrid))
    else:
        qx, qy = t.fftfreq(meshgrid=meshgrid)
        return np.sqrt(qx ** 2 + qy ** 2)


UniformTopographyInterface.register_function("fftfreq", fftfreq)
UniformTopographyInterface.register_function("wavevectors_norm2", wavevectors_norm2)


# Register analysis functions from this module
UniformTopographyInterface.register_function('power_spectrum_from_profile', power_spectrum_from_profile)
UniformTopographyInterface.register_function('power_spectrum_from_area', power_spectrum_from_area)
