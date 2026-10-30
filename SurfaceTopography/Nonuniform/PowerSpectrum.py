#
# Copyright 2018-2022 Lars Pastewka
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
Power-spectral density for nonuniform topographies.
"""

import numpy as np

from ..Exceptions import NoReliableDataError, ReentrantDataError
from ..HeightContainer import NonuniformLineScanInterface


def sinc(x):
    r"""Unnormalized sinc function, :math:`\sin(x)/x`."""
    return np.sinc(x / np.pi)


def dsinc(x):
    r"""Derivative of the unnormalized sinc function, :math:`d/dx\,[\sin(x)/x]`."""
    # Note: cos(x) - sinc(x) is formed by cancellation of two O(1) terms
    # and loses accuracy for small x; the Taylor branch is accurate to
    # machine precision up to about x = 1e-2, so the crossover sits where
    # both branches are accurate. (A threshold of 1e-6 would leave a band
    # with relative errors up to 1e-3 just above it.)
    tol = 1e-2
    x = np.asarray(x, dtype=float)
    small_values = np.abs(x) < tol
    if small_values.sum() > 0:
        ret = np.zeros_like(x)

        # For small values, use the Taylor series expansion
        # (accurate to O(x⁴))
        ret[small_values] = -x[small_values] / 3 + x[small_values] ** 3 / 30

        # For large values, use the normal expression
        large_values = np.logical_not(small_values)
        ret[large_values] = (np.cos(x[large_values]) -
                             sinc(x[large_values])) / x[large_values]
        return ret
    else:
        return (np.cos(x) - sinc(x)) / x


def ft_rectangle(q):
    r"""
    Fourier transform of a rectangle, :math:`f(x) = 1` for :math:`|x| < 1/2`,

    .. math::

        \tilde{f}(q) = 2\sin(q/2)/q = \operatorname{sinc}(q/2)
    """
    # np.sinc is sin(pi*x)/(pi*x), sinc is sin(x)/x
    return sinc(q / 2)


def ft_one_sided_triangle(q):
    r"""
    Fourier transform of the one-sided triangle, :math:`f(x) = x` for
    :math:`|x| < 1/2`,

    .. math::

        \tilde{f}(q) = i [\cos(q/2)/q - 2\sin(q/2)/q^2]
        = i\,\frac{d}{dq}\operatorname{sinc}(q/2)

    Returned value does not contain the factor of :math:`i`, i.e. this
    function is real-valued.
    """
    return dsinc(q / 2) / 2


def apply_window(x, y, window=None):
    r"""
    Apply a window function to the heights of a nonuniform line scan.

    The Hann window is evaluated at the positions of the data points,
    :math:`w(x) = \sqrt{2/3}\,[1 - \cos(2\pi(x - x_0)/L)]`; the prefactor
    normalizes the mean of :math:`w^2` over the scan to one.

    Parameters
    ----------
    x : array_like
        Position array.
    y : array_like
        Height array.
    window : str, optional
        Name of window function. Supported: 'hann' or None. (Default: None)

    Returns
    -------
    y_windowed : array_like
        Height array with window applied.
    """
    if window == 'hann':
        length = x.max() - x.min()
        # The window argument must be relative to the start of the scan;
        # the x-coordinates do not necessarily start at zero.
        return (2 / 3) ** (1 / 2) * (1 - np.cos(2 * np.pi * (x - x.min()) / length)) * y
    elif window is None or window == 'None':
        return y
    else:
        raise ValueError('Unknown window {}'.format(window))


def power_spectrum(self, reliable=True, algorithm='fft', wavevectors=None, nb_interpolate=5, short_cutoff=np.mean,
                   window=None, resampling_method='bin-average', collocation='log', nb_points=None,
                   nb_points_per_decade=10):
    r"""
    Compute the power spectral density (PSD) of a nonuniform line scan.
    The line scan is treated as a piecewise linear function, i.e. the data
    points are connected by straight lines.

    The 'fft' algorithm interpolates this function linearly onto a uniform
    grid whose spacing is the smallest distance between data points divided
    by `nb_interpolate` and then computes the PSD of the resulting uniform
    line scan with the FFT, see
    :func:`SurfaceTopography.Uniform.PowerSpectrum.power_spectrum_from_profile`.

    The 'brute-force' algorithm evaluates the Fourier integral of the
    piecewise linear function exactly at the requested wavevectors. With
    :math:`\Delta x_i = x_{i+1} - x_i`, segment midpoints
    :math:`\bar x_i = (x_i + x_{i+1})/2`, mean heights
    :math:`\bar h_i = (h_i + h_{i+1})/2` and
    :math:`\Delta h_i = h_{i+1} - h_i`,

    .. math::

        \tilde h(q) = \sum_{i=0}^{N-2} \Delta x_i\, e^{-iq\bar x_i}
        \left[\bar h_i \operatorname{sinc}(q\Delta x_i/2)
        + \frac{i}{2}\,\Delta h_i \operatorname{sinc}'(q\Delta x_i/2)\right],
        \qquad C^\text{1D}(q) = \frac{1}{L}\,|\tilde h(q)|^2,

    with :math:`\operatorname{sinc}(x) = \sin(x)/x` and
    :math:`L = x_{N-1} - x_0`.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    reliable : bool, optional
        Only return data deemed reliable. Only supported by the 'fft'
        algorithm. (Default: True)
    algorithm : {'fft', 'brute-force'}, optional
        Algorithm to compute the PSD.

        * 'fft': Interpolates the nonuniform line scan on a uniform grid
          and then uses the FFT to compute the PSD. Scales as
          :math:`O(N_\text{int}\log N_\text{int})`, where :math:`N_\text{int}`
          is the number of interpolation points.
        * 'brute-force': Exact Fourier transform of the piecewise linear
          function. Scales as :math:`O(NM)`, where :math:`M` is the number
          of wavevectors.

        (Default: 'fft')
    wavevectors : array_like, optional
        Wavevectors at which to compute the PSD. Required for, and only
        accepted by, the 'brute-force' algorithm. (Default: None)
    nb_interpolate : int, optional
        Number of grid points to put between the closest pair of data
        points. Only used for the 'fft' algorithm. (Default: 5)
    short_cutoff : callable, optional
        Function that determines the short-wavelength cutoff of the returned
        PSD. It is called with the array of distances
        :math:`\Delta x_i` between neighboring data points; wavevectors
        :math:`q < 2\pi/\text{short\_cutoff}(\Delta x_i)` are returned.
        Pass `np.max`, `np.mean` or `np.min` to use the maximum, mean or
        minimum of the distances. If set to None, the full PSD of the
        interpolated data is returned. This cutoff is applied in addition
        to the reliability cutoff. (Default: np.mean)
    window : str, optional
        Name of the window function to apply before computing the PSD.
        Supported are the Hann window ('hann') and no window (None or
        'None'). (Default: no window for periodic line scans, 'hann' window
        for nonperiodic line scans)
    resampling_method : {None, 'bin-average', 'gaussian-process'}, optional
        None returns the PSD on the wavevectors of the FFT (or the requested
        wavevectors), 'bin-average' averages it within bins,
        'gaussian-process' resamples it by Gaussian process regression.
        The 'brute-force' algorithm requires None. (Default: 'bin-average')
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
    wavevectors : np.ndarray or np.ma.MaskedArray
        Wavevectors at which the PSD has been computed. (Units: 1/length)
    psd : np.ndarray or np.ma.MaskedArray
        PSD values. (Units: length³)
    """
    if not self.is_periodic and window is None:
        window = "hann"

    s, = self.physical_sizes
    x, y = self.positions_and_heights()
    if algorithm == 'fft':
        if wavevectors is not None:
            raise ValueError("`wavevectors` can only be used with 'brute-force' algorithm.")
        if self.is_reentrant:
            raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). The power-spectral '
                                     'density cannot be computed for reentrant topographies.')
        wavevectors, psd = self.to_uniform(nb_interpolate=nb_interpolate) \
            .power_spectrum_from_profile(window=window, reliable=reliable, resampling_method=resampling_method,
                                         collocation=collocation, nb_points=nb_points,
                                         nb_points_per_decade=nb_points_per_decade)
    elif algorithm == 'brute-force':
        if reliable:
            raise ValueError("'Brute-force' algorithm for nonuniform line scans does not support reliability analysis.")
        if resampling_method is not None:
            raise ValueError("'Brute-force' algorithm for nonuniform line scans does not support resampling.")
        if wavevectors is None:
            raise ValueError("You need to specify `wavevectors` for the 'brute-force' algorithm.")

        y = apply_window(x, y, window=window)
        L = x[-1] - x[0]
        y_q = np.zeros_like(wavevectors, dtype=complex)
        for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            dx = x2 - x1
            if dx > 0:
                dy = y2 - y1
                x0 = (x1 + x2) / 2
                y0 = (y1 + y2) / 2
                y_q += dx * (y0 * ft_rectangle(wavevectors * dx) + 1j * dy *
                             ft_one_sided_triangle(wavevectors * dx)) * np.exp(
                    -1j * x0 * wavevectors)
            else:
                raise ValueError('Nonuniform data points must be sorted in order of ascending x-values.')
        psd = np.abs(y_q) ** 2 / L
    else:
        raise ValueError("Unknown algorithm '{}' specified.".format(algorithm))

    if short_cutoff is not None:
        mask = wavevectors < 2 * np.pi / short_cutoff(np.diff(x))
        if mask.sum() == 0:
            raise NoReliableDataError('Dataset contains no reliable data.')
        wavevectors = wavevectors[mask]
        psd = psd[mask]
    return wavevectors, psd


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('power_spectrum_from_profile', power_spectrum)
