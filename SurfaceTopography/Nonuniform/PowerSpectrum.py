#
# Copyright 2018-2021 Lars Pastewka
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

from ..HeightContainer import NonuniformLineScanInterface


def sinc(x):
    """sinc function"""
    return np.sinc(x / np.pi)


def dsinc(x):
    """Derivative of the sinc function"""
    tol = 1e-6
    x = np.asarray(x)
    small_values = np.abs(x) < tol
    if small_values.sum() > 0:
        ret = np.zeros_like(x)

        # For small values, use the Taylor series expansion
        # (accurate to O(x^^))
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
        length = x.max() - x.min()
        return (2 / 3) ** (1 / 2) * (1 - np.cos(2 * np.pi * x / length)) * y
    elif window is None or window == 'None':
        return y
    else:
        raise ValueError('Unknown window {}'.format(window))


def power_spectrum(line_scan, algorithm='fft', wavevectors=None,
                   nb_interpolate=5, short_cutoff=np.mean, window=None):
    r"""
    Compute power-spectral density (PSD) for a nonuniform topography. The
    topography is assumed to be given by a series of points connected by
    straight lines.

    Parameters
    ----------
    line_scan : :obj:`NonuniformLineScan`
        Container storing the nonuniform line scan.
    algorithms : str
        Algorithm to compute autocorrelation.
        * 'fft': Interpolates the nonuniform line scan on a grid and then uses
        the FFT to compute the autocorrelation. Scales O(N log N)
        * 'brute-force': Brute-force computation using between line segements.
        Scale O(N^2 M) where M is the number of distance point.
        (Default: 'fft')
    wavevectors : array, optional
        Wavevectors at which to compute the PSD. If omitted, wavevectors are
        equally spaced with a spacing that corresponds to :math:`2\pi/\lambda`
        where :math:`\lambda` is the shortest distance between two points in
        the `x`-array. (Default: None)
    nb_interpolate : int, optional
        Number of grid points to put between closest points on surface. Only
        used for 'fft' algorithm. (Default: 5)
    short_cutoff : function, optional
        Function that determines how the short cutoff for the returned PSD is
        computed. If set to None, the full PSD of the interpolated data is
        returned. If the user passes a function, that function is called with
        an array that contains the distances between the points of the uniform
        line scan. Pass `np.max`, `np.mean` or `np.min` to use the maximum,
        mean or minimum of that distance as the cutoff.
    window : str, optional
        Name of the window function to apply before computing the PSD.
        Presently only supports Hann window ('hann') or no window (None or
        'None').
        Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies

    Returns
    -------
    q : array
        Wavevector array at which the PSD has been computed.
    C : array
        PSD values.
    """
    if not line_scan.is_periodic and window is None:
        window = "hann"

    s, = line_scan.physical_sizes
    x, y = line_scan.positions_and_heights()
    if algorithm == 'fft':
        if wavevectors is not None:
            raise ValueError("`wavevectors` can only be used with "
                             "'brute-force' algorithm.")
        min_dist = np.min(np.diff(x))
        if min_dist <= 0:
            raise RuntimeError('Positions not sorted.')
        wavevectors, psd = line_scan.to_uniform(nb_interpolate=nb_interpolate) \
            .power_spectrum_from_profile(window=window)
    elif algorithm == 'brute-force':
        y = apply_window(x, y, window=window)
        L = x[-1] - x[0]
        if wavevectors is None:
            wavevectors = 2 * np.pi * np.arange(int(L / np.diff(x).min())) / L
        y_q = np.zeros_like(wavevectors, dtype=np.complex128)
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
                raise ValueError('Nonuniform data points must be sorted in '
                                 'order of ascending x-values.')
        psd = np.abs(y_q) ** 2 / L
    else:
        raise ValueError("Unknown algorithm '{}' specified."
                         .format(algorithm))

    if short_cutoff is not None:
        mask = wavevectors < 2 * np.pi / short_cutoff(np.diff(x))
        wavevectors = wavevectors[mask]
        psd = psd[mask]
    return wavevectors, psd


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('power_spectrum_from_profile',
                                              power_spectrum)
