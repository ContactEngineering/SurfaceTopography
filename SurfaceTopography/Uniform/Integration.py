#
# Copyright 2023 Antoine Sanner
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
import inspect

import numpy as np

from ..Exceptions import NoReliableDataError, UndefinedDataError
from ..HeightContainer import UniformTopographyInterface


def integrate_psd(self, factor=lambda q: 1, window=None, reliable=True, ):
    r"""
    Integral of the PSD weighted by a function of the wavevector,

    .. math::

        I = \frac{1}{(2\pi)^2} \int d^2q\, f(\mathbf q)\, C^\text{2D}(\mathbf q)
        \approx \frac{1}{L_x L_y} \sum_{\mathbf q} f(\mathbf q)\,
        C^\text{2D}(\mathbf q),

    where the sum runs over all wavevectors of the 2D FFT (the full
    rectangular Brillouin zone, including :math:`\mathbf q = 0`). For line
    scans, the 1D expression
    :math:`I = \frac{1}{L}\sum_q f(|q|)\, C^\text{1D}(q)` is evaluated. The
    PSD is not resampled, which avoids the error of integrating a binned
    PSD. By Parseval's theorem, :math:`f = 1` yields
    :math:`\langle h^2\rangle`.

    Parameters
    ----------
    factor : callable, optional
        Weight function :math:`f`. If it accepts a single argument, it is
        called with the norm :math:`|\mathbf q|` of the wavevectors,
        ``factor(q: np.ndarray) -> np.ndarray``. If it accepts two
        arguments (topography maps only), it is called with the components,
        ``factor(qx: np.ndarray, qy: np.ndarray) -> np.ndarray``.
        (Default: ``lambda q: 1``)
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only include wavevectors :math:`|\mathbf q| < 2\pi/\lambda_c`,
        where :math:`\lambda_c` is the short reliability cutoff.
        (Default: True)

    Returns
    -------
    weighted_integral : float
        Value of the integral :math:`I`.
    """
    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Power-spectrum cannot be '
                                 'computed for topographies with missing data points.')

    h = self.window(window).heights()

    fourier_topography = np.prod(self.physical_sizes) / np.prod(self.nb_grid_pts) * np.fft.fftn(h)

    # This is the raw power spectral density
    C_raw = (np.abs(fourier_topography) ** 2) / np.prod(self.physical_sizes)
    q = self.wavevectors_norm2()

    # Only keep reliable data
    short_cutoff = self.short_reliability_cutoff() if reliable else None
    if short_cutoff is not None:
        mask = q < 2 * np.pi / short_cutoff
        if mask.sum() <= 1:  # There is always q=0 in there
            raise NoReliableDataError('Dataset contains no reliable data.')
        C_raw = C_raw * mask

    # Dispatch on the number of arguments the factor accepts rather than
    # catching TypeError, which would silently mask genuine TypeErrors
    # raised inside a user-supplied factor function
    try:
        nb_factor_args = len(inspect.signature(factor).parameters)
    except (TypeError, ValueError):
        nb_factor_args = 1
    if self.dim == 2 and nb_factor_args >= 2:
        qvec = self.fftfreq()
        return np.sum(C_raw * factor(*qvec)) / np.prod(self.physical_sizes)
    else:
        return np.sum(C_raw * factor(q)) / np.prod(self.physical_sizes)


def integrate_psd_from_profile(self, factor=lambda qx: 1, window=None, reliable=True, ):
    r"""
    Integral of the 1D PSD of the profiles along :math:`x`, weighted by a
    function of the wavevector,

    .. math::

        I = \frac{1}{2\pi} \int_{-\infty}^{\infty} dq_x\, f(q_x)\,
        C^\text{1D}(q_x)
        \approx \frac{1}{L_x} \sum_{q_x} f(q_x)\, C^\text{1D}(q_x),

    where the sum runs over all (positive and negative) wavevectors of the
    FFT. For topography maps, the result is averaged over all profiles.

    Parameters
    ----------
    factor : callable, optional
        Weight function :math:`f`, called with the (signed) wavevectors
        :math:`q_x`, ``factor(qx: np.ndarray) -> np.ndarray``.
        (Default: ``lambda qx: 1``)
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only include wavevectors :math:`|\mathbf q| < 2\pi/\lambda_c`,
        where :math:`\lambda_c` is the short reliability cutoff.
        (Default: True)

    Returns
    -------
    weighted_integral : float
        Value of the integral :math:`I`.
    """
    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Power-spectrum cannot be '
                                 'computed for topographies with missing data points.')

    h = self.window(window).heights()

    sx = self.physical_sizes[0]
    nx = self.nb_grid_pts[0]

    fourier_topography = sx / nx * np.fft.fft(h, axis=0)

    # This is the raw power spectral density
    C_raw = (np.abs(fourier_topography) ** 2) / sx

    # Short reliability cutoff on the 1D PSDs
    if self.dim == 2:
        qx, qy = self.fftfreq()
        # note that qy is meaningless here since we didn't fourier transform along the y direction.
    else:
        qx = self.fftfreq()

    # Only keep reliable data
    short_cutoff = self.short_reliability_cutoff() if reliable else None
    if short_cutoff is not None:
        mask = abs(qx) < 2 * np.pi / short_cutoff
        if mask.sum() <= 1:  # There is always q=0 in there
            raise NoReliableDataError('Dataset contains no reliable data.')
        C_raw = C_raw * mask

    if self.dim == 2:
        return np.sum(C_raw * factor(qx)) / sx / self.nb_grid_pts[1]
    else:
        return np.sum(C_raw * factor(qx)) / sx


def moment_power_spectrum(self, order=0, window=None, reliable=True, ):
    r"""
    Moment of order :math:`\alpha` of the PSD. For topography maps,

    .. math::

        m_\alpha = \frac{1}{(2\pi)^2} \int d^2q\, |\mathbf q|^\alpha\,
        C^\text{2D}(\mathbf q)
        \approx \frac{1}{L_x L_y} \sum_{\mathbf q} |\mathbf q|^\alpha\,
        C^\text{2D}(\mathbf q),

    for line scans,

    .. math::

        m_\alpha = \frac{1}{2\pi} \int_{-\infty}^\infty dq\, |q|^\alpha\,
        C^\text{1D}(q)
        \approx \frac{1}{L} \sum_q |q|^\alpha\, C^\text{1D}(q).

    :math:`m_0 = \langle h^2\rangle`, :math:`m_2` is the mean square slope
    (gradient for maps) and :math:`m_4` the mean square Laplacian (second
    derivative for line scans), all evaluated with spectral derivatives.
    See :func:`integrate_psd`.

    Parameters
    ----------
    order : float, optional
        Order :math:`\alpha` of the moment. (Default: 0)
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only include wavevectors :math:`|\mathbf q| < 2\pi/\lambda_c`,
        where :math:`\lambda_c` is the short reliability cutoff.
        (Default: True)

    Returns
    -------
    moment : float
        Moment :math:`m_\alpha`.
    """
    return integrate_psd(self, lambda q: q ** order, window=window, reliable=reliable, )


def variance_half_derivative(self, window=None, reliable=True):
    r"""
    Variance of the half derivative of the heights,

    .. math::

        \langle |\nabla^{1/2} h|^2 \rangle
        = \frac{1}{(2 \pi)^2} \int d^2q\, |\mathbf q|\, C^\text{2D}(\mathbf q),

    computed from the full power spectral density, see
    :func:`integrate_psd`. For line scans, the surface is assumed to be
    invariant perpendicular to the scan, which gives
    :math:`\frac{1}{2\pi}\int dq\, |q|\, C^\text{1D}(q)`.

    Parameters
    ----------
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only include data deemed reliable. (Default: True)

    Returns
    -------
    variance : float
        Variance of the half derivative.
    """
    return self.integrate_psd(lambda q: q, window=window, reliable=reliable)


def elastic_energy(self, contact_modulus, window=None, reliable=True):
    r"""
    Elastic energy per unit area stored in an elastic half-space that is
    pressed into full contact with the topography,

    .. math::

        \frac{U}{A} = \frac{E^*}{4} \langle |\nabla^{1/2} h|^2 \rangle,

    where :math:`E^*` is the contact modulus. See
    :func:`variance_half_derivative`.

    Parameters
    ----------
    contact_modulus : float
        Contact modulus :math:`E^*`.
    window : str, optional
        Window for eliminating edge effects, see scipy.signal.get_window.
        (Default: no window for periodic topographies, 'hann' window for
        nonperiodic topographies)
    reliable : bool, optional
        Only include data deemed reliable. (Default: True)

    Returns
    -------
    energy : float
        Elastic energy per unit area. (Units: those of `contact_modulus`
        times length)
    """
    return contact_modulus / 4 * variance_half_derivative(self, window=window, reliable=reliable)


UniformTopographyInterface.register_function('moment_power_spectrum', moment_power_spectrum)
UniformTopographyInterface.register_function('integrate_psd', integrate_psd)
UniformTopographyInterface.register_function('integrate_psd_from_profile', integrate_psd_from_profile)
UniformTopographyInterface.register_function('variance_half_derivative', variance_half_derivative)
UniformTopographyInterface.register_function('elastic_energy', elastic_energy)
