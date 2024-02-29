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
import numpy as np

from ..Exceptions import NoReliableDataError, UndefinedDataError
from ..HeightContainer import UniformTopographyInterface


def integrate_psd(self, factor=lambda q: 1, window=None, reliable=True, ):
    r"""

    .. math::

        m_\alpha = \frac{1}{(2 \pi)^2} \int_{-\infty}^\infty dq_x dq_y factor(q_x, q_y) C^{2D}(q_x, q_y)

    Discrete

          m_\alpha = \frac{1}{L_x L_y} \sum_{q_x, q_y} factor(q_x, q_y) C^{2D}_{q_x, q_y}

    Parameters:
    -----------
    factor: function
        Function taking as argument the 2 norm of the wavevector (if it accepts only one argument)

            ``func(np.ndarray: q) -> np.ndarray``

        or the two components of the wavevector qx and qy

            ``func(np.ndarray: qx, np.ndarray:qy) -> np.ndarray``

    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: None)
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    Returns:
    --------
        weighted_integral: float
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

    qvec = self.fftfreq()
    try:
        return np.sum(C_raw * factor(*qvec)) / np.prod(self.physical_sizes)
    except TypeError:
        return np.sum(C_raw * factor(q)) / np.prod(self.physical_sizes)


def integrate_psd_from_profile(self, factor=lambda qx: 1, window=None, reliable=True, ):
    r"""

    Computes the integral of the 1D PSD weighted by "factor"

    Continuum:

    .. math::

        \frac{1}{2 \pi} \int_0^\infty dq_x factor(q_x) C^{1D}(q_x)

    Discrete

    .. math::

         m_\alpha = \frac{1}{L_x} \sum_{q_x} factor(q_x) C^{1D}_{q_x}


    Parameters:
    -----------
        factor: callable
            Function taking as argument the wavevector in the fast scan direction qx

                ``func(np.ndarray: qx) -> np.ndarray``
        window : str, optional
            Window for eliminating edge effect. See scipy.signal.get_window.
            (Default: None)
        reliable : bool, optional
            Only return data deemed reliable. (Default: True)

    Returns:
    --------
        weighted_integral: float
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

    Computes

    For 2D Topographies, continuum:

    .. math::

        m_\alpha = \frac{1}{(2 \pi)^2} \int_{-\infty}^\infty dq_x dq_y |q|^{\alpha} C^{2D}(q_x, q_y)

    Discrete

          m_\alpha = \frac{1}{L_x L_y} \sum_{q_x, q_y} |q|^{\alpha} C^{2D}_{q_x, q_y}

    For line scans, continuum:

    .. math::

        \frac{1}{2 \pi} \int_0^\infty dq_x |q|^{\alpha} C^{1D}(q_x)

    Discrete

    .. math::

         m_\alpha = \frac{1}{L_x} \sum_{q_x} |q|^{\alpha} C^{1D}_{q_x}

    Parameters:
    -----------
        order: int
            order of the moment to compute
        window : str, optional
            Window for eliminating edge effect. See scipy.signal.get_window.
            (Default: None)
        reliable : bool, optional
            Only return data deemed reliable. (Default: True)

    """
    return integrate_psd(self, lambda q: q ** order, window=window, reliable=reliable, )


UniformTopographyInterface.register_function('moment_power_spectrum', moment_power_spectrum)
UniformTopographyInterface.register_function('integrate_psd', integrate_psd)
UniformTopographyInterface.register_function('integrate_psd_from_profile', integrate_psd_from_profile)
