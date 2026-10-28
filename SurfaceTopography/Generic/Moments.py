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
"""

Moments of a function sampled at discrete points, e.g. of a resampled PSD.

The example below computes the (unbinned) 1D PSD of a surface topography and
integrates it to obtain the root mean square of the heights. (The zeroth moment
contains the :math:`q = 0` term, i.e. the mean of the heights is not
subtracted.)

>>> import numpy as np
>>> from SurfaceTopography.Generation import fourier_synthesis
>>> from SurfaceTopography.Generic.Moments import compute_1d_moment
>>> t = fourier_synthesis((256, 256), (256, 256), hurst=0.8, rms_height=1, short_cutoff=4, long_cutoff=64)
>>> hrms = np.sqrt(np.mean(t.heights() ** 2))
>>> q, C = t.power_spectrum_from_profile(reliable=False, resampling_method=None)
>>> hrms_from_psd = np.sqrt(compute_1d_moment(q, C, order=0))
>>> assert abs(hrms_from_psd / hrms - 1) < 1e-6

Averaging the PSD within bins before integrating introduces errors; summing
the unbinned spectrum is more accurate, see the methods `integrate_psd` and
`moment_power_spectrum` of the topographies and `integrate_psd` of the
containers.

"""
import numpy as np
import scipy.integrate


def compute_1d_moment(x, y, order=1, cumulative=False):
    r"""

    Computes the moment of order :math:`\alpha`

    .. math::

        m_\alpha = \frac{1}{\pi} \int dx\, y x^{\alpha}

    using trapezoidal interpolation of the integrand. (The prefactor
    :math:`1/\pi` corresponds to the moment of a two-sided function
    :math:`y(|x|)` with the :math:`1/(2\pi)` Fourier-transform convention
    used for power spectra in this package.)

    Parameters
    ----------
    x : np.ndarray
        Sample points.
    y : np.ndarray
        Values of the function sampled at `x`.
    order : float, optional
        Order of the moment. (Default: 1)
    cumulative : bool, optional
        If True, return the cumulative value of the integral from `x[0]` to
        each `x[i]` (uses `scipy.integrate.cumulative_trapezoid`).
        (Default: False)

    Returns
    -------
    integral : float or np.ndarray
        Value of the integral if `cumulative` is False; otherwise array of
        length `len(x) - 1` with the cumulative integral up to `x[1:]`.

    """
    power = order
    integ = (
        scipy.integrate.cumulative_trapezoid
        if cumulative
        else scipy.integrate.trapezoid
    )
    return integ(y * x**power, x) / np.pi


def compute_iso_moment(x, y, order=1, cumulative=False):
    r"""
    Computes the moment of order :math:`\alpha` of the 2D isotropic function :math:`y_{2D}(x)`

    :math:`y` is the 1D representation of the isotropic function :math:`y_{2D}(x_1, x_2) = y(|\vec x|)`

    .. math::

        m_\alpha = \frac{1}{(2\pi)^2} \int dx_1 dx_2\, y_{2D}(x_1, x_2) |\vec x|^{\alpha}
                 = \frac{1}{2\pi} \int dx\, y(x) x^{\alpha + 1}

    using trapezoidal interpolation of the integrand. (The prefactor
    corresponds to the :math:`1/(2\pi)^2` Fourier-transform convention used
    for power spectra in this package.)

    Parameters
    ----------
    x : np.ndarray
        Sample points.
    y : np.ndarray
        Values of the function sampled at `x`.
    order : float, optional
        Order of the moment. (Default: 1)
    cumulative : bool, optional
        If True, return the cumulative value of the integral from `x[0]` to
        each `x[i]` (uses `scipy.integrate.cumulative_trapezoid`).
        (Default: False)

    Returns
    -------
    integral : float or np.ndarray
        Value of the integral if `cumulative` is False; otherwise array of
        length `len(x) - 1` with the cumulative integral up to `x[1:]`.
    """
    power = order + 1
    integ = (
        scipy.integrate.cumulative_trapezoid
        if cumulative
        else scipy.integrate.trapezoid
    )
    return integ(y * x**power / (2 * np.pi), x)
