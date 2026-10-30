#
# Copyright 2023 Lars Pastewka
#           2022-2023 Antoine Sanner
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
Variance of the half derivative of the heights, estimated from the
height-difference autocorrelation function.
"""

import numpy as np
import scipy.integrate

from ..Container.SurfaceContainer import SurfaceContainer
from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..Support import doi


def _drop_undefined(x, y):
    """Remove entries where `x` or `y` is masked (empty bins) or NaN."""
    x = np.ma.filled(np.ma.asarray(x, dtype=float), np.nan)
    y = np.ma.filled(np.ma.asarray(y, dtype=float), np.nan)
    defined = np.isfinite(x) & np.isfinite(y)
    return x[defined], y[defined]


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_autocorrelation_from_area(topography, **kwargs):
    r"""
    Estimate the variance of the half derivative,
    :math:`\langle|\nabla^{1/2}h|^2\rangle`, from the radially averaged
    autocorrelation function :math:`A(r)`:

    .. math::

        \langle|\nabla^{1/2}h|^2\rangle = \int_0^\infty dr\, \frac{A(r)}{r^2}

    This identity follows from :math:`\int_0^\infty dr\,[1-J_0(qr)]/r^2 = q`.
    See :func:`variance_half_derivative_via_scale_dependent_slope` for the
    discretization.

    Parameters
    ----------
    topography : :obj:`Topography`
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to
        `autocorrelation_from_area`

    Returns
    -------
    variance_half_derivative : float
        Variance of the half derivative. (Units: length)
    """
    r, s = topography.scale_dependent_slope_from_area(**kwargs)
    return variance_half_derivative_via_scale_dependent_slope(r, s)


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_autocorrelation_from_profile(topography, **kwargs):
    r"""
    Estimate the variance of the half derivative from the profile
    autocorrelation function :math:`A(r)` (along x):

    .. math::

        \int_0^\infty dr\, \frac{A(r)}{r^2} = \frac{\pi}{2}\int \frac{d^2q}{(2\pi)^2}\, |q_x|\, C^\text{2D}(\mathbf{q}),

    which follows from :math:`\int_0^\infty dr\,[1-\cos(q_x r)]/r^2 = \pi|q_x|/2`.
    For an isotropic surface, the angular average of :math:`|q_x|` is
    :math:`2q/\pi`, so this estimates the areal
    :math:`\langle|\nabla^{1/2}h|^2\rangle = \int d^2q/(2\pi)^2\, q\, C^\text{2D}(q)`.
    For a line scan, the result is :math:`\pi/2` times
    :math:`\int dq/(2\pi)\,|q|\,C^\text{1D}(q)`, i.e.
    :math:`\frac{1}{2}\int_0^\infty dq\, q\, C^\text{1D}(q)`. This estimates
    the half derivative of the map from which the line scan was taken,
    assuming that this map is isotropic. For any isotropic surface it equals
    the areal value exactly, independent of the Hurst exponent, and it
    equals the moment of order one of the isotropic PSD estimated as
    :math:`C^\text{iso}(q) = \pi C^\text{1D}(q)/q`. The autocorrelation
    function needs no window, unlike the PSD of nonperiodic data.
    See :func:`variance_half_derivative_via_scale_dependent_slope` for the
    discretization.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan` or :obj:`NonuniformLineScan` or :obj:`SurfaceContainer`
        Topography, line scan or container of these
    **kwargs : dict
        Additional keyword parameters are passed on to
        `scale_dependent_slope_from_profile`, i.e. to
        `autocorrelation_from_profile`

    Returns
    -------
    variance_half_derivative : float
        Variance of the half derivative. (Units: length)
    """
    r, s = topography.scale_dependent_slope_from_profile(**kwargs)
    return variance_half_derivative_via_scale_dependent_slope(r, s)


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_scale_dependent_slope(r, slope):
    r"""
    Integrate the squared scale-dependent slope :math:`s(r)` to obtain

    .. math::

        \frac{1}{2}\int_0^{r_\text{max}} dr\, s^2(r)
        = \int_0^{r_\text{max}} dr\, \frac{A(r)}{r^2}
        \approx \frac{1}{2}\left[r_0 s^2(r_0) + \int_{r_0}^{r_\text{max}} dr\, s^2(r)\right],

    where :math:`s^2(r) = 2A(r)/r^2`. Below the smallest distance
    :math:`r_0`, the slope is assumed constant; the remaining integral is
    evaluated with the trapezoidal rule. Masked (empty bins) and NaN entries
    are dropped.

    Parameters
    ----------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    Returns
    -------
    variance_half_derivative : float
        Variance of the half derivative. (Units: length)
    """

    r, slope = _drop_undefined(r, slope)
    return 0.5 * (r[0] * slope[0] ** 2 + scipy.integrate.trapezoid(slope**2, x=r))


def container_variance_half_derivative_from_autocorrelation(
    container, scale_dependent=False, **kwargs
):
    r"""
    Variance of the half derivative of all topographies in a container,
    computed from the container-averaged autocorrelation function
    :math:`A(r)`:

    .. math::

        V(r) = \frac{A(r_0)}{r_0} + \int_{r_0}^{r} dr'\, \frac{A(r')}{r'^2}

    The first term assumes a constant slope below the smallest distance
    :math:`r_0`; the integral is evaluated with the trapezoidal rule. The
    scale dependence is introduced by integrating the autocorrelation function
    only up to the distance :math:`r`. This is the scale-dependent elastic
    energy of Wang and Müser (their Eq. 44), up to the prefactor
    :math:`E^*/4`.

    .. TODO: full reference for Wang and Müser?

    Parameters
    ----------
    container : :obj:`SurfaceContainer`
        Container of topographies.
    scale_dependent : bool, optional
        Return the cumulative integral :math:`V(r)` instead of its value at
        the largest distance. (Default: False)
    **kwargs : dict
        Additional keyword parameters are passed on to
        `container.autocorrelation`.

    Returns
    -------
    r : np.ndarray
        Distances, only if `scale_dependent` is true. (Units: length)
    variance_half_derivative : float or np.ndarray
        Variance of the half derivative. (Units: length)
    """

    r, acf = container.autocorrelation(**kwargs)
    r, acf = _drop_undefined(r, acf)

    # we assume continuity below the smallest distance, i.e. that the slope is constant
    small_scale_contrib = acf[0] / r[0]
    if scale_dependent:
        return r, small_scale_contrib + np.concatenate(
            [[0], scipy.integrate.cumulative_trapezoid(acf / r**2, x=r)]
        )
    else:
        return small_scale_contrib + scipy.integrate.trapezoid(acf / r**2, x=r)


SurfaceContainer.register_function(
    "variance_half_derivative_from_autocorrelation",
    container_variance_half_derivative_from_autocorrelation,
)
# Register analysis functions from this module
UniformTopographyInterface.register_function(
    "variance_half_derivative_via_autocorrelation_from_profile",
    variance_half_derivative_via_autocorrelation_from_profile,
)
NonuniformLineScanInterface.register_function(
    "variance_half_derivative_via_autocorrelation_from_profile",
    variance_half_derivative_via_autocorrelation_from_profile,
)
UniformTopographyInterface.register_function(
    "variance_half_derivative_via_autocorrelation_from_area",
    variance_half_derivative_via_autocorrelation_from_area,
)
SurfaceContainer.register_function(
    "variance_half_derivative_via_autocorrelation_from_profile",
    variance_half_derivative_via_autocorrelation_from_profile,
)
