#
# Copyright 2023 Antoine Sanner
#           2023 Lars Pastewka
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

from ..Container.SurfaceContainer import SurfaceContainer
from ..Generic.Moments import compute_1d_moment, compute_iso_moment


def _bandwidth_intervals_from_profile(self, unit, reliable=True):
    r"""
    Return the wavevector intervals covered by the topographies of this
    container as a pair of sorted arrays (lower bounds, upper bounds).
    This requires a single pass over the container; counting queries can
    then be answered without touching the topographies again (lazy
    containers read the data file on every element access).

    Parameters
    ----------
    self : SurfaceContainer
        Container with topographies and/or line scans.
    unit : str
        Unit of lengths in which the wavevector is defined.
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)

    Returns
    -------
    lower : np.ndarray
        Sorted lower bandwidth bounds of the individual topographies.
    upper : np.ndarray
        Sorted upper bandwidth bounds of the individual topographies.
    """
    lower = []
    upper = []
    for t in self:
        t = t.to_unit(unit)

        qxmax = np.pi / t.pixel_size[0]
        short_cutoff = t.short_reliability_cutoff() if reliable else None
        if short_cutoff is not None:
            qxmax = min(2 * np.pi / short_cutoff, qxmax)

        qxmin = 2 * np.pi / t.physical_sizes[0]
        if qxmax >= qxmin:
            lower += [qxmin]
            upper += [qxmax]
        # else: empty bandwidth interval (reliability cutoff longer than the
        # scan); such a topography contains no wavevector and must not enter
        # the counting arrays, where the subtraction in `_count_in_intervals`
        # would tally it as -1 between its inverted bounds
    return np.sort(lower), np.sort(upper)


def _count_in_intervals(intervals, nb_topographies, qx):
    r"""
    Return the number of bandwidth intervals (from
    `_bandwidth_intervals_from_profile`) that contain each wavevector in
    `qx`.
    """
    lower, upper = intervals
    qx = np.abs(qx)
    # Number of intervals with lower <= qx minus number of intervals with
    # upper < qx
    count = np.searchsorted(lower, qx, side='right') - np.searchsorted(upper, qx, side='left')
    # All topographies have qx == 0 wavevector
    return np.where(qx == 0, nb_topographies, count)


def _bandwidth_count_from_profile(self, qx, unit, reliable=True):
    r"""
    Return number of topographies that include qx in their bandwidth.

    Parameters
    ----------
    self : SurfaceContainer
        Container with topographies and/or line scans.
    qx : np.ndarray of floats
        Wavevectors.
    unit : str
        Unit of lengths in which the wavevector is defined.
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)

    Returns
    -------
    number : np.ndarray
        Number of topographies having qx in their bandwidth.
    """
    return _count_in_intervals(
        _bandwidth_intervals_from_profile(self, unit, reliable), len(self), qx
    )


def integrate_psd_from_profile(self, factor, unit, window=None, reliable=True):
    r"""
    Computes the integral of the 1D PSD weighted by `factor` over all
    topographies of the container.

    The integral is computed by adding up the discrete sums of the individual
    topographies. Each summand is divided by the number of topographies that
    have the respective wavevector within their bandwidth, so that wavevector
    ranges covered by several topographies are averaged rather than counted
    multiple times. For a container with a single topography, this method is
    identical to calling `topography.integrate_psd_from_profile()`.

    Continuum:

    .. math::

        \frac{1}{2 \pi} \int_{-\infty}^\infty dq_x\, f(q_x)\, C^\text{1D}(q_x)

    Discrete:

    .. math::

         \sum_i \frac{1}{L_{x,i}} \sum_{q_x} \frac{f(q_x)\, C^\text{1D}_i(q_x)}{N(q_x)}

    where the index :math:`i` runs over all topographies, :math:`f` is
    `factor` and :math:`N(q_x)` is the number of topographies with
    :math:`2\pi/L_{x,i} \le |q_x| \le \min(\pi/\Delta x_i, 2\pi/\lambda_{c,i})`
    (:math:`\lambda_{c,i}` is the short reliability cutoff, only used if
    `reliable` is True). All topographies contain :math:`q_x=0`. For 2D
    topographies, :math:`C^\text{1D}_i` is averaged over all profiles along
    :math:`x`. Only uniform topographies are supported.

    Parameters
    ----------
    self : SurfaceContainer
        Container with topographies and/or line scans.
    factor : callable
        Function taking as argument the wavevector in the fast scan
        direction qx

            ``func(np.ndarray: qx) -> np.ndarray``

    unit : str
        Length unit. All topographies are converted to this unit before the
        integral is computed; the wavevector passed to `factor` is in units
        of 1/`unit`.
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        (Default: None)
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)

    Returns
    -------
    weighted_integral : float
    """
    integ = 0

    # Precompute the bandwidth intervals in a single pass over the
    # container; the `average` callback below is invoked once per
    # topography, and looping over the container inside it would read
    # every data file N times (lazy containers construct topographies
    # from the file on each element access).
    intervals = _bandwidth_intervals_from_profile(self, unit, reliable)
    nb_topographies = len(self)

    def average(qx):
        count = _count_in_intervals(intervals, nb_topographies, qx)
        return np.where(count > 0, factor(qx) / count, 0)

    for t in self:
        t = t.to_unit(unit)

        integ += t.integrate_psd_from_profile(average,
                                              window=window, reliable=reliable)

    return integ


def ciso_moment(self, order=1, cumulative=False, **kwargs):
    r"""
    Moment of the isotropic PSD obtained from the averaged 1D PSD of the
    container, computed by trapezoidal integration.

    Containers only compute the 1D power spectrum (they may contain line
    scans), so the isotropic PSD is approximated from the averaged 1D PSD as

    .. math::

        C^\text{iso}(q) = \frac{\pi}{q} C^\text{1D}(q)

    and the moment is

    .. math::

        m_\alpha = \frac{1}{2\pi} \int dq\, q^{\alpha+1} C^\text{iso}(q).

    The relation for :math:`C^\text{iso}` holds pointwise only for an
    isotropic, self-affine surface with Hurst exponent :math:`H=1/2`. It is
    used for all surfaces, because measured surfaces are not self-affine and
    have no well-defined Hurst exponent. For any isotropic surface, the
    returned moment is :math:`\kappa_\alpha` times the moment of the true
    isotropic PSD, with

    .. math::

        \kappa_\alpha = \frac{\sqrt{\pi}\,\Gamma\left(\frac{\alpha+1}{2}\right)}
        {2\,\Gamma\left(\frac{\alpha}{2}+1\right)},

    i.e. :math:`\kappa_0 = \pi/2`, :math:`\kappa_1 = 1` and
    :math:`\kappa_2 = \pi/4`. The moment of order one (the variance of the
    half derivative) is therefore exact for isotropic surfaces, independent
    of :math:`H`.

    Masked (empty) bins of the averaged PSD are dropped before integration.

    Parameters
    ----------
    self : SurfaceContainer
        Container with topographies and/or line scans.
    order : float, optional
        Order :math:`\alpha` of the moment. (Default: 1)
    cumulative : bool, optional
        If True, return the cumulative integral evaluated at each wavevector
        instead of the full integral. (Default: False)
    **kwargs
        Further parameters (e.g. `unit`, `nb_points_per_decade`,
        `reliable`) are passed to `power_spectrum`.

    Returns
    -------
    moment : float or np.ndarray
    """
    power_spectrum = self.power_spectrum(**kwargs)  #
    q, C1D = power_spectrum
    # Filter out masked/invalid values (handle both masked arrays and NaN)
    if np.ma.isMaskedArray(C1D):
        valid = ~C1D.mask
        q = np.asarray(q[valid])
        C1D = np.asarray(C1D[valid])
    else:
        valid = ~np.isnan(C1D)
        q = q[valid]
        C1D = C1D[valid]
    Ciso = C1D * np.pi / q
    return compute_iso_moment(q, Ciso, order, cumulative=cumulative)


def c1d_moment(self, order=1, cumulative=False, **kwargs):
    r"""
    Moment of the averaged 1D PSD of the container, computed by trapezoidal
    integration,

    .. math::

        m_\alpha = \frac{1}{\pi} \int_0^\infty dq\, q^\alpha C^\text{1D}(q).

    Masked (empty) bins of the averaged PSD are dropped before integration.

    Parameters
    ----------
    self : SurfaceContainer
        Container with topographies and/or line scans.
    order : float, optional
        Order :math:`\alpha` of the moment. (Default: 1)
    cumulative : bool, optional
        If True, return the cumulative integral evaluated at each wavevector
        instead of the full integral. (Default: False)
    **kwargs
        Further parameters (e.g. `unit`, `nb_points_per_decade`,
        `reliable`) are passed to `power_spectrum`.

    Returns
    -------
    moment : float or np.ndarray
    """
    power_spectrum = self.power_spectrum(**kwargs)  #
    q, C1D = power_spectrum
    # Filter out masked/invalid values (handle both masked arrays and NaN)
    if np.ma.isMaskedArray(C1D):
        valid = ~C1D.mask
        q = np.asarray(q[valid])
        C1D = np.asarray(C1D[valid])
    else:
        valid = ~np.isnan(C1D)
        q = q[valid]
        C1D = C1D[valid]
    return compute_1d_moment(q, C1D, order, cumulative=cumulative)


SurfaceContainer.register_function("ciso_moment", ciso_moment)
SurfaceContainer.register_function("c1d_moment", c1d_moment)
SurfaceContainer.register_function("integrate_psd_from_profile", integrate_psd_from_profile)
