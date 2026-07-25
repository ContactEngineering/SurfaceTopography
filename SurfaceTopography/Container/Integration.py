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

    Parameters:
    -----------
    self : SurfaceContainer
        Collection of Height containers
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

    Parameters:
    -----------
    self : SurfaceContainer
        Collection of Height containers
    qx: np.ndarrau of floats
        wavevector
    unit : str
        Unit of lengths in which the wavevector is defined.
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)

    Returns
    -------
    number: np.ndarray
        number of topographies having qx in their bandwidth
    """
    return _count_in_intervals(
        _bandwidth_intervals_from_profile(self, unit, reliable), len(self), qx
    )


def integrate_psd_from_profile(self, factor, unit, window=None, reliable=True):
    r"""

    Computes the integral of the 1D PSD weighted by "factor"

    The integral is computed by adding up the discrete sum for each topography.

    The summand for each topography is weighted by the number of topographies having this wavevector in their bandwidth

    Hence for a container containing only one topogaphy,
    this method is identical to calling `topography.integrate_psd_from_profile()`

    Continuum:

    .. math::

        \frac{1}{2 \pi} \int_{-\infty}^\infty dq_x factor(q_x) C^{1D}(q_x)

    Discrete

    .. math::

         m_\alpha = \sum_i \frac{1}{L_{x, i}} \sum_{q_x} \frac{factor(q_x) C^{1D}_{i}(q_x)}{N(q_x)}


    Where the index :math:`i` runs over all topographies and :math:`N(q_x)` is the number
    of topographies having the wavevector `q_x` in their bandwidth


    Parameters:
    -----------
    self : SurfaceContainer
        Collection of Height containers
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
    trapz integration of the moments of the averaged isotropic PSD.

    Containers only implement the 1D power-spectrum, so that we use the approximation mapping
    the 1d PSD to the isotropic PSD

    .. math::

        C^\mathrm{iso} = \frac{\pi}{q} C^\mathrm{1D}


    Parameters
    ----------
    self : SurfaceContainer or HeightContainer
        Container with height information or collection of height containsers
    order : float
        order of moment of the PDF to compute
    Further parameters are passed to the power_spectrum method, see the documentation of the corresponding method

    Returns
    -------
    float
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
    """
    trapz integration of

    Parameters
    ----------
    self : SurfaceContainer or HeightContainer
        Container with height information or collection of height containsers
    order : float
        order of moment of the PDF to compute
    Further parameters are passed to the power_spectrum method, see the documentation of the corresponding method

    Returns
    -------
    float
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
