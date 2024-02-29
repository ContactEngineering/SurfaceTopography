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
    qx = np.abs(qx)
    factor = np.zeros_like(qx)
    for t in self:
        t = t.to_unit(unit)

        qxmax = np.pi / t.pixel_size[0]
        short_cutoff = t.short_reliability_cutoff() if reliable else None
        if short_cutoff is not None:
            qxmax = min(2 * np.pi / short_cutoff, qxmax)

        factor += np.logical_and(
            qx >= 2 * np.pi / t.physical_sizes[0],
            qx <= qxmax,
        )
    # All topographies have qx == 0 wavevector
    factor = np.where(qx == 0, len(self), factor)
    # TODO: This will be very slow on Topobank
    return factor


def integrate_psd_from_profile(self, factor, unit, window=None, reliable=True):
    r"""

    Computes the integral of the 1D PSD weighted by "factor"

    The integral is computed by adding up the discrete sum for each topography.

    The summand for each topography is weighted by the number of topographies having this wavevector in their bandwidth

    Hence for a container containing only one topogaphy,
    this method is identical to calling `topography.integrate_psd_from_profile()`

    Continuum:

    .. math::

        \frac{1}{2 \pi} \int_0^\infty dq_x factor(q_x) C^{1D}(q_x)

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

    # TODO: faster: precompute _bandwidth_count_from_profile (defining a piecewise constant function, )
    # a table with bins and number of topographies inside the bins
    # this will make us only loop 2N times through the topographies instead of N^2
    # with N the total number of topographies

    def average(qx):
        count = _bandwidth_count_from_profile(self, qx, unit, reliable)
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
    q = q[~np.isnan(C1D)]
    C1D = C1D[~np.isnan(C1D)]
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
    q = q[~np.isnan(C1D)]
    C1D = C1D[~np.isnan(C1D)]
    return compute_1d_moment(q, C1D, order, cumulative=cumulative)


SurfaceContainer.register_function("ciso_moment", ciso_moment)
SurfaceContainer.register_function("c1d_moment", c1d_moment)
SurfaceContainer.register_function("integrate_psd_from_profile", integrate_psd_from_profile)
