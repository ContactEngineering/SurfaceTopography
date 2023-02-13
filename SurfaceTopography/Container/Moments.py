from ..Generic.Moments import (compute_iso_moment,
                               compute_1d_moment, )
import numpy as np
from ..Container import SurfaceContainer


def _bandwidth_count_from_profile(self, qx, unit, reliable=True):
    """
    Return number of topographies that include qx in their bandwidth.
    """
    qx = np.abs(qx)
    factor = np.zeros_like(qx)
    for t in self._topographies:
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
    factor = np.where(qx == 0, len(self._topographies), factor)
    return factor


def integrate_psd_from_profile(self, factor, unit, window=None, reliable=True):
    integ = 0

    def average(qx):
        count = _bandwidth_count_from_profile(self, qx, unit, reliable)
        return np.where(count > 0, factor(qx) / count, 0)

    for t in self._topographies:
        t = t.to_unit(unit)

        integ += t.integrate_psd_from_profile(average,
                                              window=window, reliable=reliable)

    return integ


def ciso_moment(c, order=1, cumulative=False, **kwargs):
    """

    Computes

    Containers only implement the 1D power-spectrum, so that we use the approximation



    Parameters
    ----------
    c : container or topography
    order : float
        order of moment of the PDF to compute
    Further parameters are passed to the power_spectrum method, see the documentation of the corresponding method

    Returns
    -------
    float
    """
    power_spectrum = c.power_spectrum(**kwargs)  #
    q, C1D = power_spectrum
    q = q[~np.isnan(C1D)]
    C1D = C1D[~np.isnan(C1D)]
    Ciso = C1D * np.pi / q
    return compute_iso_moment(q, Ciso, order, cumulative=cumulative)


def c1d_moment(c, order=1, cumulative=False, **kwargs):
    """
    Parameters
    ----------
    c : container or topography
    order : float
        order of moment of the PDF to compute
    Further parameters are passed to the power_spectrum method, see the documentation of the corresponding method

    Returns
    -------
    float
    """
    power_spectrum = c.power_spectrum(**kwargs)  #
    q, C1D = power_spectrum
    q = q[~np.isnan(C1D)]
    C1D = C1D[~np.isnan(C1D)]
    return compute_1d_moment(q, C1D, order, cumulative=cumulative)


SurfaceContainer.register_function("ciso_moment", ciso_moment)
SurfaceContainer.register_function("c1d_moment", c1d_moment)
SurfaceContainer.register_function("integrate_psd_from_profile", integrate_psd_from_profile)
