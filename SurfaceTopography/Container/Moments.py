from ..Generic.Moments import (compute_iso_moment,
                               compute_1d_moment, )
import numpy as np
from ..Container import SurfaceContainer


def _bandwidth_count(self, q, unit, reliable=True):
    """
    Return number of topographies that include q in their bandwidth.
    """
    factor = np.zeros_like(q)
    for t in self._topographies:
        # TODO: adapt to 2D topographies
        t = t.to_unit(unit)

        qmax = np.pi / t.pixel_size[0]
        short_cutoff = t.short_reliability_cutoff() if reliable else None
        if short_cutoff is not None:

            qmax = min(2 * np.pi / short_cutoff, qmax)


        factor += np.logical_and(
            q >= 2 * np.pi / t.physical_sizes[0],
            q <= qmax,
        )
    # All topographies have q == 0 wavevector
    factor = np.where(q==0, len(self._topographies), factor)
    return factor


def integrate_psd(self, factor, unit, window=None, reliable=True):
    integ = 0
    for t in self._topographies:
        t = t.to_unit(unit)
        integ += t.integrate_psd(lambda q: factor(q) / _bandwidth_count(self, q, unit, reliable),
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
SurfaceContainer.register_function("integrate_psd", integrate_psd)
