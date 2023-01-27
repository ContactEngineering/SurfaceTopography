from ..Generic.Moments import (compute_iso_moment,
                               compute_1d_moment, )
import numpy as np
from ..Container import SurfaceContainer


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
