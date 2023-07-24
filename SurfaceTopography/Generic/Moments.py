"""

Computation of moments of a discrete curve, e.g. the PSD

The example below computes the PSD of a surface topography and integrates it to obtain the RMS height.

>>> from SurfaceTopography.Generation import fourier_synthesis
>>> t = fourier_synthesis((256, 256), (256,256), hurst=0.8, rms_height=1, short_cutoff=4, long_cutoff=64)
>>> hrms = t.rms_height_from_profile()
>>> hrms_from_psd = np.sqrt(compute_1d_moment(,)
>>> assert abs(hrms_from_psd / hrms - 1)  < 0.1

However, because the averaging of the PSD before integration introduces errors,
it is better to sum directly the raw spectum,
see the methods `integrate_psd` of the topographies and the container.

"""
import numpy as np
import scipy.integrate


def compute_1d_moment(x, y, order=1, cumulative=False):
    r"""

    Computes the moment of order :math:`\alpha`

    ..math::

        m_\alpha = \int dx y x^{\alpha}

    using trapezoidal interpolation of the integrand

    Parameters:
    -----------
    x: np.ndarray
        sample points
    y: np.ndarray
        values of the function sampled at x
    order: float, default=1
        order of the moment
    cumulative: bool, default=True
        if True returns the cumulative value of this integral for each x (uses cumtrapz instead of trapz)
    Returns
    -------
    If `cumulative` is `False` (Default)
    integral: float
    If `cumulative` is `True`
    integral: np.ndarray of length

    """
    power = order
    integ = scipy.integrate.cumtrapz if cumulative else np.trapz
    return integ(y * x ** power, x) / np.pi


def compute_iso_moment(x, y, order=1, cumulative=False):
    r"""
        Computes the moment of order :math:`\alpha` of the 2D isotropic funtion :math:`y_{2D}(x)`

        :math:`y` is the 1D representation of the isotropic function :math:`y_{2D}(x_1, x_2) = y(|\vec x|)`

        ..math::

            m_\alpha = \int dx_1 dx_2 y_{2D}(x_1, x_2) |\vec  x|^{\alpha} = 2\pi \int dx y(x) x^{\alpha + 1}

        using trapezoidal interpolation of the integrand

        Parameters:
        -----------
        x: np.ndarray
            sample points
        y: np.ndarray
            values of the function sampled at x
        order: float, default=1
            order of the moment
        cumulative: bool, default=True
            if True returns the cumulative value of this integral for each x (uses cumtrapz instead of trapz)
        Returns
        -------
        If `cumulative` is `False` (Default)
        integral: float
        If `cumulative` is `True`
        integral: np.ndarray of length
    """
    power = order + 1
    integ = scipy.integrate.cumtrapz if cumulative else np.trapz
    return integ(y * x ** power / (2 * np.pi), x)
