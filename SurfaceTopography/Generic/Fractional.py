import numpy as np
import scipy.integrate
from ..Support import doi

from ..Container.SurfaceContainer import SurfaceContainer
from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_autocorrelation_from_area(topography, **kwargs):
    r"""

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to
        `autocorrelation_from_area` or `autocorrelation_from_profile`
    """
    r, s = topography.scale_dependent_slope_from_area(**kwargs)
    return variance_half_derivative_via_scale_dependent_slope(r, s)


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_autocorrelation_from_profile(topography, **kwargs):
    r"""

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan or Container
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to
        `autocorrelation_from_area` or `autocorrelation_from_profile`
    """
    r, s = topography.scale_dependent_slope_from_profile(**kwargs)
    return variance_half_derivative_via_scale_dependent_slope(r, s)


@doi("10.1016/j.apsadv.2021.100190")
def variance_half_derivative_via_scale_dependent_slope(r, slope):
    r"""

    Parameters
    ----------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    Returns
    -------
    float
    variance of the half derivative  (Units: length)
    """

    # handle nans
    nan_args = np.isnan(slope)
    r = np.delete(r, nan_args)
    slope = np.delete(slope, nan_args)
    return 0.5 * (r[0] * slope[0] ** 2 + scipy.integrate.trapz(slope ** 2, x=r))


def container_variance_half_derivative_from_autocorrelation(container, scale_dependent=False, **kwargs):
    """

    The scale-dependence is introduced by considering only the ACF below the distance scale.

    This is the definition of scale-dependent elastic energy as in Wang and Müser equation 44

    see container.autocorrelation concerning further arguments

    Returns
    ------

    if scale_dependent
    r: array of floats
    distance scale
    variance_half_derivative: array of floats


    """

    # This is how martin Mueser defines the SDRP elastic energy

    r, acf = container.autocorrelation(**kwargs)
    r = r[~np.isnan(acf)]
    acf = acf[~np.isnan(acf)]

    # we assume continuity below the smallest distance, i.e. that the slope is constant
    small_scale_contrib = (acf[0] / r[0])
    if scale_dependent:
        return r, small_scale_contrib + np.concatenate([[0], scipy.integrate.cumtrapz(acf / r ** 2, x=r)])
    else:
        return small_scale_contrib + scipy.integrate.trapz(acf / r ** 2, x=r)


SurfaceContainer.register_function("variance_half_derivative_from_autocorrelation",
                                   container_variance_half_derivative_from_autocorrelation)
# Register analysis functions from this module
UniformTopographyInterface.register_function('variance_half_derivative_via_autocorrelation_from_profile',
                                             variance_half_derivative_via_autocorrelation_from_profile)
NonuniformLineScanInterface.register_function('variance_half_derivative_via_autocorrelation_from_profile',
                                              variance_half_derivative_via_autocorrelation_from_profile)
UniformTopographyInterface.register_function('variance_half_derivative_via_autocorrelation_from_area',
                                             variance_half_derivative_via_autocorrelation_from_area)
SurfaceContainer.register_function('variance_half_derivative_via_autocorrelation_from_profile',
                                   variance_half_derivative_via_autocorrelation_from_profile)
