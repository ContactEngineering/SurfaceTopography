
import numpy as np
import scipy.integrate
from ..Support import doi

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface

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

def variance_half_derivative_via_autocorrelation_from_profile(topography, **kwargs):
    r"""

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
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
    return 0.5 * (r[0] * slope[0]**2 + scipy.integrate.trapz(slope**2, x=r))



# Register analysis functions from this module
UniformTopographyInterface.register_function('variance_half_derivative_via_autocorrelation_from_profile', variance_half_derivative_via_autocorrelation_from_profile)
NonuniformLineScanInterface.register_function('variance_half_derivative_via_autocorrelation_from_profile', variance_half_derivative_via_autocorrelation_from_profile)
UniformTopographyInterface.register_function('variance_half_derivative_via_autocorrelation_from_area', variance_half_derivative_via_autocorrelation_from_area)
