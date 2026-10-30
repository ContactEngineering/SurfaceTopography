#
# Copyright 2021 Lars Pastewka
#           2021 Antoine Sanner
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
Scale-dependent slope computed from the height-difference autocorrelation
function.
"""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface


def _slope_from_autocorrelation(r, A):
    """
    Slope :math:`[2A(r)]^{1/2}/r`, without distance zero (where the slope is
    undefined). Masked (empty) bins stay masked.
    """
    nonzero = ~((np.ma.getdata(r) == 0) & ~np.ma.getmaskarray(r))
    r = r[nonzero]
    A = A[nonzero]
    return r, np.sqrt(2 * A) / r


def scale_dependent_slope_from_profile(topography, **kwargs):
    r"""
    Compute the one-dimensional scale-dependent slope.

    The scale-dependent slope is given by

    .. math::

        h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda

    where :math:`A(\lambda)=\frac{1}{2}\langle[h(x+\lambda)-h(x)]^2\rangle`
    is the height-difference autocorrelation function. This is the rms of
    the two-point finite difference :math:`[h(x+\lambda)-h(x)]/\lambda`.
    The distance :math:`\lambda = 0` is excluded.

    Parameters
    ----------
    topography : :class:`SurfaceTopography.Topography` or :class:`SurfaceTopography.UniformLineScan` or :class:`SurfaceTopography.NonuniformLineScan`
        Container storing the topography map or line scan
    **kwargs : dict
        Additional keyword parameters are passed on to
        `autocorrelation_from_profile`

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_profile(**kwargs)
    return _slope_from_autocorrelation(r, A)


def scale_dependent_slope_from_area(topography, **kwargs):
    r"""
    Compute the two-dimensional, radially averaged scale-dependent slope.

    The scale-dependent slope is given by

    .. math::

        h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda

    where :math:`A(\lambda)=\frac{1}{2}\langle[h(x+\lambda)-h(x)]^2\rangle`
    is the height-difference autocorrelation function. This is the rms of
    the two-point finite difference :math:`[h(x+\lambda)-h(x)]/\lambda`.
    The distance :math:`\lambda = 0` is excluded.

    Parameters
    ----------
    topography : :class:`SurfaceTopography.Topography`
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to
        `autocorrelation_from_area`

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_area(**kwargs)
    return _slope_from_autocorrelation(r, A)


# Register analysis functions from this module. Note: the areal variant is
# not registered for nonuniform line scans, which have no
# `autocorrelation_from_area`.
UniformTopographyInterface.register_function('scale_dependent_slope_from_profile', scale_dependent_slope_from_profile)
NonuniformLineScanInterface.register_function('scale_dependent_slope_from_profile', scale_dependent_slope_from_profile)
UniformTopographyInterface.register_function('scale_dependent_slope_from_area', scale_dependent_slope_from_area)
