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
Scale-dependent slope
"""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface


def scale_dependent_slope_from_profile(topography, **kwargs):
    r"""
    Compute the one-dimensional scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : :class:`SurfaceTopography.Topography` or :class:`SurfaceTopography.UniformLineScan`
        Container storing the uniform topography map
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
    return r[1:], np.sqrt(2 * A[1:]) / r[1:]


def scale_dependent_slope_from_area(topography, **kwargs):
    r"""
    Compute the two-dimensional, radially averaged scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[2A(\lambda)\right]^{1/2}/\lambda
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
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
    return r[1:], np.sqrt(2 * A[1:]) / r[1:]


# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_slope_from_profile', scale_dependent_slope_from_profile)
NonuniformLineScanInterface.register_function('scale_dependent_slope_from_profile', scale_dependent_slope_from_profile)
UniformTopographyInterface.register_function('scale_dependent_slope_from_area', scale_dependent_slope_from_area)
NonuniformLineScanInterface.register_function('scale_dependent_slope_from_area', scale_dependent_slope_from_area)
