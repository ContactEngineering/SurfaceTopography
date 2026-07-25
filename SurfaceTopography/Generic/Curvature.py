#
# Copyright 2021, 2023 Lars Pastewka
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


def _curvature_stencil(r, A):
    r"""
    Evaluate :math:`B(\lambda) = 8A(\lambda) - 2A(2\lambda)` from a sampled
    autocorrelation function.

    The autocorrelation may be sampled on an arbitrary (e.g. log-spaced or
    reliability-trimmed) grid, so :math:`A` is evaluated by linear
    interpolation. This is exact where :math:`2\lambda` falls onto a grid
    point, in particular everywhere on a linearly spaced grid that starts
    at zero.
    """
    valid = np.isfinite(r) & np.isfinite(A)
    r = r[valid]
    A = A[valid]
    # We need A(2 lambda), i.e. we can only evaluate B up to half the
    # largest sampled distance; distance zero is excluded because the
    # curvature expression divides by lambda^2
    mask = np.logical_and(r > 0, 2 * r <= r[-1])
    rm = r[mask]
    B = 8 * np.interp(rm, r, A) - 2 * np.interp(2 * rm, r, A)
    # Truncate at the first negative value
    nz = np.nonzero(B < 0)[0]
    if len(nz) > 0:
        rm = rm[:nz[0]]
        B = B[:nz[0]]
    return rm, B


def scale_dependent_curvature_from_profile(topography, **kwargs):
    r"""
    Compute the one-dimensional scale-dependent curvature.

    The scale-dependent curvature is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^{\prime\prime}(\lambda) = \left[8A(\lambda) - 2A(2\lambda)\right]^{1/2}/\lambda^2
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : :class:`SurfaceTopography.Topography` or :class:`SurfaceTopography.UniformLineScan`
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to `autocorrelation_from_profile`

    Returns
    -------
    r : array
        Distances. (Units: length)
    curvature : array
        Curvature. (Units: 1/length)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_profile(**kwargs)
    r, B = _curvature_stencil(r, A)
    return r, np.sqrt(B) / r ** 2


def scale_dependent_curvature_from_area(topography, **kwargs):
    r"""
    Compute the two-dimensional, radially averaged scale-dependent curvature.

    The scale-dependent curvature is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^{\prime\prime}(\lambda) = 4\left[8A(\lambda/2) - 2A(\lambda)\right]^{1/2}/\lambda^2
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : SurfaceTopography or UniformLineScan
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to `autocorrelation_from_area`

    Returns
    -------
    r : array
        Distances. (Units: length)
    curvature : array
        Curvature. (Units: 1/length)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_area(**kwargs)
    r, B = _curvature_stencil(r, A)
    return 2 * r, np.sqrt(B) / r ** 2


# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_curvature_from_profile',
                                             scale_dependent_curvature_from_profile)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_from_profile',
                                              scale_dependent_curvature_from_profile)
UniformTopographyInterface.register_function('scale_dependent_curvature_from_area',
                                             scale_dependent_curvature_from_area)
