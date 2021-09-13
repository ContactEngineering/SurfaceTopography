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
        Additional keyword parameters are passed on to `autocorrelation_1D`

    Returns
    -------
    r : array
        Distances. (Units: length)
    curvature : array
        Curvature. (Units: 1/length)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_profile(**kwargs)
    n = (len(r) + 1) // 2
    r = r[1:n]
    B = 8 * A[1:n] - 2 * A[2::2]
    nz = np.nonzero(B < 0)[0]
    if len(nz) > 0:
        n = nz[0]
    # Important: The following expression relies on the fact that r is equally spaced!
    return r[:n], np.sqrt(B[:n]) / r[:n] ** 2


def scale_dependent_curvature_from_area(topography, nbins=None):
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
    nbins : int
        Number of bins for radial average. Bins are automatically determined
        if set to None. Note: Returned array can be smaller than this because
        bins without data points are discarded. (Default: None)

    Returns
    -------
    r : array
        Distances. (Units: length)
    curvature : array
        Curvature. (Units: 1/length)
    """  # noqa: E501
    r, A = topography.autocorrelation_from_area(nbins=nbins, bin_edges='linear')
    n = (len(r) + 1) // 2
    r = r[1:n]
    B = 8 * A[1:n] - 2 * A[2::2]
    nz = np.nonzero(B < 0)[0]
    if len(nz) > 0:
        n = nz[0]
    # Important: The following expression relies on the fact that r is equally spaced!
    return 2 * r[:n], np.sqrt(B[:n]) / r[:n] ** 2


# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_curvature_from_profile',
                                             scale_dependent_curvature_from_profile)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_from_profile',
                                              scale_dependent_curvature_from_profile)
UniformTopographyInterface.register_function('scale_dependent_curvature_from_area',
                                             scale_dependent_curvature_from_area)
