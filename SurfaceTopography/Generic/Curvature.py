#
# Copyright 2021 Lars Pastewka
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

from ..HeightContainer import UniformTopographyInterface
from ..HeightContainer import NonuniformLineScanInterface


def scale_dependent_curvature_1D(topography, **kwargs):
    r"""
    Compute the one-dimensional scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[8A(\lambda) - 2A(2\lambda)\right]^{1/2}/\lambda^2
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to `autocorrelation_1D`

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_1D(**kwargs)
    n = (len(r) + 1) // 2
    # Important: The following expression relies on the fact that r is equally spaced!
    return r[1:n], np.sqrt(8 * A[1:n] - 2 * A[2::2]) / r[1:n]


def scale_dependent_curvature_2D(topography, **kwargs):
    r"""
    Compute the two-dimensional, radially averaged scale-dependent slope.

    The scale-dependent slope is given by

       .. math::
         :nowrap:

         \begin{equation}
         h_\text{rms}^\prime(\lambda) = \left[8A(\lambda) - 2A(2\lambda)\right]^{1/2}/\lambda
         \end{equation}

    where :math:`A(\lambda)` is the autocorrelation function.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Container storing the uniform topography map
    **kwargs : dict
        Additional keyword parameters are passed on to `autocorrelation_2D`

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    r, A = topography.autocorrelation_2D(**kwargs)
    n = (len(r) + 1) // 2
    return r[1:n], np.sqrt(8 * A[1:n] - 2 * A[2::2]) / r[1:n]

# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_curvature_1D',
                                             scale_dependent_curvature_1D)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_1D',
                                              scale_dependent_curvature_1D)
UniformTopographyInterface.register_function('scale_dependent_curvature_2D',
                                             scale_dependent_curvature_2D)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_2D',
                                              scale_dependent_curvature_2D)
