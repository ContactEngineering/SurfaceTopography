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

from ..common import radial_average
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
    r = r[1:n]
    B = 8 * A[1:n] - 2 * A[2::2]
    # Important: The following expression relies on the fact that r is equally spaced!
    return r[B > 0], np.sqrt(B[B > 0]) / r[B > 0] ** 2


def scale_dependent_curvature_2D(topography, nbins=100):
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
    nbins : int
        Number of bins for radial average. Note: Returned array can be smaller
        than this because bins without data points are discarded.

    Returns
    -------
    r : array
        Distances. (Units: length)
    slope : array
        Slope. (Units: dimensionless)
    """  # noqa: E501
    _, _, A_xy = topography.autocorrelation_2D(return_map=True)
    nx, ny = A_xy.shape
    nx = (nx + 1) // 2
    ny = (ny + 1) // 2

    sx, sy = topography.physical_sizes
    B_xy = 8 * A_xy[1:nx, 1:ny] - 2 * A_xy[2::2, 2::2]
    # Radial average
    r_edges, n, r_val, B_val = radial_average(
        # pylint: disable=invalid-name
        B_xy, (sx + sy) / 2, nbins, physical_sizes=(sx, sy), full=False)

    return r_val[B_val > 0], np.sqrt(B_val[B_val > 0]) / r_val[B_val > 0]

# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_curvature_1D',
                                             scale_dependent_curvature_1D)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_1D',
                                              scale_dependent_curvature_1D)
UniformTopographyInterface.register_function('scale_dependent_curvature_2D',
                                             scale_dependent_curvature_2D)
