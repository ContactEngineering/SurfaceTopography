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
    nz = np.nonzero(B < 0)[0]
    if len(nz) > 0:
        n = nz[0]
    # Important: The following expression relies on the fact that r is equally spaced!
    return r[:n], np.sqrt(B[:n]) / r[:n] ** 2


def scale_dependent_curvature_2D(topography, nbins=50, bin_edges='log'):
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
    nbins : int, optional
        Number of bins for radial average. Note: Returned array can be smaller
        than this because bins without data points are discarded.
        (Default: 50)
    bin_edges : {'log', 'quadratic', 'linear', array_like}, optional
        Edges used for binning the average. Specifying 'log' yields bins
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        Alternatively, it is possible to explicitly specify the bin edges.
        If bin_edges are explicitly specified, then `rmax` and `nbins` is
        ignored. (Default: 'log')

    Returns
    -------
    r : array
        Distances. (Units: length)
    curvature : array
        curvature. (Units: 1/length)
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
        B_xy, rmax=(sx + sy) / 2,
        nbins=nbins, bin_edges=bin_edges,
        physical_sizes=(sx, sy), full=False)

    return r_val[B_val > 0], np.sqrt(B_val[B_val > 0]) / r_val[B_val > 0]

# Register analysis functions from this module
UniformTopographyInterface.register_function('scale_dependent_curvature_1D',
                                             scale_dependent_curvature_1D)
NonuniformLineScanInterface.register_function('scale_dependent_curvature_1D',
                                              scale_dependent_curvature_1D)
UniformTopographyInterface.register_function('scale_dependent_curvature_2D',
                                             scale_dependent_curvature_2D)
