#
# Copyright 2015-2016, 2018-2022, 2024 Lars Pastewka
#           2018-2019 Antoine Sanner
#           2015-2016 Till Junge
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
Scalar roughness parameters of nonuniform line scans. Heights are
interpolated linearly between data points and all averages are integrals
over this piecewise-linear function.
"""

import _SurfaceTopography
import numpy as np

from ..Exceptions import ReentrantDataError
from ..HeightContainer import NonuniformLineScanInterface


def moment(topography, alpha):
    r"""
    Computes the moment of order :math:`\alpha` of the heights of the line
    scan of length :math:`L = x_{N-1} - x_0`:

    .. math::

        \langle h^\alpha \rangle = \frac{1}{L} \int_{x_0}^{x_{N-1}} dx\, h^\alpha(x)

    This function approximates the topography between data points as
    piece-wise linear. The piece-wise linear section between point :math:`i`
    and point :math:`i+1` contributes

    .. math::

        \frac{1}{\alpha+1} \frac{x_{i+1} - x_i}{L}\frac{h_{i+1}^{\alpha+1} - h_i^{\alpha+1}}{h_{i+1} - h_i}

    to the above integral.

    Parameters
    ----------
    topography : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.
    alpha : int
        Order of moment.

    Returns
    -------
    moment : float
        Moment of order `alpha` of the heights.
    """  # noqa: E501
    x, h = topography.positions_and_heights()
    dx = np.diff(x)
    if len(x) <= 1:
        return 0.0
    L = x[-1] - x[0]
    # The quotient (h2^(alpha+1) - h1^(alpha+1)) / (h2 - h1) given in the
    # docstring equals the complete homogeneous symmetric polynomial of
    # degree alpha in h1 and h2. The polynomial form is exact for equal
    # heights (where the quotient is 0/0, e.g. for quantized instrument
    # data) and avoids cancellation for nearly equal heights.
    h1, h2 = h[:-1], h[1:]
    s = sum(h1 ** k * h2 ** (alpha - k) for k in range(alpha + 1))
    return np.sum(dx * s) / ((alpha + 1) * L)


def rms_height(self):
    r"""
    Computes root-mean square height fluctuation of the line scan of length
    :math:`L = x_{N-1} - x_0`:

    .. math::

        h_\text{rms} = \left[\frac{1}{L} \int_{x_0}^{x_{N-1}} dx\, \left(h(x) - \langle h \rangle\right)^2\right]^{1/2}

    where :math:`\langle h \rangle` is the mean height (computed with the
    same piecewise-linear integration). Function approximates topography
    between data points as piece-wise linear. With heights :math:`h_i`
    measured relative to the mean, the piece-wise linear section between
    point :math:`i` and point :math:`i+1` contributes

    .. math::

        \int_{0}^{\Delta x_i} dx\, \left( h_i + \frac{\Delta h_i}{\Delta x_i} x \right)^2 = \frac{1}{3} \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i

    to the above integral, where :math:`\Delta x_i=x_{i+1}-x_i` and :math:`\Delta h_i=h_{i+1}-h_i`.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_height : float
        Root-mean square height. (Units: length)
    """  # noqa: E501
    return np.sqrt(_SurfaceTopography.nonuniform_variance(*self.positions_and_heights(), ref_h=self.mean()))


def rms_slope(self):
    r"""
    Computes root-mean square slope fluctuation of the line scan:

    .. math:: h^\prime_\text{rms} = \left[ \frac{1}{L} \int_0^L dx\, \left(\frac{\partial h}{\partial x}\right)^2 \right]^{1/2}

    Function approximates topography between data points as piece-wise linear.
    The piece-wise linear section between point :math:`i` and point
    :math:`i+1` contributes

    .. math:: \int_{0}^{\Delta x_i} dx\, \left( \frac{\Delta h_i}{\Delta x_i} \right)^2 = \frac{\Delta h_i^2}{\Delta x_i}

    to the above integral, where :math:`\Delta x_i=x_{i+1}-x_i`,
    :math:`\Delta h_i=h_{i+1}-h_i` and :math:`L = x_{N-1} - x_0`.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_slope : float
        Root-mean square slope. (Units: dimensionless)

    Raises
    ------
    ReentrantDataError
        If the line scan has overhangs (positions not strictly increasing).
    """  # noqa: E501
    x, h = self.positions_and_heights()
    dh = np.diff(h)
    dx = np.diff(x)

    if np.min(dx) <= 0:
        raise ReentrantDataError('This topography is reentrant (i.e. it contains overhangs). The rms slope '
                                 'cannot be computed for reentrant topographies.')

    L = x[-1] - x[0]

    return np.sqrt(np.sum(dh ** 2 / dx) / L)


def rms_curvature(self):
    r"""
    Computes root-mean square curvature fluctuation of the line scan:

    .. math::

        h^{\prime\prime}_\text{rms} = \left[ \frac{1}{x_{N-2} - x_1} \int_{x_1}^{x_{N-2}} dx\,
        \left(\frac{\partial^2 h}{\partial x^2}\right)^2 \right]^{1/2}

    The second derivative is evaluated at the interior points
    :math:`x_1, \dots, x_{N-2}` with the three-point stencil of
    :func:`SurfaceTopography.Nonuniform.Derivative.derivative` and its square
    is integrated with the trapezoidal rule.

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_curvature : float
        Root-mean square curvature. (Units: 1/length)
    """
    x = self.positions()
    d2 = self.derivative(n=2)
    # The second derivative cannot be evaluated on the two end points
    L = x[-2] - x[1]

    return np.sqrt(np.trapezoid(d2 ** 2, x[1:-1]) / L)


# Register analysis functions from this module
NonuniformLineScanInterface.register_function(
    'mean', lambda self: _SurfaceTopography.nonuniform_mean(*self.positions_and_heights()))
NonuniformLineScanInterface.register_function('moment', moment)
# Note: The `rms_*_from_profile` names are not marked as deprecated here,
# although they are deprecated for uniform data: deprecating them for
# nonuniform line scans would make generic code warn depending on the data
# type it happens to operate on.
NonuniformLineScanInterface.register_function('rms_height_from_profile', rms_height)
NonuniformLineScanInterface.register_function('Rq', rms_height)
NonuniformLineScanInterface.register_function('rms_slope_from_profile', rms_slope)
NonuniformLineScanInterface.register_function('Rdq', rms_slope)
NonuniformLineScanInterface.register_function('rms_curvature_from_profile', rms_curvature)
NonuniformLineScanInterface.register_function('Rddq', rms_curvature)
