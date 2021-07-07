#
# Copyright 2015-2016, 2018-2021 Lars Pastewka
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
Functions computing scalar roughness parameters
"""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface


def rms_height(topography):
    r"""
    Computes root-mean square height fluctuation of the line scan:

    .. math::

        h_\text{rms} = \left[\frac{1}{L} \int_0^L dx\, h^2(x)\right]^{1/2}

    Function approximates topography between data points as piece-wise linear.
    The piece-wise linear section between point :math:`i` and point
    :math:`i+1` contributes

    .. math::

        \int_{0}^{\Delta x_i} dx\, \left( h_i + \frac{\Delta h_i}{\Delta x_i} x \right)^2 = \frac{1}{3} \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i

    to the above integral, where :math:`\Delta x_i=x_{i+1}-x_i` and :math:`\Delta h_i=h_{i+1}-h_i`.

    Parameters
    ----------
    topography : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_height : float or array
        Root-mean square height.
    """  # noqa: E501
    x, h = topography.positions_and_heights()
    dx = np.diff(x)
    if len(x) <= 1:
        return 0.0
    L = x[-1] - x[0]
    h0 = h - topography.mean()
    return np.sqrt(
        np.sum((h0[:-1] ** 2 + h0[1:] ** 2 + h0[:-1] * h0[1:]) * dx) / (3 * L))


def rms_slope(topography):
    r"""
    Computes root-mean square slope fluctuation of the line scan:

    .. math:: h^\prime_\text{rms} = \left[ \frac{1}{L} \int_0^L dx\, \left(\frac{\partial h}{\partial x}\right)^2 \right]^{1/2}

    Function approximates topography between data points as piece-wise linear.
    The piece-wise linear section between point :math:`i` and point
    :math:`i+1` contributes

    .. math:: \int_{0}^{\Delta x_i} dx\, \left( \frac{\Delta h_i}{\Delta x_i} \right)^2 = \frac{\Delta h_i^2}{\Delta x_i}

    where :math:`\Delta x_i=x_{i+1}-x_i` and :math:`\Delta h_i=h_{i+1}-h_i` to the above integral.

    Parameters
    ----------
    topography : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_slope : float
        Root-mean square slope.
    """  # noqa: E501
    x, h = topography.positions_and_heights()
    dh = np.diff(h)
    dx = np.diff(x)
    L = x[-1] - x[0]

    return np.sqrt(np.sum(dh ** 2 / dx) / L)


def rms_curvature(topography):
    r"""
    Computes root-mean square slope fluctuation of the line scan:

    Parameters
    ----------
    topography : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_slope : float
        Root-mean square slope.
    """
    x = topography.positions()
    d2 = topography.derivative(n=2)
    # The second derivative cannot be evaluated on the two end points
    L = x[-2] - x[1]

    return np.sqrt(np.trapz(d2 ** 2, x[1:-1]) / L)


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('rms_height_from_profile', rms_height)
NonuniformLineScanInterface.register_function('rms_slope_from_profile', rms_slope)
NonuniformLineScanInterface.register_function('rms_curvature_from_profile', rms_curvature)
