#
# Copyright 2018-2020 Lars Pastewka
#           2019 Antoine Sanner
#           2019 Michael Röttger
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

"""Compute derivatives of nonuniform line scans."""

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface
from ..Support import toiter, fromiter


def derivative(self, n, scale_factor=None, distance=None, interpolation='linear', progress_callback=None):
    r"""
    Compute derivative of nonuniform line-scan. Function assumes nonperiodic
    topographies.

    First derivative: Central differences.

    Second derivative: Expand :math:`h(x+\Delta x_+)` and :math:`(x-\Delta x_-)` up to second order in the grid
    spacing :math:`\Delta x_+` and :math:`\Delta x_+`. Then
    :math:`\Delta x_- f(x+\Delta x_+) + \Delta x_+ f(x+\Delta x_-)` yields:

    .. math::

         \frac{d^2h}{dx^2} \approx 2 \frac{\Delta x_-\left[f(x+\Delta x_+)-f(x)\right] + \Delta x_+\left[f(x+\Delta x_-)-f(x)\right]}{\Delta x_+\Delta x_-(\Delta x_++\Delta x_-)}

    Parameters
    ----------
    self : :class:`SurfaceTopography.NonuniformLineScan`
        Object containing height information.
    n : int
        Number of times the derivative is taken.
    scale_factor : int or list of ints or list of tuples of ints, optional
        Scale factors are not supported by nonuniform line scans.
        (Default: None)
    distance : float or list of floats, optional
        Explicit distance scale for computation of the derivative. Note that
        the distance specifies the overall length of the stencil of lowest
        truncation order, not the effective grid spacing used by this stencil.
        The scale factor is then given by distance / (n * px) where n is the
        order of the derivative and px the grid spacing.
        (Default: None)
    interpolation : str, optional
        Only 'linear' interpolation is supported by nonuniform line scans.
        (Default: 'linear')
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)

    Returns
    -------
    derivative : np.ndarray
        Array with derivative values. Length of array is reduced by :math:`n` with
        respect to the input array for the :math:`n`-th derivative.
    """  # noqa: E501
    if scale_factor is not None:
        raise ValueError('Scale factors are not supported by nonuniform line scans.')
    if interpolation != 'linear':
        raise ValueError('Line scans only support linear interpolation.')

    if distance is not None:
        # If an explicit distance scale is given, we interpolate onto a regular grid and pass the derivative
        # calculation on to the uniform line scan derivative function.
        lower, upper = self.bandwidth()
        derivatives = []
        distances = toiter(distance)
        for i, d in enumerate(distances):
            if progress_callback is not None:
                progress_callback(i, len(distances))
            scale_factor = int(np.ceil(d / lower))
            stencil_size = d / scale_factor
            derivatives += [self.to_uniform(pixel_size=stencil_size / n).derivative(n=n, scale_factor=scale_factor,
                                                                                    interpolation='disable')]
        if progress_callback is not None:
            progress_callback(len(distances), len(distances))

        return fromiter(derivatives, distance)

    # If no explicit distance scale is given, we simply compute the derivatives from finite differences expressions on
    # nonuniform grids.
    x, h = self.positions_and_heights()
    if n == 1:
        return np.diff(h) / np.diff(x)
    elif n == 2:
        dxp = x[2:] - x[1:-1]
        dxm = x[1:-1] - x[:-2]

        return 2 * (dxm * (h[2:] - h[1:-1]) + dxp * (h[0:-2] - h[1:-1])) / (
                dxp * dxm * (dxp + dxm))
    else:
        raise RuntimeError('Currently only first and second derivatives are '
                           'supported for nonuniform topographies.')


# Register analysis functions from this module
NonuniformLineScanInterface.register_function('derivative', derivative)
