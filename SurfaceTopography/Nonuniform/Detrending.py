#
# Copyright 2018, 2020, 2024 Lars Pastewka
#           2019 Antoine Sanner
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
Helper functions to compute trends of surfaces
"""

import numpy as np

from SurfaceTopography.HeightContainer import NonuniformLineScanInterface
from SurfaceTopography.NonuniformLineScan import DecoratedNonuniformTopography


def polyfit(self, deg):
    r"""
    Compute the detrending plane that, if subtracted, minimizes the rms height
    of the surface. The detrending plane is parameterized as a polynomial:

    .. math::

        p(x) = \sum_{k=0}^n a_k x^k

    The values of :math:`a_k` are returned by this function.

    The rms height of the surface is given by (see `rms_height`)

    .. math::

        h_\text{rms}^2 = \frac{1}{3L} \sum_{i=0}^{N-2} \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i

    where :math:`N` is the total number of data points. Hence we need to solve the following minimization problem:

    .. math::

        \min_{\{a_k\}} \left\{ \frac{1}{3L} \sum_{i=0}^{N-2} \left[ (h_i - p(x_i))^2 + (h_{i+1} - p(x_{i+1}))^2 + (h_i - p(x_i))(h_{i+1} - p(x_{i+1})) \right] \Delta x_i \right\}

    This gives the system of linear equations (one for each :math:`k`)

    .. math::

        \sum_{i=0}^{N-2} \left( \left[ 2h_i + h_{i+1} \right] x_i^k + \left[ 2h_{i+1} + h_i \right] x_{i+1}^k \right) \Delta x_i = \sum_{l=0}^n a_l \sum_{i=0}^{N-2} \left( 2x_i^{k+l} + 2x_{i+1}^{k+l} + x_i^k x_{i+1}^l + x_{i+1}^k x_i^l \right) \Delta x_i

    Parameters
    ----------
    self : :obj:`NonuniformLineScan`
        SurfaceTopography object containing height information.
    deg : int
        Degree of polynomial :math:`n`.

    Returns
    -------
    a : array
        Array with coefficients :math:`a_k`.
    """  # noqa: E501
    x, h = self.positions_and_heights()
    dx = np.diff(x)
    k = np.arange(deg + 1).reshape(-1, 1)
    b = np.sum(((2 * h[:-1] + h[1:]) * x[:-1] ** k +
                (2 * h[1:] + h[:-1]) * x[1:] ** k) * dx,
               axis=1)
    L = k.reshape(1, -1, 1)
    k = k.reshape(-1, 1, 1)
    A = np.sum((2 * x[:-1] ** (k + L) + 2 * x[1:] ** (k + L) +
                x[:-1] ** k * x[1:] ** L + x[1:] ** k * x[:-1] ** L) * dx,
               axis=2)
    return np.linalg.solve(A, b)


class DetrendedNonuniformTopography(DecoratedNonuniformTopography):
    """
    Remove trends from a topography. This is achieved by fitting polynomials
    to the topography data to extract trend lines. The resulting topography
    is then detrended by substracting these trend lines.
    """

    _detrend_functions = {
        'mean': lambda self: [self.parent_topography.mean()],
        # same as 'mean', deprecate 'center' in the future
        'center': lambda self: [self.parent_topography.mean()],
        'median': lambda self: [self.parent_topography.median()],
        'rms-tilt': lambda self: self.parent_topography.polyfit(1),
        # same as 'rms-tilt', deprecate 'height' in the future
        'height': lambda self: self.parent_topography.polyfit(1),
        'mad-tilt': lambda self: self.parent_topography.mad_polyfit(1),
        'slope': lambda self: [self.parent_topography.mean(), self.parent_topography.derivative(1).mean()],
        'rms-curvature': lambda self: self.parent_topography.polyfit(2),
        # same as 'rms-curvature', deprecate 'curvature' in the future
        'curvature': lambda self: self.parent_topography.polyfit(2),
        'mad-curvature': lambda self: self.parent_topography.mad_polyfit(2),
    }

    def __init__(self, topography, detrend_mode='height', coeffs=None, info={}):
        """
        Parameters
        ----------
        topography : SurfaceTopography
            SurfaceTopography to be detrended.
        detrend_mode : str
            'mean': center the topography to its mean, no trend correction.
            'median': center the topography to its median, no trend correction.
            'rms-tilt': adjust slope such that rms height is minimized.
            'mad-tilt': adjust slope such that mad height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'rms-curvature': adjust slope and curvature such that rms height is minimized.
            (Default: 'height')
        coeffs : array-like, optional
            Coefficients of the detrending plane. If not given, they are
            computed from the detrend_mode. (Default: None)
        """
        super().__init__(topography, info=info)
        self._detrend_mode = detrend_mode
        self._coeffs = coeffs
        if self._coeffs is None:
            self._detrend()

    def _detrend(self):
        try:
            self._coeffs = self._detrend_functions[self._detrend_mode](self)
        except KeyError:
            raise ValueError("Unsupported detrend mode '{}' for line scans.".format(self._detrend_mode))

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._detrend_mode, self._coeffs
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._detrend_mode, self._coeffs = state
        super().__setstate__(superstate)

    @property
    def coeffs(self, ):
        return self._coeffs

    @property
    def detrend_mode(self, ):
        return self._detrend_mode

    @detrend_mode.setter
    def detrend_mode(self, detrend_mode):
        self._detrend_mode = detrend_mode
        self._detrend()

    @property
    def is_periodic(self):
        """
        SurfaceTopography stays periodic only after detrend mode "center".
        Otherwise the detrended SurfaceTopography is non-periodic.
        """
        if self.detrend_mode == "center":
            return self.parent_topography.is_periodic
        else:
            return False

    @property
    def x_range(self):
        return self.parent_topography.x_range

    def positions(self):
        return self.parent_topography.positions()

    def heights(self):
        """ Computes the combined profile.
        """
        if len(self._coeffs) == 1:
            a0, = self._coeffs
            return self.parent_topography.heights() - a0
        x = self.positions()
        if len(self._coeffs) == 2:
            a0, a1 = self._coeffs
            return self.parent_topography.heights() - a0 - a1 * x
        elif len(self._coeffs) == 3:
            a0, a1, a2 = self._coeffs
            return self.parent_topography.heights() - a0 - a1 * x - a2 * x * x
        else:
            raise RuntimeError('Unknown physical_sizes of coefficients '
                               'tuple.')

    def stringify_plane(self, fmt=lambda x: str(x)):
        str_coeffs = [fmt(x) for x in self._coeffs]
        if len(self._coeffs) == 1:
            h0, = str_coeffs
            return h0
        elif len(self._coeffs) == 2:
            return '{0} + {1} x'.format(*str_coeffs)
        elif len(self._coeffs) == 3:
            return '{0} + {1} x + {2} x^2'.format(*str_coeffs)
        else:
            raise RuntimeError('Unknown physical_sizes of coefficients '
                               'tuple.')

    @property
    def curvatures(self):
        if len(self._coeffs) == 3:
            return 2 * self._coeffs[2],
        else:
            return 0,


NonuniformLineScanInterface.register_function('polyfit', polyfit)

NonuniformLineScanInterface.register_function('detrend', DetrendedNonuniformTopography)
