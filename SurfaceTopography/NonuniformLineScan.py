#
# Copyright 2019-2020 Lars Pastewka
#           2019 Michael Röttger
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
Support for nonuniform topogography descriptions
"""

import numpy as np

from .HeightContainer import AbstractHeightContainer, DecoratedTopography, \
    NonuniformLineScanInterface
from .Nonuniform.Detrending import polyfit


class NonuniformLineScan(AbstractHeightContainer, NonuniformLineScanInterface):
    """
    Nonuniform topography with point list consisting of static numpy arrays.
    """

    def __init__(self, x, y, info={}, periodic=False):
        super().__init__(info=info)
        self._x = np.asarray(x)
        self._h = np.asarray(y)
        self._periodic = periodic

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._x, self._h, self._periodic
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._x, self._h, self._periodic = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def physical_sizes(self):
        """Returns distance between maximum and minimum x-value."""
        return self._x[-1] - self._x[0],

    @property
    def is_periodic(self):
        """Return whether the topography is periodically repeated at the
        boundaries."""
        return self._periodic

    @property
    def is_uniform(self):
        return False

    # Implement uniform line scan interface

    @property
    def nb_grid_pts(self):
        return len(self._x),

    @property
    def x_range(self):
        return self._x[0], self._x[-1]

    def positions(self):
        return self._x

    def heights(self):
        return self._h


class DecoratedNonuniformTopography(DecoratedTopography,
                                    NonuniformLineScanInterface):
    @property
    def is_periodic(self):
        return self.parent_topography.is_periodic

    @property
    def dim(self):
        return self.parent_topography.dim

    @property
    def nb_grid_pts(self):
        return self.parent_topography.nb_grid_pts

    @property
    def physical_sizes(self):
        return self.parent_topography.physical_sizes

    @property
    def x_range(self):
        return self.parent_topography.x_range

    def positions(self):
        return self.parent_topography.positions()

    def squeeze(self):
        return NonuniformLineScan(self.positions(), self.heights(),
                                  info=self.info)


class ScaledNonuniformTopography(DecoratedNonuniformTopography):
    """ used when geometries are scaled
    """

    def __init__(self, topography, scale_factor, info={}):
        """
        Keyword Arguments:
        topography  -- SurfaceTopography to scale
        coeff -- Scaling factor
        """
        super().__init__(topography, info=info)
        self._scale_factor = float(scale_factor)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._scale_factor
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._scale_factor = state
        super().__setstate__(superstate)

    @property
    def scale_factor(self):
        return self._scale_factor

    def heights(self):
        """ Computes the rescaled profile.
        """
        return self._scale_factor * self.parent_topography.heights()


class DetrendedNonuniformTopography(DecoratedNonuniformTopography):
    """
    Remove trends from a topography. This is achieved by fitting polynomials
    to the topography data to extract trend lines. The resulting topography
    is then detrended by substracting these trend lines.
    """

    def __init__(self, topography, detrend_mode='height', info={}):
        """
        Parameters
        ----------
        topography : SurfaceTopography
            SurfaceTopography to be detrended.
        detrend_mode : str
            'center': center the topography, no trend correction.
            'height': adjust slope such that rms height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'curvature': adjust slope and curvature such that rms height is
            minimized.
            (Default: 'height')
        """
        super().__init__(topography, info=info)
        self._detrend_mode = detrend_mode
        self._detrend()

    def _detrend(self):
        if self._detrend_mode == 'center':
            x, y = self.parent_topography.positions_and_heights()
            self._coeffs = polyfit(x, y, 0)
        elif self._detrend_mode == 'height':
            x, y = self.parent_topography.positions_and_heights()
            self._coeffs = polyfit(x, y, 1)
        elif self._detrend_mode == 'slope':
            sl = self.parent_topography.derivative().mean()
            self._coeffs = [self.parent_topography.mean(), sl]
        elif self._detrend_mode == 'curvature':
            x, y = self.parent_topography.positions_and_heights()
            self._coeffs = polyfit(x, y, 2)
        else:
            raise ValueError(
                "Unsupported detrend mode '{}' for line scans."
                .format(self._detrend_mode))

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


# Register analysis functions from this module
NonuniformLineScanInterface.register_function(
    'mean', lambda this:
    np.trapz(this.heights(), this.positions()) / this.physical_sizes[0])
NonuniformLineScanInterface.register_function(
    'min', lambda this: this.heights().min())
NonuniformLineScanInterface.register_function(
    'max', lambda this: this.heights().max())

# Register pipeline functions from this module
NonuniformLineScanInterface.register_function(
    'scale', ScaledNonuniformTopography)
NonuniformLineScanInterface.register_function(
    'detrend', DetrendedNonuniformTopography)
