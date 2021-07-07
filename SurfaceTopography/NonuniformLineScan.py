#
# Copyright 2019-2021 Lars Pastewka
#           2019, 2021 Michael Röttger
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

from .HeightContainer import AbstractTopography, DecoratedTopography, \
    NonuniformLineScanInterface
from .Nonuniform.Detrending import polyfit
from .Support import deprecated
from .UnitConversion import get_unit_conversion_factor


class NonuniformLineScan(AbstractTopography, NonuniformLineScanInterface):
    """
    Nonuniform topography with point list consisting of static numpy arrays.
    """

    def __init__(self, x, y, unit=None, info={}):
        """
        Constructor.

        Arguments
        ---------
        x : array_like
            x-positions of the data points that sample the line scan.
        y : array_like
            y-positions of the data points that sample the line scan.
        unit : str, optional
            The length unit.
        info : dict, optional
            The info dictionary containing auxiliary data. This data is never
            used by SurfaceTopography but can be used by third-party codes.
        """
        super().__init__(unit=unit, info=info)
        self._x = np.asarray(x)
        self._h = np.asarray(y)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._x, self._h
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._x, self._h = state
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
        return False

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

    def squeeze(self):
        return self


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
    def unit(self):
        if self._unit is None:
            return self.parent_topography.unit
        else:
            return self._unit

    @property
    def info(self):
        info = self.parent_topography.info
        info.update(self._info)
        if self.unit is not None:
            info.update(dict(unit=self.unit))
        return info

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
    """Scale heights, positions, or both."""

    def __init__(self, topography, unit, info={}):
        """
        This topography wraps a parent topography and rescales x and z
        coordinates according to certain rules.

        Arguments
        ---------
        topography : :obj:`NonuniformTopographyInterface`
            Parent topography
        unit : str
            Target unit.
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, unit=unit, info=info)

    @property
    def height_scale_factor(self):
        return get_unit_conversion_factor(self.parent_topography.unit, self.unit)

    # For backwards compatibility
    @property
    @deprecated(version='0.94.0', alternative='height_scale_factor')
    def scale_factor(self):
        return self.height_scale_factor

    @property
    def position_scale_factor(self):
        return get_unit_conversion_factor(self.parent_topography.unit, self.unit)

    @property
    def physical_sizes(self):
        """Compute rescaled physical sizes."""
        return self.position_scale_factor * super().physical_sizes[0],

    def positions(self):
        """Compute the rescaled positions."""
        return self.position_scale_factor * super().positions()

    def heights(self):
        """ Computes the rescaled profile.
        """
        return self.height_scale_factor * self.parent_topography.heights()


class StaticallyScaledNonuniformTopography(ScaledNonuniformTopography):
    """Scale heights, positions, or both."""

    def __init__(self, topography, height_scale_factor, position_scale_factor=1, unit=None, info={}):
        """
        This topography wraps a parent topography and rescales x and z
        coordinates according to certain rules.

        Arguments
        ---------
        topography : :obj:`NonuniformTopographyInterface`
            Parent topography
        height_scale_factor : float
            Factor to scale heights with.
        position_scale_factor : float, optional
            Factor to scale lateral positions (`physical_sizes`, etc.) with.
            (Default: 1)
        unit : str, optional
            Target unit. This is simply used to update the metadata, not for
            determining scale factors. (Default: None)
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, unit=unit, info=info)
        self._height_scale_factor = None if height_scale_factor is None else float(height_scale_factor)
        self._position_scale_factor = None if position_scale_factor is None else float(position_scale_factor)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._height_scale_factor, self._position_scale_factor, self._unit
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._height_scale_factor, self._position_scale_factor, self._unit = state
        super().__setstate__(superstate)

    @property
    def height_scale_factor(self):
        return self._height_scale_factor

    # For backwards compatibility
    @property
    @deprecated(version='0.95.0', alternative='height_scale_factor')
    def scale_factor(self):
        return self.height_scale_factor

    @property
    def position_scale_factor(self):
        return self._position_scale_factor


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


# Register analysis functions from this module
NonuniformLineScanInterface.register_function(
    'mean', lambda this:
    np.trapz(this.heights(), this.positions()) / this.physical_sizes[0])
NonuniformLineScanInterface.register_function(
    'min', lambda this: this.heights().min())
NonuniformLineScanInterface.register_function(
    'max', lambda this: this.heights().max())

# Register pipeline functions from this module
NonuniformLineScanInterface.register_function('to_unit', ScaledNonuniformTopography)
NonuniformLineScanInterface.register_function('scale', StaticallyScaledNonuniformTopography)
NonuniformLineScanInterface.register_function('detrend', DetrendedNonuniformTopography)
