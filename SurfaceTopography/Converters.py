#
# Copyright 2019-2021 Lars Pastewka
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
Utility function for conversion between uniform and nonuniform
representations.
"""

import numpy as np

from .HeightContainer import UniformTopographyInterface, NonuniformLineScanInterface
from .NonuniformLineScan import NonuniformLineScan, DecoratedNonuniformTopography
from .UniformLineScanAndTopography import UniformLineScan, DecoratedUniformTopography


class WrapAsNonuniformLineScan(DecoratedNonuniformTopography):
    """
    Wrap a uniform topography into a nonuniform one.
    """

    def __init__(self, topography, info={}):
        """
        Parameters
        ----------
        topography : :obj:`NonuniformLineScan`
            SurfaceTopography to wrap.
        """
        super().__init__(topography, info=info)

        # This is populated with functions from the nonuniform topography, but
        # this is a uniform topography
        self._functions = NonuniformLineScan._functions

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def physical_sizes(self):
        s, = self.parent_topography.physical_sizes
        p, = self.parent_topography.pixel_size
        return s - p,

    @property
    def is_periodic(self):
        return False

    # Implement nonuniform line scan interface

    @property
    def nb_grid_pts(self):
        return self.parent_topography.nb_grid_pts

    @property
    def x_range(self):
        s, = self.parent_topography.physical_sizes
        p, = self.parent_topography.pixel_size
        return 0, s - p

    def positions(self):
        """
        Returns array containing the lateral positions.
        """
        r, = self.parent_topography.nb_grid_pts
        p, = self.parent_topography.pixel_size
        return np.arange(r) * p

    def heights(self):
        """
        Returns array containing the topography data.
        """
        return self.parent_topography.heights()


class UniformlyInterpolatedLineScan(DecoratedUniformTopography):
    """
    Interpolate a topography onto a uniform grid.
    """

    def __init__(self, topography, nb_points=None, padding=0, nb_interpolate=None, info={}):
        """
        Convert a nonuniform line scan to a uniform line scan by explicit
        linear interpolation between the discrete coordinates of the
        nonuniform scan.

        Parameters
        ----------
        topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
            SurfaceTopography to interpolate.
        nb_points : int, optional
            Number of equidistant grid points. Number of points will be
            automatically computed from the mean grid spacing and
            `nb_interpolate` if set to None. (Default: None)
        padding : int, optional
            Number of padding grid points, zeros appended to the data.
            (Default: 0)
        nb_interpolate : int
            Number of grid points to between closest points on surface.
            (Default: None)
        """
        super().__init__(topography, info=info)
        self.nb_points = nb_points
        self.nb_interpolate = nb_interpolate
        self._update_nb_points()
        self.padding = padding

        # This is populated with functions from the nonuniform topography, but
        # this is a uniform topography
        self._functions = UniformLineScan._functions

    def _update_nb_points(self):
        """Automatically compute nb_points if it is None"""
        if self.nb_points is None:
            if self.nb_interpolate is None:
                raise ValueError('You need to specify either `nb_points` or `nb_interpolate`.')
            s, = self.parent_topography.physical_sizes
            x = self.parent_topography.positions()
            min_dist = np.min(np.diff(x))
            if min_dist <= 0:
                raise RuntimeError('This is a reentrant nonuniform line scan. Reentrant line scans cannot be converted '
                                   'to uniform line scans.')
            self._nb_points = self.nb_interpolate * int(s / min_dist)
        else:
            if self.nb_interpolate is not None:
                raise ValueError('You cannot specify both, `nb_points` and `nb_interpolate`.')
            self._nb_points = self.nb_points

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.nb_points, self.padding, self.nb_interpolate
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.nb_points, self.padding, self.nb_interpolate = state
        self._update_nb_points()
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def physical_sizes(self):
        s, = self.parent_topography.physical_sizes
        return s * (self._nb_points + self.padding) / self._nb_points,

    @property
    def is_periodic(self):
        return False

    @property
    def is_uniform(self):
        return True

    # Implement uniform line scan interface

    @property
    def nb_grid_pts(self):
        """Return nb_grid_pts, i.e. number of pixels, of the topography."""
        return self._nb_points + self.padding,

    @property
    def pixel_size(self):
        return tuple(s / r for s, r in zip(self.physical_sizes, self.nb_grid_pts))

    @property
    def area_per_pt(self):
        return self.pixel_size

    @property
    def has_undefined_data(self):
        return False

    def positions(self):
        left, right = self.parent_topography.x_range
        size = right - left
        return np.linspace(left - size * self.padding / (2 * self._nb_points),
                           right + size * self.padding / (2 * self._nb_points),
                           self._nb_points + self.padding)

    def heights(self):
        """ Computes the rescaled profile.
        """
        x = self.positions()
        return np.interp(x, *self.parent_topography.positions_and_heights())


# Register pipeline functions from this module
UniformTopographyInterface.register_function('to_nonuniform', WrapAsNonuniformLineScan)
NonuniformLineScanInterface.register_function('to_uniform', UniformlyInterpolatedLineScan)
