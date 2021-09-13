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

from SurfaceTopography.HeightContainer import NonuniformLineScanInterface
from SurfaceTopography.UniformLineScanAndTopography import UniformLineScan, DecoratedUniformTopography


class UniformlyInterpolatedLineScan(DecoratedUniformTopography):
    """
    Interpolate a topography onto a uniform grid.
    """

    def __init__(self, topography, nb_points=None, padding=0, nb_interpolate=None, pixel_size=None, info={}):
        """
        Convert a nonuniform line scan to a uniform line scan by explicit
        linear interpolation between the discrete coordinates of the
        nonuniform scan.

        `nb_points`, `nb_interpolate` or `pixel_size` needs to be specified.

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
        nb_interpolate : int, optional
            Number of grid points to between closest points on surface.
            (Default: None)
        pixel_size : float, optional
            Grid spacing. (Default: None)
        """
        super().__init__(topography, info=info)
        self.nb_points = nb_points
        self.nb_interpolate = nb_interpolate
        self.padding = padding
        self.in_pixel_size = pixel_size
        self._update_nb_points_and_pixel_size()

        # This is populated with functions from the nonuniform topography, but
        # this is a uniform topography
        self._functions = UniformLineScan._functions

    def _update_nb_points_and_pixel_size(self):
        """Automatically compute `nb_points` and `pixel_size` if it is None"""
        s, = self.parent_topography.physical_sizes
        x = self.parent_topography.positions()
        min_dist = np.min(np.diff(x))
        if min_dist <= 0:
            raise RuntimeError('This is a reentrant nonuniform line scan. Reentrant line scans cannot be converted '
                               'to uniform line scans.')

        if self.in_pixel_size is not None:
            if self.nb_points is not None or self.nb_interpolate is not None:
                raise ValueError('You need to specify either `nb_points`, `nb_interpolate` or `pixel_size`.')
            self._nb_points = int(s / self.in_pixel_size) + 1
            self._pixel_size = self.in_pixel_size
        elif self.nb_interpolate is not None:
            if self.nb_points is not None or self.in_pixel_size is not None:
                raise ValueError('You need to specify either `nb_points`, `nb_interpolate` or `pixel_size`.')
            self._nb_points = self.nb_interpolate * int(s / min_dist)
            self._pixel_size = s / (self._nb_points - 1)
        elif self.nb_points is not None:
            if self.nb_interpolate is not None or self.in_pixel_size is not None:
                raise ValueError('You need to specify either `nb_points`, `nb_interpolate` or `pixel_size`.')
            self._nb_points = self.nb_points
            self._pixel_size = s / (self._nb_points - 1)
        else:
            raise ValueError('You need to specify one of `nb_points`, `nb_interpolate` or `pixel_size`.')

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.nb_points, self.padding, self.nb_interpolate, self.in_pixel_size
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.nb_points, self.padding, self.nb_interpolate, self.in_pixel_size = state
        super().__setstate__(superstate)
        self._update_nb_points_and_pixel_size()

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def physical_sizes(self):
        return (self._nb_points + self.padding - 1) * self._pixel_size,

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
        return self._pixel_size,

    @property
    def area_per_pt(self):
        return self._pixel_size

    @property
    def has_undefined_data(self):
        return False

    def positions(self):
        return np.arange(self._nb_points + self.padding) * self._pixel_size

    def heights(self):
        """ Computes the rescaled profile.
        """
        x = self.positions()
        return np.interp(x, *self.parent_topography.positions_and_heights())


# Register pipeline functions from this module
NonuniformLineScanInterface.register_function('to_uniform', UniformlyInterpolatedLineScan)
