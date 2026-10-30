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
    r"""
    Interpolate a nonuniform line scan onto a uniform grid.

    The uniform grid has positions :math:`x_0 + i \Delta x` (reported as
    :math:`i \Delta x`), :math:`i = 0, \dots, M-1`, where :math:`x_0` is
    the first position of the nonuniform line scan; heights are linearly
    interpolated between the data points. The resulting line scan is
    nonperiodic and its physical size is :math:`(M-1) \Delta x`.
    """

    def __init__(self, topography, nb_points=None, padding=0, nb_interpolate=None, pixel_size=None, info={}):
        r"""
        Convert a nonuniform line scan to a uniform line scan by explicit
        linear interpolation between the discrete coordinates of the
        nonuniform scan.

        `nb_points`, `nb_interpolate` or `pixel_size` needs to be specified.

        Parameters
        ----------
        topography : :obj:`NonuniformLineScan`
            Line scan to interpolate. Must not be reentrant.
        nb_points : int, optional
            Number of equidistant grid points :math:`M`; the grid spacing is
            :math:`\Delta x = L/(M-1)`. (Default: None)
        padding : int, optional
            Number of padding grid points appended to the data; they hold
            the value of the final data point (np.interp clamps to the
            boundary values).
            (Default: 0)
        nb_interpolate : int, optional
            Number of grid points per smallest spacing between data points;
            :math:`M` is `nb_interpolate` times :math:`\lfloor L / \min_i
            \Delta x_i \rfloor`. (Default: None)
        pixel_size : float, optional
            Grid spacing :math:`\Delta x`; :math:`M = \lfloor L/\Delta x
            \rfloor + 1`. (Default: None)
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, info=info)
        self.nb_points = nb_points
        self.nb_interpolate = nb_interpolate
        self.padding = padding
        self.in_pixel_size = pixel_size
        self._update_nb_points_and_pixel_size()

        # This is populated with functions from the nonuniform topography, but
        # this is a uniform topography
        self._functions = UniformLineScan._all_functions()

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
        parent_x, parent_h = self.parent_topography.positions_and_heights()
        # The uniform grid starts at zero, but the parent topography's
        # x-coordinates may start anywhere; shift the interpolation points
        # by the parent's origin. (np.interp clamps outside the data range,
        # so without the shift most of the grid would be filled with the
        # first or last height value.)
        return np.interp(self.positions() + parent_x[0], parent_x, parent_h)


# Register pipeline functions from this module
NonuniformLineScanInterface.register_function('to_uniform', UniformlyInterpolatedLineScan)
