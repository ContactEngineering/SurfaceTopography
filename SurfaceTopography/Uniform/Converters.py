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

from SurfaceTopography.HeightContainer import UniformTopographyInterface
from SurfaceTopography.NonuniformLineScan import NonuniformLineScan, DecoratedNonuniformTopography


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
        info : dict
            Additional entries for the info dictionary.
        """
        super().__init__(topography, info=info)

        if topography.dim != 1:
            raise ValueError('Only (one-dimensional) uniform line scans can be turned into nonuniform line scans.')

        if topography.is_periodic:
            raise ValueError('Periodic uniform line scans cannot be turned into nonuniform line scans.')

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


# Register pipeline functions from this module
UniformTopographyInterface.register_function('to_nonuniform', WrapAsNonuniformLineScan)
