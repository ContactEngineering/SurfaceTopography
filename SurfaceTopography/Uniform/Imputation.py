#
# Copyright 2021 Lars Pastewka
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
Filter pipelines for data imputation (filling undefined data points)
"""

import numpy as np

from _SurfaceTopography import assign_patch_numbers

from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography


# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nnn_stencil = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]


def coordination(c, stencil=nn_stencil):
    """
    Return a map with coordination numbers, i.e. number of neighboring patches that also contact
    """

    coordination = np.zeros_like(c, dtype=int)
    for dx, dy in stencil:
        tmp = np.array(c, dtype=bool, copy=True)
        if dx != 0:
            tmp = np.roll(tmp, dx, 0)
        if dy != 0:
            tmp = np.roll(tmp, dy, 1)
        coordination += tmp
    return coordination


def outer_perimeter(c, stencil=nn_stencil):
    """
    Return a map where surface points on the outer perimeter are marked.
    """

    return np.logical_and(np.logical_not(c), coordination(c, stencil=stencil) > 0)


class InterpolateUndefinedData(DecoratedUniformTopography):
    """
    Replace undefined data points by linear interpolation of neighboring
    points.
    """

    name = 'fill_undefined_linear'

    def __init__(self, topography, info={}):
        super().__init__(topography, info=info)

    @property
    def has_undefined_data(self):
        """
        By definition, this topography has no undefined data.
        """
        return False

    def heights(self):
        """
        Computes the topography with filled in data points.
        """
        heights = super().heights()
        if super().has_undefined_data:
            if self.dim != 2:
                raise NotImplementedError('Imputation is only implemented for topographic maps')

            # Coordinates for each point on the topography
            nx, ny = self.nb_grid_pts
            x, y = np.mgrid[:nx, :ny]

            # Get undefined data points and identify continuous patches
            mask = np.ma.getmaskarray(heights)
            nb_patches, patch_ids = assign_patch_numbers(mask, self.is_periodic)
            assert np.max(patch_ids) == nb_patches

            # We now fill in the patches individually
            for id in range(1, nb_patches):
                # Mask identifying undefined data points
                patch_mask = patch_ids == id
                patch_x = x[patch_mask]
                patch_y = y[patch_mask]

                # Mask identifying points with existing data on the edge of
                # the undefined patch
                edge_mask = outer_perimeter(patch_mask)
                edge_x = x[edge_mask]
                edge_y = y[edge_mask]

                # Matrix with distances from undefined to defined data points
                diff_x = patch_x.reshape(-1, 1) - edge_x.reshape(1, -1)
                diff_y = patch_y.reshape(-1, 1) - edge_y.reshape(1, -1)
                if self.is_periodic:
                    # Minimum image convention
                    diff_x = (diff_x + nx // 2) % nx - nx // 2
                    diff_y = (diff_y + ny // 2) % ny - ny // 2
                distances = np.sqrt(diff_x ** 2 + diff_y ** 2)
                
                # Interpolate undefined points
                heights[patch_mask] = distances.dot(heights[edge_mask]) / distances.sum(axis=1)
        return heights


UniformTopographyInterface.register_function("interpolate_undefined_data", InterpolateUndefinedData)
