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

from _SurfaceTopography import assign_patch_numbers as assign_patch_numbers_area

from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography

# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nnn_stencil = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]


def coordination(c, periodic, stencil=nn_stencil):
    """
    Return a map with coordination numbers, i.e. number of neighboring patches that also contact
    """

    coordination = np.zeros_like(c, dtype=int)
    for dx, dy in stencil:
        tmp = np.array(c, dtype=bool, copy=True)
        if dx != 0:
            tmp = np.roll(tmp, dx, 0)
            if not periodic:
                if dx > 0:
                    tmp[:dx, :] = 0
                else:
                    tmp[dx:, :] = 0
        if dy != 0:
            tmp = np.roll(tmp, dy, 1)
            if not periodic:
                if dy > 0:
                    tmp[:, :dy] = 0
                else:
                    tmp[:, dy:] = 0
        coordination += tmp
    return coordination


def outer_perimeter_area(c, periodic, stencil=nn_stencil):
    """
    Return a map where surface points on the outer perimeter are marked.
    """
    return np.logical_and(np.logical_not(c), coordination(c, periodic, stencil=stencil) > 0)


def assign_patch_numbers_profile(mask, periodic):
    patch_ids = np.cumsum(np.abs(np.diff(mask)))
    if mask[0]:
        # Patches are odd numbers
        patch_ids += 1
        if periodic and mask[-1]:
            # Assign same patch id to first and last patch
            patch_ids[patch_ids == patch_ids[-1]] = patch_ids[0]
    # Patches are odd numbers, set even numbers to zero
    patch_ids += 1
    patch_ids[(patch_ids & 0x1).astype(bool)] = 0
    patch_ids //= 2
    if mask[0]:
        patch_ids = np.append([1], patch_ids)
    else:
        patch_ids = np.append([0], patch_ids)
    nb_patches = np.max(patch_ids)
    return nb_patches, patch_ids


def outer_perimeter_profile(mask, periodic):
    p_left = np.logical_and(np.roll(mask, -1), np.logical_not(mask))
    p_right = np.logical_and(np.roll(mask, 1), np.logical_not(mask))
    if not periodic:
        p_left[-1] = False
        p_right[0] = False
    return np.logical_or(p_left, p_right)


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
        heights = self.parent_topography.heights()
        if super().has_undefined_data:
            dim = self.dim

            # Coordinates for each point on the topography
            if dim == 1:
                nx, = self.nb_grid_pts
                x = np.arange(nx)
            elif dim == 2:
                nx, ny = self.nb_grid_pts
                x, y = np.mgrid[:nx, :ny]
            else:
                # Should not happen
                raise NotImplementedError

            # Get undefined data points and identify continuous patches
            mask = np.ma.getmaskarray(heights)
            if dim == 2:
                nb_patches, patch_ids = assign_patch_numbers_area(mask, self.is_periodic)
            else:
                nb_patches, patch_ids = assign_patch_numbers_profile(mask, self.is_periodic)
            assert np.max(patch_ids) == nb_patches

            # We now fill in the patches individually
            for id in range(1, nb_patches + 1):
                # Mask identifying undefined data points
                patch_mask = patch_ids == id
                patch_x = x[patch_mask]
                if dim == 2:
                    patch_y = y[patch_mask]

                # Mask identifying points with existing data on the edge of
                # the undefined patch
                if dim == 2:
                    edge_mask = outer_perimeter_area(patch_mask, self.is_periodic)
                else:
                    edge_mask = outer_perimeter_profile(patch_mask, self.is_periodic)
                edge_x = x[edge_mask]
                if dim == 2:
                    edge_y = y[edge_mask]

                # Matrix with distances from undefined to defined data points
                diff_x = patch_x.reshape(-1, 1) - edge_x.reshape(1, -1)
                if dim == 2:
                    diff_y = patch_y.reshape(-1, 1) - edge_y.reshape(1, -1)
                if self.is_periodic:
                    # Minimum image convention
                    diff_x = (diff_x + nx // 2) % nx - nx // 2
                    if dim == 2:
                        diff_y = (diff_y + ny // 2) % ny - ny // 2
                if dim == 2:
                    inv_distances_sq = 1 / (diff_x ** 2 + diff_y ** 2)
                else:
                    inv_distances_sq = 1 / diff_x ** 2

                # Interpolate undefined points
                heights[patch_mask] = inv_distances_sq.dot(heights[edge_mask]) / inv_distances_sq.sum(axis=1)
        return heights


UniformTopographyInterface.register_function("interpolate_undefined_data", InterpolateUndefinedData)
