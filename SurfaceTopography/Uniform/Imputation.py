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
import scipy.sparse

from .GeometryAnalysis import assign_patch_numbers_area, assign_patch_numbers_profile, outer_perimeter_area, \
    outer_perimeter_profile
from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography


class InterpolateUndefinedDataHarmonic(DecoratedUniformTopography):
    """
    Replace undefined data points by interpolation of neighboring
    points with harmonic functions (solutions of the Laplace equation).
    """

    name = 'interpolate_undefined_data_harmonic'

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
        heights = self.parent_topography.heights().copy()
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
                # Create mask identifying points in patch
                patch_mask = patch_ids == id
                nb_patch = np.sum(patch_mask)

                # Create mask identifying perimeter points
                if dim == 2:
                    perimeter_mask = outer_perimeter_area(patch_mask, self.is_periodic)
                else:
                    perimeter_mask = outer_perimeter_profile(patch_mask, self.is_periodic)
                nb_perimeter = np.sum(perimeter_mask)

                # Total number of pixels
                nb_pixels = nb_patch + nb_perimeter

                # Create unique pixel indices in patch and perimeter
                pixel_index = np.zeros_like(patch_ids)
                pixel_index[patch_mask] = np.arange(nb_patch)
                pixel_index[perimeter_mask] = np.arange(nb_patch, nb_pixels)

                # Assemble Laplace matrix. Each patch pixel couples to its
                # nearest neighbors; for nonperiodic topographies, stencil
                # legs that cross the domain boundary must be dropped (and
                # the diagonal reduced correspondingly, yielding a natural
                # boundary condition). An unconditional `np.roll` would wrap
                # around the boundary and couple edge pixels to unrelated
                # pixels (whose `pixel_index` entry is zero, aliasing patch
                # unknown #0).
                legs = [(1, 0), (-1, 0)]
                if dim == 2:
                    legs += [(1, 1), (-1, 1)]

                diagonal = np.zeros(nb_patch)
                rows = []
                cols = []
                for shift, axis in legs:
                    neighbor = np.roll(pixel_index, shift, axis)
                    valid = np.ones_like(patch_mask)
                    if not self.is_periodic:
                        # `np.roll(a, 1)[p]` is the neighbor a[p - 1], so
                        # for shift == 1 the wrapped (invalid) entries are
                        # at the low edge of the axis, and at the high edge
                        # for shift == -1
                        edge = [slice(None)] * dim
                        edge[axis] = 0 if shift == 1 else -1
                        valid[tuple(edge)] = False
                    sel = np.logical_and(patch_mask, valid)
                    rows += [pixel_index[sel]]
                    cols += [neighbor[sel]]
                    diagonal[pixel_index[sel]] -= 1

                laplace = scipy.sparse.coo_matrix(
                    (np.concatenate([diagonal] + [np.ones(len(r)) for r in rows] + [np.ones(nb_perimeter)]),
                     (np.concatenate([np.arange(nb_patch)] + rows + [np.arange(nb_patch, nb_pixels)]),
                      np.concatenate([np.arange(nb_patch)] + cols + [np.arange(nb_patch, nb_pixels)]))),
                    shape=(nb_pixels, nb_pixels))

                # Dirichlet boundary conditions (heights on perimeter)
                rhs = np.zeros(nb_pixels)
                rhs[nb_patch:] = heights[perimeter_mask]

                # Solve for undefined heights
                heights[patch_mask] = scipy.sparse.linalg.spsolve(laplace.tocsr(), rhs)[:nb_patch]
        return heights


def interpolate_undefined_data(self, method='harmonic'):
    """
    Imputation of undefined data points in topography information that
    typically occurs in optical measurements.

    Parameters
    ----------
    self : SurfaceTopography.Topography or SurfaceTopography.UniformLineScan
        Input topography containing undefined data points.
    method : str
        Imputation methods. Options
           'harmonic': Interpolate with harmonic functions
        (Default: 'harmonic')
    """
    if method == 'harmonic':
        return self.interpolate_undefined_data_with_harmonic_function()
    else:
        raise ValueError(f"Unsupported imputation method '{method}'.")


UniformTopographyInterface.register_function('interpolate_undefined_data',
                                             interpolate_undefined_data)
UniformTopographyInterface.register_function('interpolate_undefined_data_with_harmonic_function',
                                             InterpolateUndefinedDataHarmonic)
