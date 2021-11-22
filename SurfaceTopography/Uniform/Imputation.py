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

from _SurfaceTopography import assign_patch_numbers as assign_patch_numbers_area

from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography

# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nnn_stencil = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]


def coordination(mask, periodic, stencil=nn_stencil):
    """
    Return a map with coordination numbers, i.e. the number of neighboring
    pixels where `mask` is true.

    Parameters
    ----------
    mask : array of bool
        Mask with true/false values (typically indenfying specified regions on
        a two-dimensional map)
    periodic : bool
        Mask will be treated as periodic if true.
    stencil : list of tuples
        Stencil for determining neighborhood. Each entry of the stencil
        contains relative coordinates of neighboring pixels, i.e. an
        entry (1, 0) indicates the the pixel to the right is a neighbor.

    Returns
    -------
    coordination : np.ndarray
        For each pixel, this array contains the number of neighboring pixels
        where mask is true.
    """

    coordination = np.zeros_like(mask, dtype=int)
    for dx, dy in stencil:
        tmp = np.array(mask, dtype=bool, copy=True)
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

                # Assemble Laplace matrix; diagonal terms
                i0 = np.arange(nb_patch)
                j0 = np.arange(nb_patch)

                # Off-diagonal terms
                i1 = pixel_index[patch_mask]
                j1 = np.roll(pixel_index, 1, 0)[patch_mask]
                i2 = pixel_index[patch_mask]
                j2 = np.roll(pixel_index, -1, 0)[patch_mask]

                if dim == 2:
                    i3 = pixel_index[patch_mask]
                    j3 = np.roll(pixel_index, 1, 1)[patch_mask]
                    i4 = pixel_index[patch_mask]
                    j4 = np.roll(pixel_index, -1, 1)[patch_mask]

                    # Laplace matrix from coordinates
                    laplace = scipy.sparse.coo_matrix(
                        (np.concatenate((-4 * np.ones(nb_patch), np.ones(nb_patch), np.ones(nb_patch),
                                         np.ones(nb_patch), np.ones(nb_patch), np.ones(nb_perimeter))),
                         (np.concatenate((i0, i1, i2, i3, i4, np.arange(nb_patch, nb_pixels))),
                          np.concatenate((j0, j1, j2, j3, j4, np.arange(nb_patch, nb_pixels))))),
                        shape=(nb_pixels, nb_pixels))
                else:
                    # Laplace matrix from coordinates
                    laplace = scipy.sparse.coo_matrix(
                        (np.concatenate((-2 * np.ones(nb_patch), np.ones(nb_patch), np.ones(nb_patch),
                                         np.ones(nb_perimeter))),
                         (np.concatenate((i0, i1, i2, np.arange(nb_patch, nb_pixels))),
                          np.concatenate((j0, j1, j2, np.arange(nb_patch, nb_pixels))))),
                        shape=(nb_pixels, nb_pixels))

                # Dirichlet boundary conditions (heights on perimeter)
                rhs = np.zeros(nb_pixels)
                rhs[nb_patch:] = heights[perimeter_mask]

                # Solve for undefined heights
                heights[patch_mask] = scipy.sparse.linalg.spsolve(laplace, rhs)[:nb_patch]
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
