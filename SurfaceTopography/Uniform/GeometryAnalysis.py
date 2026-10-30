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
Analysis of island geometries on boolean maps (e.g. contact areas or regions
above a threshold height) defined on a uniform grid.

Besides the functions defined here, this module exposes the compiled
functions

- `assign_patch_numbers_area(mask, periodic, stencil=None)`: label connected
  patches of a two-dimensional boolean map (flood fill). The default stencil
  is the 8-neighbor stencil `nnn_stencil`; a custom stencil is a flat integer
  array of offset pairs. Returns the number of patches and a map of patch ids
  (0 outside patches, 1, 2, ... inside).
- `assign_segment_numbers_area(mask)`: label connected segments along the
  second (y) axis, row by row; the map is always treated as periodic.
  Returns the number of segments and a map of segment ids.
- `distance_map(mask)`: Euclidean distance (in units of grid points) from
  each pixel to the closest pixel where `mask` is true, measured with
  periodic boundary conditions.
"""

import numpy as np

from _SurfaceTopography import assign_patch_numbers as assign_patch_numbers_area  # noqa: F401
from _SurfaceTopography import assign_segment_numbers as assign_segment_numbers_area  # noqa: F401
from _SurfaceTopography import distance_map  # noqa: F401

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
        Mask with true/false values (typically identifying specified regions
        on a two-dimensional map)
    periodic : bool
        Mask will be treated as periodic if true.
    stencil : list of tuples
        Stencil for determining neighborhood. Each entry of the stencil
        contains relative coordinates of neighboring pixels, i.e. an
        entry (1, 0) indicates that the pixel displaced by one along the
        first axis is a neighbor. (Default: `nn_stencil`, the four nearest
        neighbors)

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
    Return a map where surface points on the outer perimeter are marked,
    i.e. points outside `c` that have at least one neighbor (as defined by
    `stencil`) inside `c`.
    """
    return np.logical_and(np.logical_not(c), coordination(c, periodic, stencil=stencil) > 0)


def inner_perimeter_area(c, periodic, stencil=nn_stencil):
    """
    Return a map where surface points on the inner perimeter are marked,
    i.e. points inside `c` that have at least one neighbor (as defined by
    `stencil`) outside `c`.
    """

    return np.logical_and(c, coordination(c, periodic, stencil=stencil) < len(stencil))


def patch_areas(patch_ids):
    """
    Return an array containing the area (number of pixels) of each patch,
    given a map of patch ids as returned by `assign_patch_numbers_area`.
    Entry `i` is the area of the patch with id `i + 1`.
    """
    return np.bincount(patch_ids.reshape((-1,)))[1:]


def assign_patch_numbers_profile(mask, periodic):
    """
    Label connected patches (segments) of a one-dimensional boolean mask.

    Parameters
    ----------
    mask : array of bool
        One-dimensional mask.
    periodic : bool
        If true, a patch touching the end of the mask is joined with a patch
        touching its start.

    Returns
    -------
    nb_patches : int
        Number of patches.
    patch_ids : np.ndarray of int
        Patch id for each point; 0 outside patches, 1, 2, ... inside.
    """
    patch_ids = np.cumsum(np.abs(np.diff(mask)))
    if mask[0]:
        # Patches are odd numbers
        patch_ids += 1
        if periodic and mask[-1]:
            # Assign same patch id to first and last patch. Note that the
            # patch containing the first pixel always has (pre-transform)
            # id 1 here; `patch_ids[0]` would be the id of the *second*
            # pixel, which is not necessarily part of the first patch.
            patch_ids[patch_ids == patch_ids[-1]] = 1
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
    """
    Return a mask marking points of a one-dimensional mask that are outside
    `mask` but have a neighbor inside `mask`.
    """
    p_left = np.logical_and(np.roll(mask, -1), np.logical_not(mask))
    p_right = np.logical_and(np.roll(mask, 1), np.logical_not(mask))
    if not periodic:
        p_left[-1] = False
        p_right[0] = False
    return np.logical_or(p_left, p_right)
