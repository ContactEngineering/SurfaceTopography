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
Analysis of island geometries
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


def inner_perimeter_area(c, periodic, stencil=nn_stencil):
    """
    Return a map where surface points on the inner perimeter are marked.
    """

    return np.logical_and(c, coordination(c, periodic, stencil=stencil) < len(stencil))


def patch_areas(patch_ids):
    """
    Return a list containing patch areas
    """
    return np.bincount(patch_ids.reshape((-1,)))[1:]


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
