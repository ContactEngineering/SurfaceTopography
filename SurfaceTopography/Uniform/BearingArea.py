#
# Copyright 2024 Lars Pastewka
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
Bearing area curve, also known as Abbott-Firestone curve. The fractional
bearing area at height h is the fraction of the projected area where the
surface lies above h, i.e. the complementary cumulative distribution function
of the heights of the piecewise-linear interpolation of the data.
"""

from functools import cached_property

import _SurfaceTopography
import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction

from ..HeightContainer import UniformTopographyInterface
from .common import check_profile_decomposition


class UniformBearingArea:
    def bounds(self, heights):
        """
        Compute bounds on the bearing area for a specific height. These bounds
        can be computed in O(log(N)) time, where N is the number of elements
        (segments or triangles). The lower bound is the fraction of elements
        that lie entirely above the height, the upper bound the fraction of
        elements that have at least one point above the height.

        Parameters
        ----------
        heights : float or np.ndarray
            Heights for which to compute the bearing area.

        Returns
        -------
        lower_bound : float or np.ndarray
            Lower bound on fractional area above a threshold height.
        upper_bound : float or np.ndarray
            Upper bound on fractional area above a threshold height.
        """
        el_min = np.searchsorted(self._el_min_heights, heights)
        el_max = np.searchsorted(self._el_max_heights, heights)
        return (self._nb_els - el_min) / self._nb_els, (self._nb_els - el_max) / self._nb_els

    @cached_property
    def min(self):
        return np.nanmin(self._h)  # There are NaNs if the original topography was masked

    @cached_property
    def max(self):
        return np.nanmax(self._h)  # There are NaNs if the original topography was masked


class Uniform1DBearingArea(UniformBearingArea):
    """
    Accelerated bearing area calculation for uniform line scans.
    """

    def __init__(self, h, is_periodic):
        self._is_periodic = is_periodic

        if self._is_periodic:
            self._el_min_heights = np.sort(np.ma.compressed(np.minimum(h, np.roll(h, -1))))
            self._el_max_heights = np.sort(np.ma.compressed(np.maximum(h, np.roll(h, -1))))
        else:
            self._el_min_heights = np.sort(np.ma.compressed(np.minimum(h[:-1], h[1:])))
            self._el_max_heights = np.sort(np.ma.compressed(np.maximum(h[:-1], h[1:])))
        self._nb_els = len(self._el_min_heights)  # This treats undefined data correctly
        self._h = np.ma.filled(h.astype(float), fill_value=np.nan)  # We pass the mask as NaNs to the C++ code

    def __call__(self, heights):
        """
        Compute the bearing area for a specific height.

        The bearing area as a function of height is also known as the
        Abbott-Firestone curve. If expressed as a fractional area, it is the
        complementary cumulative distribution function of the surface heights.
        This function returns this fractional area. The topography is
        interpolated linearly between grid points (line scans: linear
        segments; maps: two triangles per grid cell) and the area above the
        threshold is computed exactly for this interpolation. Elements that
        touch undefined (masked) data are ignored.

        The function has complexity O(N) per height, where N is the number of
        elements (segments or triangles).

        Parameters
        ----------
        heights : float or np.ndarray
            Heights for which to compute the bearing area.

        Returns
        -------
        fractional_area : float or np.ndarray
            Fractional area above the threshold height.
        """
        if np.isscalar(heights):
            return _SurfaceTopography.uniform1d_bearing_area(self._h, self._is_periodic,
                                                             np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopography.uniform1d_bearing_area(self._h, self._is_periodic,
                                                             heights.astype(float))


class Uniform2DBearingArea(UniformBearingArea):
    """
    Accelerated bearing area calculation for topographies.
    """

    def __init__(self, h, is_periodic):
        self._is_periodic = is_periodic

        if self._is_periodic:
            # Note: `np.roll` needs explicit axes; with a tuple shift and no
            # axis it would flatten the array and roll by the sum of the
            # shifts.
            h10 = np.roll(h, -1, axis=0)
            h01 = np.roll(h, -1, axis=1)
            h11 = np.roll(h, (-1, -1), axis=(0, 1))
            self._el_min_heights = np.sort(np.ma.compressed([
                np.minimum(np.minimum(h, h10), h01),
                np.minimum(np.minimum(h11, h10), h01)
            ]))
            self._el_max_heights = np.sort(np.ma.compressed([
                np.maximum(np.maximum(h, h10), h01),
                np.maximum(np.maximum(h11, h10), h01)
            ]))
        else:
            self._el_min_heights = np.sort(np.ma.compressed([
                np.minimum(np.minimum(h[:-1, :-1], h[1:, :-1]), h[:-1, 1:]),
                np.minimum(np.minimum(h[1:, 1:], h[1:, :-1]), h[:-1, 1:])
            ]))
            self._el_max_heights = np.sort(np.ma.compressed([
                np.maximum(np.maximum(h[:-1, :-1], h[1:, :-1]), h[:-1, 1:]),
                np.maximum(np.maximum(h[1:, 1:], h[1:, :-1]), h[:-1, 1:])
            ]))
        self._nb_els = len(self._el_min_heights)  # This treats undefined data correctly
        # We pass the mask as NaNs to the C++ code
        self._h = np.ascontiguousarray(np.ma.filled(h, fill_value=np.nan), dtype=float)

    def __call__(self, heights):
        """
        Compute the bearing area for a specific height.

        The bearing area as a function of height is also known as the
        Abbott-Firestone curve. If expressed as a fractional area, it is the
        complementary cumulative distribution function of the surface heights.
        This function returns this fractional area. The topography is
        interpolated linearly between grid points (line scans: linear
        segments; maps: two triangles per grid cell) and the area above the
        threshold is computed exactly for this interpolation. Elements that
        touch undefined (masked) data are ignored.

        The function has complexity O(N) per height, where N is the number of
        elements (segments or triangles).

        Parameters
        ----------
        heights : float or np.ndarray
            Heights for which to compute the bearing area.

        Returns
        -------
        fractional_area : float or np.ndarray
            Fractional area above the threshold height.
        """
        if np.isscalar(heights):
            return _SurfaceTopography.uniform2d_bearing_area(self._h, self._is_periodic,
                                                             np.array([heights], dtype=float))[0]
        else:
            return _SurfaceTopography.uniform2d_bearing_area(self._h, self._is_periodic,
                                                             heights.astype(float))


class DecomposedUniform2DBearingArea:
    """
    Bearing area of a domain-decomposed (MPI-parallel) topography. Every
    process evaluates the triangles of its subdomain, including the cells
    that connect it to the next subdomain along y (whose first column is
    exchanged), and the results are summed over all processes. Only
    decompositions along y (slabs that contain all points along x) are
    supported.
    """

    def __init__(self, topography):
        self._communicator = topography.communicator
        self._reduction = Reduction(topography.communicator)
        nx, ny = topography.nb_grid_pts
        check_profile_decomposition(topography, "Bearing area calculations")
        y0, lny = topography.subdomain_locations[1], topography.nb_subdomain_grid_pts[1]
        h = np.ma.filled(np.ma.asarray(topography.heights(), dtype=float), np.nan)
        # First column of the subdomain that follows along y
        next_y0 = y0 + lny
        if topography.is_periodic:
            next_y0 %= ny
        columns = topography.communicator.allgather((y0, h[:, 0].copy()))
        for column_y0, column in columns:
            if column_y0 == next_y0:  # Never true for the last nonperiodic subdomain
                h = np.concatenate([h, column.reshape(-1, 1)], axis=1)
                break
        if topography.is_periodic:
            # Cells that connect the last and first row
            h = np.concatenate([h, h[:1]], axis=0)
        self._local = Uniform2DBearingArea(np.ma.masked_invalid(h), is_periodic=False)
        self._nb_els = self._reduction.sum(self._local._nb_els)

    def bounds(self, heights):
        """See :meth:`UniformBearingArea.bounds`."""
        el_min = self._sum(np.searchsorted(self._local._el_min_heights, heights))
        el_max = self._sum(np.searchsorted(self._local._el_max_heights, heights))
        return (self._nb_els - el_min) / self._nb_els, (self._nb_els - el_max) / self._nb_els

    def __call__(self, heights):
        """See :meth:`Uniform2DBearingArea.__call__`."""
        if self._local._nb_els > 0:
            local_area = self._local(heights) * self._local._nb_els
        else:
            local_area = np.zeros_like(heights, dtype=float)
        return self._sum(local_area) / self._nb_els

    def _sum(self, values):
        """Element-wise sum over all processes"""
        return self._communicator.allreduce(values, op=MPI.SUM)

    @cached_property
    def min(self):
        return self._reduction.min(np.nanmin(self._local._h))

    @cached_property
    def max(self):
        return self._reduction.max(np.nanmax(self._local._h))


def bearing_area(self, heights=None):
    """
    Compute the bearing area for a specific height.

    The bearing area as a function of height is also known as the
    Abbott-Firestone curve. If expressed as a fractional area, it is the
    complementary cumulative distribution function of the surface heights.
    This function returns this fractional area.

    Parameters
    ----------
    self : :obj:`UniformLineScan` or :obj:`Topography`
        Topography or line scan container object.
    heights : float or np.ndarray, optional
        Heights for which to compute the bearing area. (Default: None)

    Returns
    -------
    fractional_area : float or np.ndarray
        Fractional area above the threshold height, if `heights` is given.
    bearing_area : :obj:`Uniform1DBearingArea` or :obj:`Uniform2DBearingArea`
        Instance of a class that caches the bearing area calculation, if
        `heights` is None. The instance can be called with heights and
        provides `bounds`, `min` and `max`.
    """
    if self.dim == 1:
        b = Uniform1DBearingArea(self.heights(), self.is_periodic)
    elif self.is_domain_decomposed:
        b = DecomposedUniform2DBearingArea(self)
    elif self.dim == 2:
        b = Uniform2DBearingArea(self.heights(), self.is_periodic)
    else:
        raise NotImplementedError('Bearing area is only implemented for 1D line scans and 2D topography maps.')

    if heights is None:
        return b
    else:
        return b(heights)


UniformTopographyInterface.register_function('bearing_area', bearing_area, domain_decomposition=True)
