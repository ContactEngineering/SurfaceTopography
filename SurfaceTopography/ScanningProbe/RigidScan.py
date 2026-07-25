#
# Copyright 2024 Lars Pastewka
#           2023 Antoine Sanner
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

import numpy as np

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import UniformLineScan


def scan_with_rigid_sphere(topography, radius):
    """
    Scan a topography with a rigid, spherical tip. This emulated scanning of
    a physical topography in an scanning probe instrument.

    Paramaters
    ----------
    topography : :obj:`SurfaceTopography.UniformLineScan` or :obj:`SurfaceTopography.NonuniformLineScan`
        Topography to be scanned.
    radius : float
        Tip radius.

    Returns
    -------
    scanned_heights : np.ndarray
         Scannned heights of topography on the same grid as the topography.
    """
    if topography.dim != 1:
        raise ValueError("Only one-dimensional scans are supported at present.")

    positions, heights = topography.positions_and_heights()
    nb_pts = len(positions)

    # For each scan position, the tip contacts the data points within
    # [x - radius, x + radius]; the scanned height is the maximum over that
    # window of the height plus the local tip profile.
    lefts = np.searchsorted(positions, positions - radius)
    rights = np.searchsorted(positions, positions + radius)
    widths = rights - lefts

    # The windows are evaluated block-wise: a block of scan positions is
    # padded to its widest window and reduced with a masked max. The block
    # size is chosen such that the temporary array stays below a fixed
    # element count, so memory use is bounded irrespective of scan size
    # and tip radius (a single scan-sized block could otherwise allocate
    # nb_pts * max_window elements).
    target_nb_elements = 2 ** 22  # 32 MB of doubles
    scanned_heights = np.empty(nb_pts)
    start = 0
    while start < nb_pts:
        nb_block = min(nb_pts - start,
                       max(1, target_nb_elements // max(1, widths[start])))
        max_width = widths[start:start + nb_block].max()
        # Shrinking the block can only shrink its widest window, so after
        # this step nb_block * max_width <= target_nb_elements holds
        nb_block = min(nb_block, max(1, target_nb_elements // max_width))
        max_width = widths[start:start + nb_block].max()

        block = slice(start, start + nb_block)
        col = lefts[block].reshape(-1, 1) + np.arange(max_width).reshape(1, -1)
        valid = col < rights[block].reshape(-1, 1)
        col = np.minimum(col, nb_pts - 1)
        distance = positions[col] - positions[block].reshape(-1, 1)
        tip_heights = heights[col] + np.sqrt(
            np.maximum(radius ** 2 - distance * distance, 0)) - radius
        scanned_heights[block] = np.max(
            np.where(valid, tip_heights, -np.inf), axis=1)
        start += nb_block

    return scanned_heights


def pipeline_scan_with_rigid_sphere(self, radius):
    r"""
    Scan the topography with a rigid, spherical tip. This emulated scanning of
    a physical topography in a scanning probe instrument.


    Paramaters
    ----------
    radius : float
        Tip radius, in the same units as the topography

    Returns
    -------
    topography : :obj:`SurfaceTopography.UniformLineScan` or :obj:`SurfaceTopography.NonuniformLineScan`
         Topography with scannned heights on the same grid as the topography.
    """
    if self.dim != 1:
        raise ValueError(
            "Scanning with a rigid sphere is only supported for line scans."
        )
    info_dict = dict(
        instrument=dict(
            name="Scanning rigid sphere simulation",
            parameters=dict(tip_radius=dict(value=radius, unit=self.unit)),
        )
    )
    if self.is_periodic:
        extended_topography = UniformLineScan(
            np.concatenate([self.heights(), self.heights(), self.heights()]),
            self.physical_sizes[0] * 3,
        )
        scanned_heights = scan_with_rigid_sphere(extended_topography, radius)
        return UniformLineScan(
            scanned_heights[self.nb_grid_pts[0]: (self.nb_grid_pts[0] * 2)],
            self.physical_sizes,
            periodic=True,
            unit=self.unit,
            info=info_dict,
        )
    elif self.is_uniform:
        # Note: structural check rather than isinstance, so that decorated
        # line scans (detrended, scaled, ...) can be scanned as well
        scanned_heights = scan_with_rigid_sphere(self, radius)
        return UniformLineScan(
            scanned_heights,
            self.physical_sizes,
            periodic=False,
            unit=self.unit,
            info=info_dict,
        )
    else:
        scanned_heights = scan_with_rigid_sphere(self, radius)
        return NonuniformLineScan(
            self.positions(), scanned_heights, unit=self.unit, info=info_dict
        )


# Register on the interfaces so that decorated topographies (detrended,
# scaled, etc.) can also be scanned; the function itself checks that the
# data is one-dimensional
UniformTopographyInterface.register_function(
    "scan_with_rigid_sphere", pipeline_scan_with_rigid_sphere
)
NonuniformLineScanInterface.register_function(
    "scan_with_rigid_sphere", pipeline_scan_with_rigid_sphere
)
