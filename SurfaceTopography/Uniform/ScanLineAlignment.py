#
# Copyright 2026 Lars Pastewka
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
Scan line alignment for AFM topography data.

This module provides functions to correct scan line artifacts common in AFM
measurements, including per-line tilt and inter-line height offsets.
"""

import numpy as np

from SurfaceTopography.UniformLineScanAndTopography import (
    DecoratedUniformTopography,
    UniformTopographyInterface,
)


class ScanLineAlignedTopography(DecoratedUniformTopography):
    """
    Remove scan line artifacts from topography.

    This correction performs two steps:
    1. Removes trends from each scan line individually by fitting and
       subtracting a polynomial of specified degree
    2. Aligns adjacent lines by adjusting their mean heights to minimize
       discontinuities between lines

    This is useful for AFM data where each scan line may have different offset
    and tilt due to scanner drift or feedback errors. Higher polynomial degrees
    can correct for scanner bow or other nonlinear distortions within each line.

    Note on periodicity: Aligned topographies will have `is_periodic` property
    set to False, since the alignment process breaks periodicity.
    """

    def __init__(self, topography, direction='x', mode='median', degree=1,
                 info={}):
        """
        Parameters
        ----------
        topography : Topography
            2D topography to align.
        direction : str, optional
            Scan line direction: 'x' (rows, default) or 'y' (columns).
            'x' means each row is a scan line (fast scan in x-direction).
            For AFM, 'x' is typically the fast scan direction.
        mode : str, optional
            Method for computing line centers during alignment:
            - 'median': Use median for robust offset alignment (default).
                        More robust against outliers and spikes.
            - 'mean': Use mean for offset alignment.
        degree : int, optional
            Degree of polynomial to fit and subtract from each line.
            - 0: Only offset (mean/median) removal per line
            - 1: Linear fit (offset + tilt) per line (default)
            - 2: Quadratic fit (offset + tilt + curvature) per line
            - Higher values for more complex corrections
        info : dict, optional
            Metadata dictionary.
        """
        if topography.dim != 2:
            raise ValueError("Scan line alignment is only supported for "
                             "2D topographies")
        if degree < 0:
            raise ValueError("Polynomial degree must be non-negative")
        super().__init__(topography, info=info)
        self._direction = direction
        self._mode = mode
        self._degree = degree
        self._line_coeffs = None  # Per-line polynomial coefficients
        self._offsets = None      # Inter-line alignment offsets

    def __getstate__(self):
        """Called for pickling the instance."""
        state = (super().__getstate__(), self._direction, self._mode,
                 self._degree, self._line_coeffs, self._offsets)
        return state

    def __setstate__(self, state):
        """Called for unpickling the instance."""
        superstate, self._direction, self._mode, self._degree, \
            self._line_coeffs, self._offsets = state
        super().__setstate__(superstate)

    @property
    def direction(self):
        """Return the scan line direction ('x' or 'y')."""
        return self._direction

    @property
    def mode(self):
        """Return the alignment mode ('median' or 'mean')."""
        return self._mode

    @property
    def degree(self):
        """Return the polynomial degree used for per-line fitting."""
        return self._degree

    @property
    def line_coeffs(self):
        """
        Return the per-line polynomial coefficients.

        Returns
        -------
        ndarray of shape (n_lines, degree+1)
            Coefficients for each line in numpy polyfit order (highest power
            first). For a polynomial p(t) = c[0]*t^n + c[1]*t^(n-1) + ... + c[n],
            where t is the normalized position along the line (0 to 1).
        """
        if self._line_coeffs is None:
            # Trigger computation
            self.heights()
        return self._line_coeffs

    @property
    def offsets(self):
        """
        Return the inter-line alignment offsets.

        Returns
        -------
        ndarray of shape (n_lines,)
            Offset applied to each line to align adjacent lines.
        """
        if self._offsets is None:
            # Trigger computation
            self.heights()
        return self._offsets

    @property
    def is_periodic(self):
        """
        Scan line alignment breaks periodicity.
        """
        return False

    def _compute_alignment(self, heights):
        """
        Compute per-line polynomial coefficients and inter-line offsets.

        Parameters
        ----------
        heights : ndarray
            2D array of height values.
        """
        # Transpose if aligning columns instead of rows
        if self._direction == 'y':
            heights = heights.T

        nx, ny = heights.shape
        t = np.arange(ny) / ny  # Normalized position along line (0 to 1)

        # Step 1: Fit polynomial to each line
        self._line_coeffs = np.zeros((nx, self._degree + 1))
        line_detrended = np.zeros_like(heights)

        for i in range(nx):
            line = heights[i, :]
            if np.ma.is_masked(line):
                mask = ~np.ma.getmaskarray(line)
                n_valid = mask.sum()
                if n_valid > self._degree:
                    # Enough points for polynomial fit
                    coeffs = np.polyfit(t[mask], line[mask], self._degree)
                elif n_valid > 0:
                    # Not enough points for full fit - use lower degree or constant
                    actual_degree = max(0, n_valid - 1)
                    coeffs_low = np.polyfit(t[mask], line[mask], actual_degree)
                    # Pad with zeros to match expected shape
                    coeffs = np.zeros(self._degree + 1)
                    coeffs[-(actual_degree + 1):] = coeffs_low
                else:
                    # No valid points
                    coeffs = np.zeros(self._degree + 1)
            else:
                coeffs = np.polyfit(t, line, self._degree)

            # Store coefficients in numpy polyfit order (highest power first)
            self._line_coeffs[i] = coeffs
            line_detrended[i, :] = heights[i, :] - np.polyval(coeffs, t)

        # Step 2: Compute inter-line offsets to minimize discontinuities
        if self._mode == 'median':
            line_centers = np.array([np.ma.median(line_detrended[i, :])
                                     for i in range(nx)])
        else:  # 'mean'
            line_centers = np.array([np.ma.mean(line_detrended[i, :])
                                     for i in range(nx)])

        # Cumulative offset adjustment to align adjacent lines
        # Start with first line at offset 0, then adjust each subsequent line
        # to match the previous line's center
        self._offsets = np.zeros(nx)
        for i in range(1, nx):
            self._offsets[i] = (self._offsets[i - 1] +
                                (line_centers[i - 1] - line_centers[i]))

        # Center the result (subtract mean offset so overall mean is ~0)
        self._offsets -= np.mean(self._offsets)

    def heights(self):
        """
        Compute aligned heights.

        Returns
        -------
        ndarray
            2D array of aligned height values with per-line polynomial removed
            and adjacent lines aligned.
        """
        heights = self.parent_topography.heights().copy()

        if self._line_coeffs is None:
            self._compute_alignment(heights)

        # Transpose if aligning columns instead of rows
        if self._direction == 'y':
            heights = heights.T

        nx, ny = heights.shape
        t = np.arange(ny) / ny  # Normalized position along line

        # Apply per-line detrending using polynomial evaluation
        for i in range(nx):
            heights[i, :] = heights[i, :] - np.polyval(self._line_coeffs[i], t)

        # Apply inter-line offset alignment
        for i in range(nx):
            heights[i, :] += self._offsets[i]

        # Transpose back if we transposed earlier
        if self._direction == 'y':
            heights = heights.T

        return heights


UniformTopographyInterface.register_function('scan_line_align',
                                             ScanLineAlignedTopography)
