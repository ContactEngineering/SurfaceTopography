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
Adding and subtracting topographies, e.g. for computing gaps or the
effective roughness of two contacting surfaces
"""

import numpy as np

from ..Exceptions import ReentrantDataError
from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import Topography, UniformLineScan


def _in_unit_of(self, other):
    """Convert `other` to the unit of `self`, if both have units."""
    if self.unit is not None and other.unit is not None and other.unit != self.unit:
        return other.to_unit(self.unit)
    return other


def add_uniform(self, other, interpolate=False):
    """
    Add the heights of another topography or line scan.

    Parameters
    ----------
    other : Topography or UniformLineScan
        Topography or line scan to add. It is converted to the unit of this
        topography.
    interpolate : bool, optional
        If False, both must have the same grid and physical size. If True,
        `other` is linearly interpolated onto the grid of this topography;
        it then needs to cover the domain of this topography. (Default:
        False)

    Returns
    -------
    topography : Topography or UniformLineScan
        Sum of both, on the grid of this topography.
    """
    other = _in_unit_of(self, other)
    if interpolate:
        interpolator = other.interpolate_linear()
        if self.dim == 1:
            other = UniformLineScan(
                interpolator(self.positions()),
                self.physical_sizes,
                periodic=self.is_periodic,
                unit=self.unit,
            )
        else:
            other = Topography(
                interpolator(*self.positions()),
                self.physical_sizes,
                periodic=self.is_periodic,
                unit=self.unit,
            )
    return self.superpose(other)


def subtract_uniform(self, other, interpolate=False):
    """
    Subtract the heights of another topography or line scan, e.g. to
    compute the gap between two surfaces. See `add` for the parameters.
    """
    return add_uniform(self, other.scale(-1), interpolate=interpolate)


def add_nonuniform(self, other):
    """
    Add the heights of another line scan.

    The result is a nonuniform line scan with the positions of both line
    scans within the range where they overlap; heights are linearly
    interpolated.

    Parameters
    ----------
    other : NonuniformLineScan or UniformLineScan
        Line scan to add. It is converted to the unit of this line scan.

    Returns
    -------
    line_scan : NonuniformLineScan
        Sum of both.
    """
    if self.is_reentrant or other.is_reentrant:
        raise ReentrantDataError(
            "Line scans with overhangs (reentrant line scans) cannot be added."
        )
    other = _in_unit_of(self, other)
    x1, h1 = self.positions_and_heights()
    x2, h2 = other.positions_and_heights()
    lower, upper = max(x1[0], x2[0]), min(x1[-1], x2[-1])
    if lower >= upper:
        raise ValueError("The line scans do not overlap.")
    x = np.union1d(x1, x2)
    x = x[np.logical_and(x >= lower, x <= upper)]
    return NonuniformLineScan(
        x, np.interp(x, x1, h1) + np.interp(x, x2, h2), unit=self.unit
    )


def subtract_nonuniform(self, other):
    """
    Subtract the heights of another line scan. See `add` for details.
    """
    return add_nonuniform(self, other.scale(-1))


UniformTopographyInterface.register_function("add", add_uniform)
UniformTopographyInterface.register_function("subtract", subtract_uniform)
NonuniformLineScanInterface.register_function("add", add_nonuniform)
NonuniformLineScanInterface.register_function("subtract", subtract_nonuniform)
