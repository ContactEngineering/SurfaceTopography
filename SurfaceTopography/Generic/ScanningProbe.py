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
Analysis functions related to scanning-probe microscopy
"""

import numpy as np
import scipy

from SurfaceTopography.HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface


def scanning_probe_reliability_cutoff(self, tip_radius, safety_factor=1 / 2):
    """
    Estimate a length scale below which tip radius artifacts affect the
    measured data. See: https://arxiv.org/abs/2106.16103

    Parameters
    ----------
    self : Topography or UniformLineScan
        Topogaphy or line scan.
    tip_radius : float
        Tip radius.
    safety_factor : float, optional
        Tip radius artifacts are expected to play a role for scales below
        which the minimum scale-dependent curvature drops below
        -safety_factor/radius. The `safety_factor` should be on the order of 1.
        In https://arxiv.org/abs/2106.16103 a factor of 1/2 is estimated based
        on synthetic (simulated) data. (Default: 1/2)

    Returns
    -------
    reliability_cutoff : float
        Length scale below which data is affected by tip radius artifacts.
    """
    target_curvature = safety_factor / tip_radius

    if self.dim == 1:
        sx, = self.physical_sizes
        px, = self.pixel_size
    elif self.dim == 2:
        sx, sy = self.physical_sizes
        px, py = self.pixel_size
    else:
        raise ValueError(f"Don't know how to handle a topography of dimension {self.dim}.")

    def negative_minimum(dx, dy=None):
        return -np.min(dx)

    def objective(scale_factor):
        return target_curvature - negative_minimum(*self.derivative(n=2, scale_factor=scale_factor))

    reliability_cutoff = 2 * scipy.optimize.brentq(objective,
                                                   1, sx / 4 / px,  # bounds
                                                   xtol=1e-4
                                                   ) * px

    return reliability_cutoff


UniformTopographyInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
NonuniformLineScanInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
