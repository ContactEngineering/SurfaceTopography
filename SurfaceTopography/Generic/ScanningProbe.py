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

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface


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
        Note that this is the length of the stencil for the second derivative
        used to compute the reliability cutoff.
    """
    target_curvature = safety_factor / tip_radius

    def objective(distance):
        d = self.derivative(n=2, distance=distance)
        if self.dim == 1:
            return target_curvature + np.min(d)
        elif self.dim == 2:
            return target_curvature + np.min(d[0])
        else:
            raise ValueError(f'Cannot handle a {self.dim}-dimensional topography.')

    lower, upper = self.bandwidth()
    if objective(2 * lower) > 0:
        # Curvature is at lower end is smaller than tip curvature
        return None
    elif objective(upper / 2) < 0:
        # Curvature at upper end is larger than tip curvature;
        # None of the data is reliable
        return upper
    else:
        return scipy.optimize.brentq(objective,
                                     2 * lower, upper / 2,  # bounds
                                     xtol=1e-4)


UniformTopographyInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
NonuniformLineScanInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
