#
# Copyright 2021, 2023 Lars Pastewka
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
from ..Support import doi


@doi('10.1016/j.apsadv.2021.100190')
def scanning_probe_reliability_cutoff(self, tip_radius, safety_factor=1 / 2, xtol=2e-12, rtol=1e-8):
    r"""
    Estimate a length scale below which tip radius artifacts affect the
    measured data.

    A rigid spherical tip of radius :math:`R` cannot resolve features with
    curvature :math:`h'' < -1/R` (sharper than the tip). The function
    determines the stencil length :math:`\lambda_c` at which the minimum of
    the scale-dependent second derivative :math:`h''(\lambda)` (computed with
    `derivative(n=2, distance=lambda)`, along x for topography maps) equals
    :math:`-f/R`, where :math:`f` is the `safety_factor`. The root is
    bracketed in :math:`[2\lambda_\text{min}, \lambda_\text{max}/2]`, where
    :math:`\lambda_\text{min}, \lambda_\text{max}` are the bounds of
    `bandwidth()`, and found with Brent's method on distances normalized by
    the geometric mean of the bandwidth.

    See Sanner et al., Appl. Surf. Sci. Adv. 7, 100190 (2022),
    https://doi.org/10.1016/j.apsadv.2021.100190.

    Parameters
    ----------
    self : Topography or UniformLineScan or NonuniformLineScan
        Topography or line scan.
    tip_radius : float
        Tip radius. (Units: length)
    safety_factor : float, optional
        Tip radius artifacts are expected to play a role for scales below
        which the minimum scale-dependent curvature drops below
        -safety_factor/radius. The `safety_factor` should be on the order of 1.
        In 10.1016/j.apsadv.2021.100190 a factor of 1/2 is estimated based
        on synthetic (simulated) data. (Default: 1/2)
    xtol : float, optional
        Absolute tolerance for bracketing search. (Default: 2e-12)
    rtol : float, optional
        Relative tolerance for bracketing search. (Default: 1e-8)

    Returns
    -------
    reliability_cutoff : float or None
        Length scale below which data is affected by tip radius artifacts.
        Note that this is the length of the stencil for the second derivative
        used to compute the reliability cutoff. None if the curvature criterion
        is fulfilled at the smallest scale (no tip artifacts detected); the
        upper bound of the bandwidth if it is violated at the largest scale
        (no reliable data). (Units: length)
    """
    lower, upper = self.bandwidth()
    # We need to normalize the bracket search to avoid numerical issues;
    # this is the geometric mean of the two bandwidth limits
    fac = np.exp((np.log(lower) + np.log(upper)) / 2)

    target_curvature = fac * safety_factor / tip_radius

    def objective(distance):
        d = self.derivative(n=2, distance=fac * distance)
        if self.dim == 1:
            if len(d) == 0:
                return target_curvature
            return target_curvature + fac * np.min(d)
        elif self.dim == 2:
            if len(d[0]) == 0:
                return target_curvature
            return target_curvature + fac * np.min(d[0])
        else:
            raise ValueError(f'Cannot handle a {self.dim}-dimensional topography.')

    if objective(2 * lower / fac) > 0:
        # Curvature is at lower end is smaller than tip curvature
        return None
    elif objective(upper / (2 * fac)) < 0:
        # Curvature at upper end is larger than tip curvature;
        # None of the data is reliable
        return upper
    else:
        return fac * scipy.optimize.brentq(objective,
                                           2 * lower / fac, upper / (2 * fac),  # bounds
                                           xtol=xtol, rtol=rtol)


UniformTopographyInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
NonuniformLineScanInterface.register_function('scanning_probe_reliability_cutoff', scanning_probe_reliability_cutoff)
