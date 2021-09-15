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
Automatic determination of the reliability of the underyling data
"""

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..UnitConversion import get_unit_conversion_factor


def short_reliability_cutoff(self):
    """
    Determine down to which distance scale the data is realiable, i.e. below
    which distance it may be affected by instrumental artifacts. Currently
    supported are tip radius artifacts and a user specified resolution.

    Returns
    -------
    short_cutoff : float
        All data below this length scale is unreliable
    """
    info = self.info
    if 'instrument' in info:
        instrument = info['instrument']
        if 'tip_radius' in instrument:
            # We are carrying out a scanning probe analysis, get tip radius in correct units
            tip_radius = instrument['tip_radius']
            r = tip_radius['value'] * get_unit_conversion_factor(tip_radius['unit'], self.unit)
            return self.scanning_probe_reliability_cutoff(r)
        else:
            # Don't know what type of instrument this is and how to carry out
            # a reliability analysis
            return None
    else:
        # We cannot say anything about reliability if we do not know the
        # instrument and its parameters
        return None


UniformTopographyInterface.register_function('short_reliability_cutoff', short_reliability_cutoff)
NonuniformLineScanInterface.register_function('short_reliability_cutoff', short_reliability_cutoff)