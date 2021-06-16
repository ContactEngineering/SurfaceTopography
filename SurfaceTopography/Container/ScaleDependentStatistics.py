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

import numpy as np

from .SurfaceContainer import SurfaceContainer


def scale_dependent_statistical_property(container, func, n, distance, unit):
    retvals = {}
    distance = np.array(distance)
    for topography in container:
        topography = topography.scale(unit=unit)

        # Only compute the statistical property for distances that actually exist for this specific topography
        l, u = topography.bandwidth()
        m = np.logical_and(distance > l, distance < u)
        existing_distances = distance[m]

        # Are there any distances left?
        if len(existing_distances) > 0:
            # Yes! Let's compute the derivative at these scales
            derivatives = topography.derivative(n=n, distance=existing_distances)
            if topography.dim == 1:
                stat = [func(dx) for dx in derivatives]
            else:
                stat = [func(dx, dy) for dx, dy in zip(*derivatives)]
            for e, s in zip(existing_distances, stat):
                if e in retvals:
                    retvals[e] += [s]
                else:
                    retvals[e] = [s]
    return [np.mean(retvals[d], axis=0) for d in distance]


SurfaceContainer.register_function('scale_dependent_statistical_property', scale_dependent_statistical_property)
