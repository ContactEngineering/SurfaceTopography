#
# Copyright 2020-2021 Lars Pastewka
#           2019-2020 Antoine Sanner
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
    """
    Compute statistical properties of a topography container (i.e. of a set of
    topographies and line scans) at specific scales. These properties are
    statistics of derivatives carried out at specific scales, as computed
    using the `derivative` pipeline function of the topography and/or line
    scan.

    The specific statistical properties is computed by the `func` argument.
    The output of `func` needs to be homogeneous, i.e. if an array is returned,
    this array must have the same size independent of the derivative data that
    is fed into `func`. The results of `func` are averaged over all topographies
    and/or line scans with each topography/line scan weighted equally. (The
    underlying assumption is that a topography/line scan represents a
    statistically independent dataset.)

    All topographies/line scans are converted to the same unit given by the
    `unit` argument before analysis.

    Parameters
    ----------
    container : SurfaceContainer
        Container object containing the underlying data sets.
    func : callable
        The function that computes the statistical properties:

            ``func(dx, dy=None) -> np.ndarray``

        A function taking the derivative in x-direction and optionally the
        derivative in y-direction (only for topographies, i.e. maps). The
        function needs to be able to ignore the second argument as a container
        can be a mixture of topographies and line scans. The function can
        return a scalar value or an array, but the array size must be fixed.
    n : int
        Order of derivative.
    distance : float or np.ndarray
        Characteristic distances at which the derivatives are computed. If
        this is an array, then the statistical property is computed at each
        of these distances.
    unit : str
        Unit of the distance array. All topographies are converted to this
        unit before the derivative is computed.

    Returns
    -------
    statistical_fingerprint : np.ndarray or list of np.ndarray
        Array containing the result of `func`, average over all topographies
        and line scans. If multiple distances are provided, then this is a
        list of arrays that contains results for each distance. If no data
        exists for a certain distance, then None is returned for that
        distance.

    Examples
    --------
    This example yields the the height-difference autocorrelation function in
    the x-direction, i.e. it should give a result identical to the average of
    `autocorrelation_from_profile` executed for all topographies/line scans in
    the container.

    >>> s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.1, 1.0, 10], unit='um')
    """
    retvals = {}
    distance = np.array(distance)
    for topography in container:
        topography = topography.to_unit(unit)

        # Only compute the statistical property for distances that actually exist for this specific topography
        l, u = topography.bandwidth()
        m = np.logical_and(distance > l, distance < u)
        existing_distances = distance[m]

        # Are there any distances left?
        if len(existing_distances) > 0:
            # Yes! Let's compute the statistical properties at these scales
            stat = topography.scale_dependent_statistical_property(func, n=n, distance=existing_distances)
            # Append results to our return values
            for e, s in zip(existing_distances, stat):
                if e in retvals:
                    retvals[e] += [s]
                else:
                    retvals[e] = [s]
    return [np.mean(retvals[d], axis=0) if d in retvals else None for d in distance]


SurfaceContainer.register_function('scale_dependent_statistical_property', scale_dependent_statistical_property)
