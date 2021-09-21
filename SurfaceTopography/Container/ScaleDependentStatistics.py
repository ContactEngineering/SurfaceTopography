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

from collections import defaultdict

import numpy as np

from .SurfaceContainer import SurfaceContainer


def scale_dependent_statistical_property(self, func, n, unit, nb_points_per_decade=10, distances=None,
                                         interpolation='linear', reliable=True, progress_callback=None):
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
    self : SurfaceContainer
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
    unit : str
        Unit of the distance array. All topographies are converted to this
        unit before the derivative is computed.
    nb_points_per_decade : int, optional
        Number of points per decade in length for automatic grid construction.
        (Default: 10)
    distances : float or np.ndarray, optional
        Characteristic distances at which the derivatives are computed. If
        this is an array, then the statistical property is computed at each
        of these distances. (Default: None)
    interpolation : str, optional
        Interpolation method to use for computing derivatives at distances
        that do not equal an integer multiple of the grid spacing. Use
        'linear' for a local liner interpolation or 'fourier' for global
        Fourier interpolation. Note that Fourier interpolation carries large
        errors for nonperiodic topographies and should be used with care.
        (Default: 'linear')
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)

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

    Note that the statistics function (the lambda expression below) needs to be
    able to accept one argument (for line scans) and two arguments
    (for topographies).

    >>> s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, distance=[0.1, 1.0, 10], unit='um')
    """
    results = defaultdict(list)
    for i, topography in enumerate(self):
        if progress_callback is not None:
            progress_callback(i, len(self))

        topography = topography.to_unit(unit)

        # Only compute the statistical property for distances that actually exist for this specific topography...
        lower, upper = topography.bandwidth()
        # ...and that are reliable
        if reliable:
            short_cutoff = topography.short_reliability_cutoff()
            if short_cutoff is None:
                short_cutoff = 0
            lower = max(short_cutoff, lower)

        if distances is None:
            # The automatic grid must include the full decades, i.e. 0.01, 0.1, 1, 10, 100, etc. as values. This
            # ensures that the grid of subsequent calculations are aligned.
            lower_decade, upper_decade = int(np.floor(np.log10(lower))), int(np.ceil(np.log10(upper)))
            existing_distances = np.logspace(lower_decade, upper_decade,
                                             (upper_decade - lower_decade) * nb_points_per_decade + 1)
            unique_distance_index = lower_decade * nb_points_per_decade + np.arange(len(existing_distances))
        else:
            existing_distances = np.array(distances)
            unique_distance_index = np.arange(len(distances))
        # For the factor n see arXiv:2106.16103
        m = np.logical_and(existing_distances > n * lower, existing_distances < upper)
        existing_distances = existing_distances[m]
        unique_distance_index = unique_distance_index[m]

        # Are there any distances left?
        if len(existing_distances) > 0:
            # Yes! Let's compute the statistical properties at these scales
            _, stat = topography.scale_dependent_statistical_property(
                func, n=n, distance=existing_distances, interpolation=interpolation)
            # Append results to our return values
            for i, e, s in zip(unique_distance_index, existing_distances, stat):
                results[i] += [(e, s)]

    if progress_callback is not None:
        progress_callback(len(self), len(self))

    if distances is not None:
        # If distances are specified by the user, we return exactly those distances with Nones where no data exists
        return distances, [np.mean(results[i], axis=0)[1] if i in results else None for i in range(len(distances))]
    else:
        # If distances are not specified by the, the distance array contains only distances where data exists
        sorted_results = sorted(results.items(), key=lambda x: x[0])
        distances, properties = np.array([np.mean(vals, axis=0) for i, vals in sorted_results]).T
        return distances, properties


SurfaceContainer.register_function('scale_dependent_statistical_property', scale_dependent_statistical_property)
