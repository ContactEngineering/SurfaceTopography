#
# Copyright 2020-2021 Lars Pastewka
#           2021 Paul Strauch
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

from ..Exceptions import NoReliableDataError
from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..Support import doi
from ..Support.Regression import make_grid


@doi('10.1016/j.apsadv.2021.100190')
def scale_dependent_statistical_property(self, func, n=1, scale_factor=None, distance=None, reliable=True,
                                         interpolation='linear', progress_callback=None,
                                         collocation='log', nb_points=None, nb_points_per_decade=10, threshold=4):
    r"""
    Compute statistical properties of a topography at specific scales.
    The scale is specified either by `scale_factor` or `distance`. These
    properties are statistics of derivatives carried out at specific scales,
    as computed using the `derivative` analysis function. See
    Sanner et al., Appl. Surf. Sci. Adv. 7, 100190 (2022),
    https://doi.org/10.1016/j.apsadv.2021.100190.

    The specific statistical properties is computed by the `func` argument.
    The output of `func` needs to be homogeneous, i.e. if an array is returned,
    this array must have the same size independent of the derivative data that
    is fed into `func`.

    Parameters
    ----------
    self : Topography or UniformLineScan or NonuniformLineScan
        Topography or line scan.
    func : callable
        The function that computes the statistical properties:

            ``func(dx, dy=None) -> np.ndarray``

        A function taking the derivative in x-direction and optionally the
        derivative in y-direction (only for topographies, i.e. maps). The
        function needs to be able to ignore the second argument as a container
        can be a mixture of topographies and line scans. The function can
        return a scalar value or an array, but the array size must be fixed.
    n : int, optional
        Order of derivative. (Default: 1)
    scale_factor : float or np.ndarray, optional
        Scale factor for rescaling the finite differences stencil. A scale
        factor of unity means the derivative is computed at the size of the
        individual pixel. Only supported for uniform topographies.
        (Default: None)
    distance : float or np.ndarray, optional
        Characteristic distances at which the derivatives are computed. This
        is the overall length of the underlying stencil of lowest truncation
        order, not the effective grid spacing used by this stencil. (The
        corresponding scale factor is then given by distance / (n * px) where
        n is the order of the derivative and px the grid spacing.)
        If this is an array, then the statistical property is computed at each
        of these distances. If neither `scale_factor` nor `distance` is
        given, distances are generated automatically between `n` times the
        lower and the upper bound of `bandwidth()` (see `collocation`).
        (Default: None)
    reliable : bool, optional
        Only incorporate data deemed reliable. If true and the topography
        reports a `short_reliability_cutoff` :math:`\lambda_c`, only
        distances larger than :math:`n\lambda_c/2` are returned. (The cutoff
        is the length of the second-derivative stencil, i.e. :math:`\lambda_c/2`
        is its grid spacing.) This filter is only applied if `scale_factor`
        or `distance` is an array or is generated automatically.
        (Default: True)
    interpolation : str, optional
        Interpolation method to use for computing derivatives at distances
        that do not equal an integer multiple of the grid spacing. Use
        'linear' for a local linear interpolation or 'fourier' for global
        Fourier interpolation. Note that Fourier interpolation carries large
        errors for nonperiodic topographies and should be used with care.
        (Default: 'linear')
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Automatic computation of sampling grid if neither `scale_factor` nor
        `distance` is specified. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        (Default: 'log')
    nb_points : int, optional
        Number of points for automatic sampling grid. Bins are automatically
        determined if set to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)
    threshold : int, optional
        Minimum number of derivative values required to evaluate `func`;
        derivative arrays with fewer values are replaced by np.nan before
        `func` is called. E.g. the scipy.stats.kstat function needs at least
        4 data points to calculate the 4th cumulant function, otherwise it
        returns nan or inf. (Default: 4)

    Returns
    -------
    distance : float or np.ndarray
        Distances at which the property was evaluated. (Units: length)
    statistical_fingerprint : np.ndarray
        Result of `func` for each distance.

    Raises
    ------
    NoReliableDataError
        If no distance lies above the reliability cutoff.

    Examples
    --------
    This example yields the variance of the scale-dependent slope in the
    x-direction, which equals :math:`2A(\lambda)/\lambda^2` (up to the
    squared mean slope) with the autocorrelation function :math:`A`:

    >>> distances, A = t.autocorrelation_from_profile(resampling_method=None)
    >>> d, s = t.scale_dependent_statistical_property(lambda x, y=None: np.var(x), distance=distances[1::20])
    >>> np.testing.assert_allclose(2 * A[1::20] / distances[1::20] ** 2, s)
    """
    if scale_factor is None and distance is None:
        lower, upper = self.bandwidth()
        distance, _ = make_grid(collocation, n * lower, upper, nb_points=nb_points,
                                nb_points_per_decade=nb_points_per_decade)
    d = self.derivative(n=n, scale_factor=scale_factor, distance=distance, interpolation=interpolation,
                        progress_callback=progress_callback)
    if distance is None:
        # The scale factor scales the stencil of the derivative, i.e. the
        # physical distance is scale_factor * n * pixel_size (see
        # `Uniform.Derivative.derivative`, which converts distances to scale
        # factors via scale_factor = distance / (n * px)). Note that scale
        # factors are only supported for uniform topographies, so
        # `pixel_size` is guaranteed to exist here.
        distance = np.asarray(scale_factor) * n * np.mean(self.pixel_size)
    if self.dim == 1:
        try:
            if scale_factor is not None:
                iter(scale_factor)
            if distance is not None:
                iter(distance)
            # check if the amount of data points fulfill threshold, otherwise set to nan
            for i in range(0, len(d)):
                if len(d[i]) < threshold:
                    d[i] = np.nan
            retvals = np.array([func(_d) for _d in d])
        except TypeError:
            # If there is a single distance, the reliability analysis is skipped
            if len(d) < threshold:
                d = np.nan
            return distance, func(d)
    else:
        try:
            if scale_factor is not None:
                iter(scale_factor)
            if distance is not None:
                iter(distance)
            # check if the amount of data points fulfill threshold, otherwise set to nan
            for i in range(0, len(d)):
                for j in range(0, len(d[i])):
                    if len(d[i][j].flatten()) < threshold:
                        d[i][j] = np.nan
            retvals = np.array([func(dx, dy) for dx, dy in zip(*d)])
        except TypeError:
            # If there is a single distance, the reliability analysis is skipped
            d = list(d)  # may be a tuple, which does not support item assignment
            for i in range(0, len(d)):
                if len(d[i]) < threshold:
                    d[i] = np.nan
            return distance, func(*d)

    short_cutoff = self.short_reliability_cutoff() if reliable else None
    if short_cutoff is not None:
        distance = np.asarray(distance)
        mask = distance > short_cutoff * n / 2
        if mask.sum() == 0:
            raise NoReliableDataError('Dataset contains no reliable data.')
        distance = distance[mask]
        retvals = retvals[mask]

    return distance, retvals


UniformTopographyInterface.register_function(
    'scale_dependent_statistical_property', scale_dependent_statistical_property)
NonuniformLineScanInterface.register_function(
    'scale_dependent_statistical_property', scale_dependent_statistical_property)
