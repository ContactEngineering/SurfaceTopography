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

from ..HeightContainer import NonuniformLineScanInterface, UniformTopographyInterface
from ..Support.Regression import make_grid


def scale_dependent_statistical_property(self, func, n=1, scale_factor=None, distance=None, reliable=True,
                                         interpolation='linear', progress_callback=None,
                                         collocation='log', nb_points=None, nb_points_per_decade=10):
    """
    Compute statistical properties of a uniform topography at specific scales.
    The scale is specified either by `scale_factors` or `distance`. These
    properties are statistics of derivatives carried out at specific scales,
    as computed using the `derivative` pipeline function.

    The specific statistical properties is computed by the `func` argument.
    The output of `func` needs to be homogeneous, i.e. if an array is returned,
    this array must have the same size independent of the derivative data that
    is fed into `func`.

    Parameters
    ----------
    self : Topography or UniformLineScan
        Topogaphy or line scan.
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
    scale_factor : float or np.ndarray
        Scale factor for rescaling the finite differences stencil. A scale
        factor of unity means the derivative is computed at the size of the
        individual pixel.
    distance : float or np.ndarray
        Characteristic distances at which the derivatives are computed. This
        is the overall length of the underlying stencil of lowest truncation
        order, not the effective grid spacing used by this stencil. (The
        corresponding scale factor is then given by distance / (n * px) where
        n is the order of the derivative and px the grid spacing.)
        If this is an array, then the statistical property is computed at each
        of these distances.
    reliable : bool, optional
        Only incorporate data deemed reliable. (Default: True)
    unit : str
        Unit of the distance array. All topographies are converted to this
        unit before the derivative is computed.
    interpolation : str, optional
        Interpolation method to use for computing derivatives at distances
        that do not equal an integer multiple of the grid spacing. Use
        'linear' for a local liner interpolation or 'fourier' for global
        Fourier interpolation. Note that Fourier interpolation carries large
        errors for nonperiodic topographies and should be used with care.
        (Default: 'linear')
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Automatic computation of sampling grid if neigther `scale_factor` nor
        `distance` is specified. Specifying 'log' yields collocation points
        equally spaced on a log scale, 'quadratic' yields bins with
        similar number of data points and 'linear' yields linear bins.
        (Default: 'log')
    nb_points : int, optional
        Number of points for automatic sampling grid. Bins are automatically
        determined if set to None. (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: None)

    Returns
    -------
    statistical_fingerprint : np.ndarray or list of np.ndarray
        Array containing the result of `func`

    Examples
    --------
    This example yields the the scale-dependent derivative (equivalent to
    the autocorrelation function divided by the distance) in the x-direction:

    >>> distances, A = t.autocorrelation_from_profile(resampling_method=None)
    >>> s = t.scale_dependent_statistical_property(lambda x, y=None: np.var(x), distance=distances[1::20])
    >>> np.testing.assert_allclose(2 * A[1::20] / distances[1::20] ** 2, s)
    """
    if scale_factor is None and distance is None:
        lower, upper = self.bandwidth()
        distance, _ = make_grid(collocation, n * lower, upper, nb_points=nb_points,
                                nb_points_per_decade=nb_points_per_decade)
    d = self.derivative(n=n, scale_factor=scale_factor, distance=distance, interpolation=interpolation,
                        progress_callback=progress_callback)
    if distance is None:
        distance = scale_factor * np.mean(self.physical_sizes)
    if self.dim == 1:
        try:
            if scale_factor is not None:
                iter(scale_factor)
            if distance is not None:
                iter(distance)
            retvals = np.array([func(_d) for _d in d])
        except TypeError:
            # If there is a single distance, the reliability analysis is skipped
            return distance, func(d)
    else:
        try:
            if scale_factor is not None:
                iter(scale_factor)
            if distance is not None:
                iter(distance)
            retvals = np.array([func(dx, dy) for dx, dy in zip(*d)])
        except TypeError:
            # If there is a single distance, the reliability analysis is skipped
            return distance, func(*d)

    short_cutoff = self.short_reliability_cutoff() if reliable else None
    if short_cutoff is not None:
        mask = distance > short_cutoff * n / 2
        distance = distance[mask]
        retvals = retvals[mask]

    return distance, retvals


UniformTopographyInterface.register_function(
    'scale_dependent_statistical_property', scale_dependent_statistical_property)
NonuniformLineScanInterface.register_function(
    'scale_dependent_statistical_property', scale_dependent_statistical_property)
