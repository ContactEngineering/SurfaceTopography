#
# Copyright 2018-2021 Lars Pastewka
#           2019 Antoine Sanner
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
Height-difference autocorrelation functions
"""

import numpy as np

from ..Exceptions import NoReliableDataError
from ..HeightContainer import UniformTopographyInterface
from ..Support import doi, fold_fft_half
from ..Support.Regression import resample, resample_radial


def _masked_height_difference_correlation(heights, periodic, axes):
    r"""
    Sum of squared height differences and number of pairs of defined points
    as a function of the distance vector, for heights with undefined (masked)
    points.

    With :math:`w(\mathbf x) = 1` for defined and :math:`0` for undefined
    points, :math:`g = w h` and :math:`s = w h^2`, the sum over all pairs of
    defined points at distance :math:`\mathbf d` is

    .. math::

        \frac{1}{2}\sum_{\mathbf x} w(\mathbf x) w(\mathbf x + \mathbf d)
        \left[h(\mathbf x + \mathbf d) - h(\mathbf x)\right]^2
        = \frac{1}{2}\left[c_{ws}(\mathbf d) + c_{ws}(-\mathbf d)\right]
        - c_{gg}(\mathbf d),

    with the correlation :math:`c_{ab}(\mathbf d) = \sum_{\mathbf x}
    a(\mathbf x) b(\mathbf x + \mathbf d)`, and the number of pairs is
    :math:`c_{ww}(\mathbf d)`. The correlations are computed with three
    forward and two inverse real FFTs (circular for periodic data,
    zero-padded to :math:`2N-1` points along each axis otherwise). The mean
    of the defined heights is subtracted first to reduce round-off.

    Parameters
    ----------
    heights : np.ma.MaskedArray
        Heights with undefined points masked.
    periodic : bool
        Whether the heights are periodic.
    axes : tuple of int
        Axes along which the distance vector is taken.

    Returns
    -------
    squared_differences : np.ndarray
        Half the sum of squared height differences for each distance vector,
        in the layout of the FFT output.
    nb_pairs : np.ndarray
        Number of pairs of defined points for each distance vector.
    """
    mask = np.ma.getmaskarray(heights)
    weights = (~mask).astype(float)
    h = np.ma.getdata(heights).astype(float)
    g = np.where(mask, 0.0, h - np.mean(h[~mask]))
    shape = [heights.shape[a] if periodic else 2 * heights.shape[a] - 1 for a in axes]
    w_q = np.fft.rfftn(weights, s=shape, axes=axes)
    g_q = np.fft.rfftn(g, s=shape, axes=axes)
    s_q = np.fft.rfftn(g ** 2, s=shape, axes=axes)
    # The real part of the cross spectrum symmetrizes the correlation,
    # Re(conj(w_q) s_q) -> [c_ws(d) + c_ws(-d)] / 2
    squared_differences = np.fft.irfftn((np.conj(w_q) * s_q).real - np.abs(g_q) ** 2, s=shape, axes=axes)
    nb_pairs = np.round(np.fft.irfftn(np.abs(w_q) ** 2, s=shape, axes=axes))
    return squared_differences, nb_pairs


def _masked_profile_autocorrelation(heights, periodic):
    """
    Height-difference autocorrelation along the first axis of heights with
    undefined points, summed over all profiles. Returns the (masked) ACF for
    distances 0 to N/2 - 1 (periodic) or 0 to N - 1 (nonperiodic) grid
    points; distances without pairs of defined points are masked.
    """
    n = heights.shape[0]
    squared_differences, nb_pairs = _masked_height_difference_correlation(heights, periodic, axes=(0,))
    if periodic:
        squared_differences = fold_fft_half(squared_differences, n)
        nb_pairs = fold_fft_half(nb_pairs, n)
    else:
        squared_differences = squared_differences[:n]
        nb_pairs = nb_pairs[:n]
    if squared_differences.ndim == 2:
        # Average over the pairs of all profiles
        squared_differences = squared_differences.sum(axis=1)
        nb_pairs = nb_pairs.sum(axis=1)
    undefined = nb_pairs < 0.5
    return np.ma.masked_array(squared_differences / np.where(undefined, 1, nb_pairs), mask=undefined)


@doi('10.1016/j.triboint.2018.02.002')
def autocorrelation_from_profile(self, reliable=True, resampling_method='bin-average', collocation='log',
                                 nb_points=None, nb_points_per_decade=10):
    r"""
    Compute the one-dimensional height-difference autocorrelation function
    (ACF)

    .. math::

        A(r) = \frac{1}{2}\left\langle \left[h(x+r) - h(x)\right]^2
        \right\rangle

    of a line scan, or the average of the ACFs of all profiles (along
    :math:`x`) of a topography map.

    For periodic topographies, the average runs over all :math:`N` grid
    points and the ACF is obtained from the circular height-height
    correlation,
    :math:`A(r_k) = \langle h^2\rangle - \langle h(x)\,h(x+r_k)\rangle`,
    which is computed with the FFT. The values at :math:`r_k` and
    :math:`L - r_k` are averaged and :math:`0 \le r_k < L/2` is returned.

    For nonperiodic topographies, the average at distance
    :math:`r_k = k\Delta x` runs over the :math:`N-k` pairs of points that
    lie within the scan,

    .. math::

        A(r_k) = \frac{1}{2(N-k)} \sum_{i=0}^{N-k-1} \left(h_i - h_{i+k}\right)^2
        = \frac{1}{2(N-k)} \sum_{i=0}^{N-k-1} \left(h_i^2 + h_{i+k}^2\right)
        - \frac{1}{N-k} \sum_{i=0}^{N-k-1} h_i h_{i+k}.

    The last sum is computed with an FFT of the heights zero-padded to
    :math:`2N-1` points, the first from cumulative sums of :math:`h_i^2`.
    Distances :math:`0 \le r_k < L` are returned.

    For topographies with undefined (masked) points, the average runs over
    all pairs of defined points at distance :math:`r_k`, in all profiles;
    the sums are computed with FFTs of the heights and of the mask (about
    2.5 times the cost of the computation without undefined points).
    Distances without such pairs are masked.

    Parameters
    ----------
    self : :obj:`Topography` or :obj:`UniformLineScan`
        Container storing the uniform topography map or line scan.
    reliable : bool, optional
        Only return data deemed reliable, i.e. distances
        :math:`r > \lambda_c/2`, where :math:`\lambda_c` is the short
        reliability cutoff. (Default: True)
    resampling_method : {None, 'bin-average'}, optional
        None returns the ACF on the distances of the grid, 'bin-average'
        averages it within bins. (Default: 'bin-average')
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. 'log' yields collocation points equally spaced on
        a log scale, 'quadratic' yields bins equally spaced in :math:`r^2`
        and 'linear' yields linear bins. An array specifies the bin edges
        explicitly. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. For 'log' collocation, the number of
        bins is determined from `nb_points_per_decade` if set to None;
        'linear' and 'quadratic' collocation require this argument.
        (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 10)

    Returns
    -------
    r : np.ndarray or np.ma.MaskedArray
        Distances. Bins without data are masked for 'bin-average'.
        (Units: length)
    A : np.ndarray or np.ma.MaskedArray
        Height-difference autocorrelation function. (Units: length²)

    References
    ----------
    A. Wang, M. H. Müser, Tribol. Int. 123, 224 (2018),
    https://doi.org/10.1016/j.triboint.2018.02.002
    """  # noqa: E501
    try:
        nx, ny = self.nb_grid_pts
        sx, sy = self.physical_sizes
    except ValueError:
        nx, = self.nb_grid_pts
        sx, = self.physical_sizes

    p = self.heights()

    if self.has_undefined_data:
        # Profiles are combined into a single ACF (averaged over pairs)
        A = _masked_profile_autocorrelation(p, self.is_periodic)
        if self.is_periodic:
            r = sx * np.arange(nx // 2) / nx
            max_distance = sx / 2
        else:
            r = sx * np.arange(nx) / nx
            max_distance = sx
    elif self.is_periodic:
        # Compute height-height autocorrelation function from a convolution
        # using FFT. This is periodic by nature.
        surface_qy = np.fft.fft(p, axis=0)
        C_qy = abs(surface_qy) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft(C_qy, axis=0).real / nx

        # Convert height-height autocorrelation to height-difference
        # autocorrelation:
        #     <(h(x) - h(x+d))^2>/2 = <h^2(x)> - <h(x)h(x+d)>
        A_xy = A_xy[0] - A_xy

        A = fold_fft_half(A_xy, nx)

        r = sx * np.arange(nx // 2) / nx
        max_distance = sx / 2
    else:
        # Compute height-height autocorrelation function. We need to zero
        # pad the FFT for the nonperiodic case in order to separate images.
        surface_qy = np.fft.fft(p, n=2 * nx - 1, axis=0)
        C_qy = abs(surface_qy) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft(C_qy, axis=0).real

        # Correction to turn height-height into height-difference
        # autocorrelation:
        #     <(h(x) - h(x+d))^2>_d/2 = <h^2(x)>_d - <h(x)h(x+d)>_d
        # but we need to take care about h_rms^2=<h^2(x)>, which in the
        # nonperiodic case needs to be computed only over a subsection of
        # the surface. This is because the average < >_d now depends on d,
        # which determines the number of data points that are actually
        # included into the computation of <h(x)h(x+d)>_d. h_rms^2 needs to
        # be computed over the same data points.
        p_sq = p ** 2
        A0_xy = (p_sq.cumsum(axis=0)[::-1] + p_sq[::-1].cumsum(axis=0)[::-1]) / 2

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A = ((A0_xy - A_xy[:nx]).T / (nx - np.arange(nx))).T

        r = sx * np.arange(nx) / nx
        max_distance = sx

    # The factor of two comes from the fact that the short cutoff is estimated
    # from the curvature but the ACF is the slope, see 10.1016/j.apsadv.2021.100190
    if resampling_method is None:
        short_cutoff = self.short_reliability_cutoff() if reliable else None
        if short_cutoff is not None:
            mask = r > short_cutoff / 2
            if mask.sum() == 0:
                raise NoReliableDataError('Dataset contains no reliable data.')
            r = r[mask]
            A = A[mask]
        if A.ndim == 2:
            return r, A.mean(axis=1)
        else:
            return r, A
    else:
        short_cutoff = self.short_reliability_cutoff(2 * sx / nx) if reliable else 2 * sx / nx
        if isinstance(collocation, str) and collocation == 'log':
            # Exclude zero distance because that does not work on a log-scale
            r = r[1:]
            A = A[1:]
        if A.ndim == 2:
            r = np.resize(r, (A.shape[1], r.shape[0])).T.ravel()
            A = np.ravel(A)
        if np.ma.isMaskedArray(A):
            # Distances without pairs of defined points
            defined = ~np.ma.getmaskarray(A)
            r = r[defined]
            A = np.ma.getdata(A)[defined]
        r, _, A, _ = resample(r, A, min_value=short_cutoff / 2, max_value=max_distance, collocation=collocation,
                              nb_points=nb_points, nb_points_per_decade=nb_points_per_decade, method=resampling_method)
        return r, A


@doi('10.1016/j.triboint.2018.02.002')
def autocorrelation_from_area(self, reliable=True, collocation='log', nb_points=None, nb_points_per_decade=5,
                              return_map=False, resampling_method='bin-average'):
    r"""
    Compute the two-dimensional height-difference autocorrelation function
    :math:`A(\mathbf r) = \frac{1}{2}\langle[h(\mathbf x + \mathbf r) -
    h(\mathbf x)]^2\rangle` of a topography map and average it radially.

    The map :math:`A(\mathbf r)` is computed with 2D FFTs, as described
    for :func:`autocorrelation_from_profile`: circular for periodic
    topographies, and zero-padded to :math:`(2N_x-1)\times(2N_y-1)` points
    with normalization by the number :math:`(N_x-k)(N_y-l)` of overlapping
    pairs for nonperiodic topographies. Only distance vectors with
    nonnegative components are available in the nonperiodic case. For
    topographies with undefined (masked) points, the average runs over all
    pairs of defined points (see :func:`autocorrelation_from_profile`);
    distance vectors without such pairs are masked in the map and excluded
    from the radial average. Radial
    bins extend from half the short reliability cutoff (at least one pixel)
    to :math:`(L_x+L_y)/4` (periodic) or :math:`(L_x+L_y)/2`
    (nonperiodic).

    Parameters
    ----------
    self : :obj:`Topography`
        Container storing the (two-dimensional) topography map.
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    collocation : {'log', 'quadratic', 'linear', array_like}, optional
        Resampling grid. 'log' yields collocation points equally spaced on
        a log scale, 'quadratic' yields bins equally spaced in :math:`r^2`
        (similar number of grid points per bin) and 'linear' yields linear
        bins. An array specifies the bin edges explicitly. (Default: 'log')
    nb_points : int, optional
        Number of bins for averaging. For 'log' collocation, the number of
        bins is determined from `nb_points_per_decade` if set to None;
        'linear' and 'quadratic' collocation require this argument.
        (Default: None)
    nb_points_per_decade : int, optional
        Number of points per decade for log-spaced collocation points.
        (Default: 5)
    return_map : bool, optional
        Return full 2D autocorrelation map. (Default: False)
    resampling_method : {'bin-average'}, optional
        'bin-average' for averaging within bins. (Default: 'bin-average')

    Returns
    -------
    r : np.ndarray or np.ma.MaskedArray
        Distances. Bins without data are masked for 'bin-average'.
        (Units: length)
    A : np.ndarray or np.ma.MaskedArray
        Radially averaged autocorrelation function. (Units: length²)
    A_xy : np.ndarray
        2D autocorrelation function. Only returned if `return_map` is True.
        (Units: length²)

    References
    ----------
    A. Wang, M. H. Müser, Tribol. Int. 123, 224 (2018),
    https://doi.org/10.1016/j.triboint.2018.02.002
    """
    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes

    # The factor of two (below, in the `min_radius` arguments) comes from the
    # fact that the short cutoff is estimated from the curvature but the ACF
    # is the slope, see 10.1016/j.apsadv.2021.100190. The lower bound of two
    # pixels mirrors the profile ACF (`autocorrelation_from_profile`), which
    # uses the same convention.
    short_cutoff = self.short_reliability_cutoff(2 * np.mean(self.pixel_size)) if reliable \
        else 2 * np.mean(self.pixel_size)

    # Compute FFT and normalize
    if self.has_undefined_data:
        squared_differences, nb_pairs = _masked_height_difference_correlation(self.heights(), self.is_periodic,
                                                                              axes=(0, 1))
        if not self.is_periodic:
            # Only distance vectors with nonnegative components, as below
            squared_differences = squared_differences[:nx, :ny]
            nb_pairs = nb_pairs[:nx, :ny]
        undefined = nb_pairs < 0.5
        A_xy = np.ma.masked_array(squared_differences / np.where(undefined, 1, nb_pairs), mask=undefined)

        # Radial average
        r_val, r_edges, A_val, _ = resample_radial(A_xy, physical_sizes=(sx, sy), collocation=collocation,
                                                   nb_points=nb_points, nb_points_per_decade=nb_points_per_decade,
                                                   min_radius=short_cutoff / 2,
                                                   max_radius=(sx + sy) / 4 if self.is_periodic else (sx + sy) / 2,
                                                   full=self.is_periodic, method=resampling_method)
    elif self.is_periodic:
        surface_qk = np.fft.fft2(self[...])
        C_qk = abs(surface_qk) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft2(C_qk).real / (nx * ny)

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = A_xy[0, 0] - A_xy

        # Radial average
        r_val, r_edges, A_val, _ = resample_radial(A_xy, physical_sizes=(sx, sy), nb_points=nb_points,
                                                   nb_points_per_decade=nb_points_per_decade, collocation=collocation,
                                                   min_radius=short_cutoff / 2, max_radius=(sx + sy) / 4,
                                                   method=resampling_method)
    else:
        p = self.heights()

        # Compute height-height autocorrelation function
        surface_qk = np.fft.fft2(p, s=(2 * nx - 1, 2 * ny - 1))
        C_qk = abs(surface_qk) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft2(C_qk).real

        # Correction to turn height-height into height-difference
        # autocorrelation
        p_sq = p ** 2
        A0_xy = (p_sq.cumsum(axis=0).cumsum(axis=1)[::-1, ::-1] +
                 p_sq[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]) / 2

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = (A0_xy - A_xy[:nx, :ny]) / (
                (nx - np.arange(nx)).reshape(-1, 1) * (ny - np.arange(ny)).reshape(1, -1))

        # Radial average
        r_val, r_edges, A_val, _ = resample_radial(A_xy, physical_sizes=(sx, sy), collocation=collocation,
                                                   nb_points=nb_points, nb_points_per_decade=nb_points_per_decade,
                                                   min_radius=short_cutoff / 2, max_radius=(sx + sy) / 2, full=False,
                                                   method=resampling_method)

    if return_map:
        return r_val, A_val, A_xy
    else:
        return r_val, A_val


# Register analysis functions from this module
UniformTopographyInterface.register_function('autocorrelation_from_profile', autocorrelation_from_profile)
UniformTopographyInterface.register_function('autocorrelation_from_area', autocorrelation_from_area)
