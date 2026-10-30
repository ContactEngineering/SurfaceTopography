#
# Copyright 2016, 2018-2020, 2024 Lars Pastewka
#           2019 Antoine Sanner
#           2019 Michael Röttger
#           2015-2016 Till Junge
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

r"""
Polynomial trends of uniform line scans and topographies and the `detrend`
pipeline step that removes them.

Trends are polynomials in the normalized coordinates :math:`\xi = x/L_x`
and :math:`\eta = y/L_y`, i.e. :math:`\xi_i = i/N_x` on the grid. The
coefficients therefore all carry the unit of the heights.
"""

import numpy as np

from SurfaceTopography.UniformLineScanAndTopography import (
    DecoratedUniformTopography, UniformTopographyInterface)


def polyfit_line_scan(self, deg):
    r"""
    Compute the polynomial that, if subtracted, minimizes the rms height of a
    uniform line scan.

    Solves the discrete least-squares problem

    .. math::

        \min_{\{a_k\}} \sum_i \left( h_i - \sum_{k=0}^n a_k \xi_i^k \right)^2

    with :math:`\xi_i = x_i / L` over all defined (unmasked) data points.

    Parameters
    ----------
    self : :obj:`UniformLineScan`
        Line scan container object.
    deg : int
        Degree :math:`n` of the polynomial.

    Returns
    -------
    a : np.ndarray
        Coefficients :math:`a_0, \dots, a_n` (lowest order first, in units
        of the heights).
    """
    x, h = self.positions_and_heights()
    x /= self.physical_sizes[0]
    # Exclude undefined (masked) data points from the fit; np.polyfit on a
    # masked array would silently fit the raw values under the mask
    mask = np.ma.getmaskarray(h)
    if mask.any():
        x = x[~mask]
        h = np.asarray(h[~mask])
    coeffs = np.polyfit(x, h, deg)
    return np.array(coeffs)[::-1]


def polyfit1_topography(self, full_output=False):
    r"""
    Compute the tilt plane that, if subtracted, minimizes the rms height of
    the surface. The tilt plane is parameterized as

    .. math::

        p(\xi, \eta) = h_0 + m \xi + n \eta

    with normalized coordinates :math:`\xi = x/L_x`, :math:`\eta = y/L_y`.
    The coefficients solve the linear least-squares problem
    :math:`\min \sum_{ij} (h_{ij} - p(\xi_i, \eta_j))^2` over all defined
    (unmasked) data points.

    Parameters
    ----------
    self : :obj:`Topography`
        Height information.
    full_output : bool, optional
        If True, also return the design matrix of the least-squares
        problem. (Default: False)

    Returns
    -------
    coeffs : np.ndarray
        Coefficients :math:`h_0, m, n` (in this order and in units of the
        heights; the slopes are :math:`m/L_x` and :math:`n/L_y`).
    location_matrix : np.ndarray
        Design matrix, only returned if `full_output` is True.
    """
    arr = self.heights()
    nb_dim = self.dim
    x_grids = (np.arange(n) / n for n in self.nb_grid_pts)
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    # linear regression model
    location_matrix = np.hstack([np.ones_like(columns[-1])] + columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def polyfit2_topography(self, full_output=False):
    r"""
    Compute the second-order polynomial that, if subtracted, minimizes the
    rms height of the surface,

    .. math::

        p(\xi, \eta) = a_0 + a_1 \xi + a_2 \eta + a_3 \xi^2 + a_4 \eta^2 + a_5 \xi \eta

    with normalized coordinates :math:`\xi = x/L_x`, :math:`\eta = y/L_y`.
    The coefficients solve the linear least-squares problem
    :math:`\min \sum_{ij} (h_{ij} - p(\xi_i, \eta_j))^2` over all defined
    (unmasked) data points.

    Parameters
    ----------
    self : :obj:`Topography`
        Height information.
    full_output : bool, optional
        If True, also return the design matrix of the least-squares
        problem. (Default: False)

    Returns
    -------
    coeffs : np.ndarray
        Coefficients :math:`a_0, \dots, a_5` in the order given above (in
        units of the heights).
    location_matrix : np.ndarray
        Design matrix, only returned if `full_output` is True.
    """
    arr = self.heights()
    nb_dim = self.dim
    assert nb_dim == 2
    x_grids = (np.arange(n) / n for n in self.nb_grid_pts)
    # Linear terms
    x_grids = np.meshgrid(*x_grids, indexing='ij')
    # Quadratic terms
    x, y = x_grids
    x_grids += (x * x, y * y, x * y)
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    # linear regression model
    location_matrix = np.hstack([np.ones_like(columns[-1])] + columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def polyfit_topography(self, deg):
    """
    Compute the detrending polynomial that, if subtracted, minimizes the rms
    height of a topography. See `polyfit1_topography` and
    `polyfit2_topography`.

    Parameters
    ----------
    self : :obj:`Topography`
        Topography container object.
    deg : int
        Degree of the polynomial, 1 or 2.

    Returns
    -------
    coeffs : np.ndarray
        Three (`deg` = 1) or six (`deg` = 2) coefficients.
    """
    if deg == 1:
        return polyfit1_topography(self)
    elif deg == 2:
        return polyfit2_topography(self)
    else:
        raise NotImplementedError(
            f"Fitting detrending planes is only supported for polynomials of degree 1 and 2, not {deg}.")


def polyfit(self, deg):
    """
    Compute the polynomial that, if subtracted, minimizes the rms height of
    a uniform line scan or topography. Positions enter as the normalized
    coordinates :math:`x/L_x` (and :math:`y/L_y`).

    Parameters
    ----------
    self : :obj:`UniformLineScan` or :obj:`Topography`
        Container object.
    deg : int
        Degree of the polynomial. Arbitrary for line scans, 1 or 2 for
        topographies.

    Returns
    -------
    coeffs : np.ndarray
        Coefficients, lowest order first. For topographies see
        `polyfit1_topography` and `polyfit2_topography` for the order.
    """
    if self.dim == 1:
        return polyfit_line_scan(self, deg)
    elif self.dim == 2:
        return polyfit_topography(self, deg)
    else:
        raise RuntimeError(f'Unknown dimension {self.dim}.')


def _detrend_1d_slope_coeffs(self):
    """
    Coefficients [a0, a1] of the line whose slope is the mean finite
    difference slope (this minimizes the rms slope of the detrended line
    scan); a0 makes the mean of the detrended line scan zero.
    """
    sl = self.parent_topography.derivative(1, periodic=False).mean()
    n, = self.nb_grid_pts
    s, = self.physical_sizes
    grad = sl * s
    return [self.parent_topography.mean() - grad * (n - 1) / (2 * n), sl * s]


def _detrend_2d_slope_coeffs(self):
    """
    Coefficients [a0, a1x, a1y] of the plane whose slopes are the mean
    finite difference slopes in x and y (this minimizes the rms gradient of
    the detrended topography); a0 makes the mean of the result zero.
    """
    slx, sly = self.parent_topography.derivative(1, periodic=False)
    slx = slx.mean()
    sly = sly.mean()
    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes
    return [self.parent_topography.mean() - slx * sx * (nx - 1) / (2 * nx) - sly * sy * (ny - 1) / (2 * ny),
            slx * sx, sly * sy]


class DetrendedUniformTopography(DecoratedUniformTopography):
    """
    Remove a polynomial trend from a uniform line scan or topography. The
    trend is a polynomial in the normalized coordinates :math:`x/L_x` (and
    :math:`y/L_y`) whose coefficients are determined according to
    `detrend_mode`; the result is the parent heights minus this polynomial.

    Note on periodicity: Detrended topographies with mode other than
    'center' have their `is_periodic` property set to False.
    """

    _detrend_functions = {
        'mean': lambda self: [self.parent_topography.mean()],
        # alias of 'mean'
        'center': lambda self: [self.parent_topography.mean()],
        'median': lambda self: [self.parent_topography.median()],
        'rms-tilt': lambda self: self.parent_topography.polyfit(1),
        # alias of 'rms-tilt'
        'height': lambda self: self.parent_topography.polyfit(1),
        'mad-tilt': lambda self: self.parent_topography.mad_polyfit(1),
        'slope': lambda self: _detrend_1d_slope_coeffs(self) if self.dim == 1 else _detrend_2d_slope_coeffs(self),
        'rms-curvature': lambda self: self.parent_topography.polyfit(2),
        # alias of 'rms-curvature'
        'curvature': lambda self: self.parent_topography.polyfit(2),
        'mad-curvature': lambda self: self.parent_topography.mad_polyfit(2),
    }

    def __init__(self, topography, detrend_mode='height', coeffs=None, info={}):
        """
        Note on periodicity: Detrended topographies with mode other than
        'center' have their `is_periodic` property set to False.

        Parameters
        ----------
        topography : :obj:`UniformLineScan` or :obj:`Topography`
            Topography to be detrended.
        detrend_mode : str, optional
            'mean' (alias 'center'): subtract the mean height, no tilt
            correction.
            'median': subtract the median height, no tilt correction.
            'rms-tilt' (alias 'height'): subtract the plane that minimizes the
            rms height.
            'mad-tilt': subtract the plane that minimizes the median absolute
            deviation of the heights.
            'slope': subtract the plane that minimizes the rms slope.
            'rms-curvature' (alias 'curvature'): subtract the second-order
            polynomial that minimizes the rms height.
            'mad-curvature': subtract the second-order polynomial that
            minimizes the median absolute deviation of the heights.
            (Default: 'height')
        coeffs : array_like, optional
            Coefficients of the detrending polynomial in normalized
            coordinates, in the order returned by `polyfit`. If given,
            `detrend_mode` is not used to compute them. (Default: None)
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, info=info)
        self._detrend_mode = detrend_mode
        self._coeffs = coeffs
        if self._coeffs is None:
            self._detrend()

    def _detrend(self):
        try:
            self._coeffs = self._detrend_functions[self._detrend_mode](self)
        except KeyError:
            raise ValueError(f"Unsupported detrend mode '{self._detrend_mode}' for uniform line scans or topographies.")

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._detrend_mode, self._coeffs
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._detrend_mode, self._coeffs = state
        super().__setstate__(superstate)

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def detrend_mode(self):
        return self._detrend_mode

    @detrend_mode.setter
    def detrend_mode(self, detrend_mode):
        self._detrend_mode = detrend_mode
        self._detrend()

    @property
    def is_periodic(self):
        """
        SurfaceTopography stays periodic only after detrend mode "center".
        Otherwise the detrended SurfaceTopography is non-periodic.
        """
        if self.detrend_mode == "center":
            return self.parent_topography.is_periodic
        else:
            return False

    def heights(self):
        """ Computes the combined profile.
        """
        if len(self._coeffs) == 1:
            a0, = self._coeffs
            return self.parent_topography.heights() - a0
        elif self.dim == 1:
            nx, = self.nb_grid_pts
            x = np.arange(self.nb_grid_pts[0]) / nx
            if len(self._coeffs) == 2:
                a0, a1 = self._coeffs
                return self.parent_topography.heights() - a0 - a1 * x
            elif len(self._coeffs) == 3:
                a0, a1, a2 = self._coeffs
                return self.parent_topography.heights() - a0 - a1 * x - a2 * x * x
            else:
                raise RuntimeError('Unknown size of coefficients tuple for line scans.')
        else:  # self.dim == 2
            x, y = np.meshgrid(*(np.arange(n) / n for n in self.nb_grid_pts),
                               indexing='ij')
            if len(self._coeffs) == 3:
                a0, a1x, a1y = self._coeffs
                return self.parent_topography.heights() - a0 - a1x * x - a1y * y
            elif len(self._coeffs) == 6:
                h0, m, n, mm, nn, mn = self._coeffs
                xx = x * x
                yy = y * y
                xy = x * y
                return self.parent_topography.heights() - h0 - m * x - n * y - mm * xx - nn * yy - mn * xy
            else:
                raise RuntimeError('Unknown size of coefficients tuple for 2D topographies.')

    def stringify_plane(self, fmt=lambda x: str(x)):
        """
        Return a string giving the expression for the detrending polynomial.
        `x` and `y` in this expression are the normalized coordinates
        :math:`x/L_x` and :math:`y/L_y`.

        Parameters
        ----------
        fmt : callable, optional
            Function that converts a coefficient to a string.
            (Default: str)
        """

        str_coeffs = [fmt(x) for x in self._coeffs]
        if self.dim == 1:
            if len(self._coeffs) == 1:
                h0, = str_coeffs
                return h0
            elif len(self._coeffs) == 2:
                return '{0} + {1} x'.format(*str_coeffs)
            elif len(self._coeffs) == 3:
                return '{0} + {1} x + {2} x^2'.format(*str_coeffs)
            else:
                raise RuntimeError(
                    'Unknown physical_sizes of coefficients tuple.')
        else:
            if len(self._coeffs) == 1:
                h0, = str_coeffs
                return h0
            elif len(self._coeffs) == 3:
                return '{0} + {1} x + {2} y'.format(*str_coeffs)
            elif len(self._coeffs) == 6:
                return '{0} + {1} x + {2} y + {3} x^2 + {4} y^2 + {5} xy' \
                    .format(*str_coeffs)
            else:
                raise RuntimeError(
                    'Unknown physical_sizes of coefficients tuple.')

    @property
    def curvatures(self):
        r"""
        Curvature(s) of the detrending polynomial in physical units (inverse
        length).

        Returns
        -------
        curvatures : tuple
            :math:`(\rho,)` with :math:`\rho = 1 / R = \partial^2 p/\partial x^2` for
            line scans, :math:`(\rho_{xx}, \rho_{yy}, \rho_{xy})` for
            topographies. Zero for modes without second-order terms.
        """

        if self.dim == 1:
            if len(self._coeffs) == 3:
                sx, = self.physical_sizes
                return 2 * self._coeffs[2] / sx ** 2,
            elif len(self._coeffs) in {1, 2}:
                return 0,
            else:
                raise RuntimeError('Unknown size of coefficients tuple.')
        else:
            if len(self._coeffs) == 6:
                sx, sy = self.physical_sizes
                return 2 * self._coeffs[3] / sx ** 2, 2 * self._coeffs[4] / sy ** 2, 2 * self._coeffs[5] / (sx * sy)
            elif len(self._coeffs) in {1, 3}:
                return 0, 0, 0
            else:
                raise RuntimeError('Unknown size of coefficients tuple.')


UniformTopographyInterface.register_function('polyfit', polyfit)
UniformTopographyInterface.register_function('detrend', DetrendedUniformTopography)
