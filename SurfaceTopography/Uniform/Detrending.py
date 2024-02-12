#
# Copyright 2016, 2018-2020 Lars Pastewka
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

"""
Helper functions to compute trends of surfaces
"""

import numpy as np

from SurfaceTopography.UniformLineScanAndTopography import (
    DecoratedUniformTopography, UniformLineScan, UniformTopographyInterface)


def polyfit(self, deg):
    """
    Compute the detrending plane that, if subtracted, minimizes the rms height.

    Parameters
    ----------
    self : :obj:`UniformLineScan` or :obj:`Topography`
        Topography or line scan container object.
    """
    coeffs = np.polyfit(*self.positions_and_heights(), deg)
    return np.array(coeffs)[::-1]


def tilt_from_height(topography, full_output=False):
    """
    Compute the tilt plane that if subtracted minimizes the rms height of the
    surface. The tilt plane is parameterized as:

    .. math::

        p(x, y) = h_0 + m x + n y

    The values of :math:`m`, :math:`n` and :math:`h0` are return by this
    function.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()

    Parameters
    ----------
    arr : UniformTopography
        Height information.

    Returns
    -------
    m : float
        Slope in x-direction.
    n : float
        Slope in y-direction.
    h0 : float
        Mean value.
    """
    arr = topography.heights()
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) / arr.shape[i] for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def tilt_and_curvature(arr, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()

    Returns:
    ---------
    coeffs [, location_matrix (if full_output)]

    coeffs ordered as follows

    {5} + {0} x + {1} y + {2} x^2 + {3} y^2 + {4} xy

    """
    arr = arr[...]
    nb_dim = len(arr.shape)
    assert nb_dim == 2
    x_grids = (np.arange(arr.shape[i]) / arr.shape[i] for i in range(nb_dim))
    # Linear terms
    x_grids = np.meshgrid(*x_grids, indexing='ij')
    # Quadratic terms
    x, y = x_grids
    x_grids += [x * x, y * y, x * y]
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    # res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def shift_and_tilt(topography):
    """
    returns an array of same shape and physical_sizes as arr, but shifted and
    tilted so that mean(arr) = 0 and mean(arr**2) is minimized
    """
    arr = topography.heights()
    coeffs, location_matrix = tilt_from_height(topography, full_output=True)
    coeffs = np.array(coeffs)
    offsets = arr.reshape((-1,))

    return (offsets - location_matrix @ coeffs).reshape(arr.shape)


def _detrend_1d_slope_coeffs(self):
    sl = self.parent_topography.derivative(1, periodic=False).mean()
    n, = self.nb_grid_pts
    s, = self.physical_sizes
    grad = sl * s
    return [self.parent_topography.mean() - grad * (n - 1) / (2 * n), sl]


def _detrend_2d_slope_coeffs(self):
    slx, sly = self.parent_topography.derivative(1, periodic=False)
    slx = slx.mean()
    sly = sly.mean()
    nx, ny = self.nb_grid_pts
    sx, sy = self.physical_sizes
    return [
        slx * sx, sly * sy,
        self.parent_topography.mean() - slx * sx * (nx - 1) / (2 * nx) - sly * sy * (ny - 1) / (2 * ny)
    ]


class DetrendedUniformTopography(DecoratedUniformTopography):
    """
    Remove trends from a topography. This is achieved by fitting polynomials
    to the topography data to extract trend lines. The resulting topography
    is then detrended by substracting these trend lines.

    Note on periodicity: Detrended Topographies with mode other than 'center'
    will have `is_periodic` property set to False.
    """

    _detrend_1d_functions = {
        'mean': lambda self: [self.parent_topography.mean()],
        # same as 'mean', deprecate 'center' in the future
        'center': lambda self: [self.parent_topography.mean()],
        'median': lambda self: [self.parent_topography.median()],
        'rms-tilt': lambda self: self.parent_topography.polyfit(1),
        # same as 'rms-tilt', deprecate 'height' in the future
        'height': lambda self: self.parent_topography.polyfit(1),
        'mad-tilt': lambda self: self.parent_topography.mad_polyfit(1),
        'slope': _detrend_1d_slope_coeffs,
        'rms-curvature': lambda self: self.parent_topography.polyfit(2),
        # same as 'rms-curvature', deprecate 'curvature' in the future
        'curvature': lambda self: self.parent_topography.polyfit(2),
        'mad-curvature': lambda self: self.parent_topography.mad_polyfit(2),
    }

    def __init__(self, topography, detrend_mode='height', coeffs=None, info={}):
        """
        Note on periodicity: Detrended Topographies with mode other than
        "center" will have `is_periodic` property set to False.

        Parameters
        ----------
        topography : Topography
            SurfaceTopography to be detrended.
        detrend_mode : str
            'center': center the topography, no trend correction.
            'height': adjust slope such that rms height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'curvature': adjust slope and curvature such that rms height is
            minimized.
            (Default: 'height')
        coeffs : array_like, optional
            Coefficients of the detrending plane. If not given, they are
            computed from the topography. (Default: None)
        """
        super().__init__(topography, info=info)
        self._detrend_mode = detrend_mode
        self._coeffs = coeffs
        if self._coeffs is None:
            self._detrend()

    def _detrend(self):
        if self.dim == 1:
            try:
                self._coeffs = self._detrend_1d_functions[self._detrend_mode](self)
            except KeyError:
                raise ValueError("Unsupported detrend mode '{}' for line scans.".format(self._detrend_mode))
        else:  # self.dim == 2
            if self._detrend_mode is None or self._detrend_mode == 'center':
                self._coeffs = [self.parent_topography.mean()]
            elif self._detrend_mode == 'height':
                self._coeffs = [s for s in tilt_from_height(self.parent_topography)]
            elif self._detrend_mode == 'slope':
                slx, sly = self.parent_topography.derivative(1, periodic=False)
                slx = slx.mean()
                sly = sly.mean()
                nx, ny = self.nb_grid_pts
                sx, sy = self.physical_sizes
                self._coeffs = [
                    slx * sx, sly * sy,
                    self.parent_topography.mean() - slx * sx * (nx - 1) / (2 * nx) - sly * sy * (ny - 1) / (2 * ny)
                ]
            elif self._detrend_mode == 'curvature':
                self._coeffs = [s for s in tilt_and_curvature(self.parent_topography)]
            else:
                raise ValueError("Unsupported detrend mode '{}' for 2D topographies.".format(self._detrend_mode))

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
            dx, = self.pixel_size
            x = np.arange(self.nb_grid_pts[0]) * dx
            if len(self._coeffs) == 2:
                a0, a1 = self._coeffs
                return self.parent_topography.heights() - a0 - a1 * x
            elif len(self._coeffs) == 3:
                a0, a1, a2 = self._coeffs
                return self.parent_topography.heights() - a0 - a1 * x - a2 * x * x
            else:
                raise RuntimeError('Unknown size of coefficients tuple for line scans.')
        else:  # self.dim == 2
            x, y = np.meshgrid(*(np.arange(n) / n for n in self.nb_grid_pts), indexing='ij')
            if len(self._coeffs) == 3:
                a1x, a1y, a0 = self._coeffs
                return self.parent_topography.heights() - a0 - a1x * x - a1y * y
            elif len(self._coeffs) == 6:
                m, n, mm, nn, mn, h0 = self._coeffs
                xx = x * x
                yy = y * y
                xy = x * y
                return self.parent_topography.heights() - h0 - m * x - n * y - mm * xx - nn * yy - mn * xy
            else:
                raise RuntimeError('Unknown size of coefficients tuple for 2D topographies.')

    def stringify_plane(self, fmt=lambda x: str(x)):
        """
        Return a string giving the expression for the detrending plane.
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
                return '{2} + {0} x + {1} y'.format(*str_coeffs)
            elif len(self._coeffs) == 6:
                return '{5} + {0} x + {1} y + {2} x^2 + {3} y^2 + {4} xy' \
                    .format(*str_coeffs)
            else:
                raise RuntimeError(
                    'Unknown physical_sizes of coefficients tuple.')

    @property
    def curvatures(self):
        r"""
        Curvature(s) of the fitted plane.

        Returns
        -------
        :math:`\rho = 1 / R` for line scans or tuple :math:`\rho_{xx}, \rho_{yy}, \rho_{xy}` for topographies
        """

        if self.dim == 1:
            if len(self._coeffs) == 3:
                return 2 * self._coeffs[2],
            elif len(self._coeffs) in {1, 2}:
                return 0,
            else:
                raise RuntimeError(
                    'Unknown physical_sizes of coefficients tuple.')
        else:
            if len(self._coeffs) == 6:
                sx, sy = self.physical_sizes
                return (2 * self._coeffs[2]) / sx ** 2, (
                        2 * self._coeffs[3]) / sy ** 2, (
                               2 * self._coeffs[4]) / (sx * sy)
            elif len(self._coeffs) in {1, 3}:
                return 0, 0, 0
            else:
                raise RuntimeError(
                    'Unknown physical_sizes of coefficients tuple.')


UniformLineScan.register_function('polyfit', polyfit)
UniformTopographyInterface.register_function('detrend', DetrendedUniformTopography)
