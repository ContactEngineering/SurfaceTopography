#
# Copyright 2020-2021 Lars Pastewka
#           2020 Antoine Sanner
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

from ..HeightContainer import UniformTopographyInterface
from SurfaceTopography.Support.Interpolation import Bicubic
from ..UniformLineScanAndTopography import Topography
from ..UniformLineScanAndTopography import DecoratedUniformTopography


def interpolate_linear(self):
    r"""
    Returns a linear interpolation function based on the topography's heights.
    For (2D) topographies, the interpolation is not unique. Each rectangle is
    split into two triangles that side in the lower left and upper right part
    of the rectangle.

    Note that for nonperiodic line scans/topographies, interpolation only
    works for positions between 0 and sx - sx/nx - i.e. the interpolated
    physical domain of the topography is smaller than the nominal domain.
    """

    def linear_interpolator_line_scan(x, periodic=None):
        """Override periodicity of underlying topography with the `periodic` argument"""
        scaled_x = x / px
        periodic = is_periodic if periodic is None else periodic
        if not periodic:
            if np.any(scaled_x < 0) or np.any(scaled_x > nx-1):
                raise ValueError('Cannot interpolate outside of physical domain for nonperiodic line scans.')
        int_x = np.array(scaled_x, dtype=int)
        frac_x = scaled_x - int_x
        return (1 - frac_x) * heights[int_x % nx] + frac_x * heights[(int_x + 1) % nx]

    def linear_interpolator_topography(x, y, periodic=None):
        """Override periodicity of underlying topography with the `periodic` argument"""
        scaled_x = x / px
        scaled_y = y / py
        periodic = is_periodic if periodic is None else periodic
        if not periodic:
            if np.any(scaled_x < 0) or np.any(scaled_x > nx-1) or np.any(scaled_y < 0) or np.any(scaled_y > ny-1):
                raise ValueError('Cannot interpolate outside of physical domain for nonperiodic topographies.')
        int_x = np.array(scaled_x, dtype=int)
        int_y = np.array(scaled_y, dtype=int)
        frac_x = scaled_x - int_x
        frac_y = scaled_y - int_y

        int_x %= nx
        int_y %= ny
        int1_x = (int_x + 1) % nx
        int1_y = (int_y + 1) % ny

        lower_triangle = \
            (1 - frac_x - frac_y) * heights[int_x, int_y] + \
            frac_x * heights[int1_x, int_y] + \
            frac_y * heights[int_x, int1_y]
        upper_triangle = \
            (frac_x + frac_y - 1) * heights[int1_x, int1_y] + \
            (1 - frac_x) * heights[int_x, int1_y] + \
            (1 - frac_y) * heights[int1_x, int_y]

        return np.where(frac_x + frac_y > 1, upper_triangle, lower_triangle)

    heights = self.heights()
    is_periodic = self.is_periodic
    if self.dim == 1:
        nx, = self.nb_grid_pts
        px, = self.pixel_size
        return linear_interpolator_line_scan
    elif self.dim == 2:
        nx, ny = self.nb_grid_pts
        px, py = self.pixel_size
        return linear_interpolator_topography
    else:
        raise ValueError(f'Cannot interpolate {self.dim}-dimensional height containers.')


def interpolate_bicubic(self, derivative='fourier'):
    r"""
    Returns a bicubic interpolation function based on the topography's heights
    and slopes as obtained using finite-differences (derivative='fd') or Fourier
    derivative (derivative='fourier').

    Note that for nonperiodic line scans/topographies, interpolation only
    works for positions between 0 and sx - sx/nx - i.e. the interpolated
    physical domain of the topography is smaller than the nominal domain.
    """
    if not self.is_periodic:
        raise ValueError('Bicubic interpolation is only implemented for periodic surfaces.')

    if self.dim != 2:
        raise ValueError('Bicubic interpolation is only implemented for topographies (not line scans).')

    dx, dy = self.pixel_size
    if derivative == 'fourier':
        derx, dery = self.fourier_derivative()
        interp = Bicubic(self.heights(),
                         derx * dx,  # Bicubic assumes a grid spacing of 1
                         dery * dy
                         )
    elif derivative == 'fd':
        # The Bicubic class automatically determines derivatives from fd if left unspecified
        interp = Bicubic(self.heights())
    else:
        raise ValueError(f"Unknown `derivative` type '{derivative}'.")

    def bicubic_interpolator_topography(x, y, derivative=0):
        r"""

        Parameters
        ----------
        x:  array
            coordinate where to interpolate the topography
        y:  array
            coordinate where to interpolate the topography
        derivative: int 0, 1 or 2, optional
            if > 0 returns also derivatives. Note that the 2nd order
            derivatives are not continuous and hence not really usefull

        Returns
        -------
        array of interpolated values or tuple with arrays of values and
        derivatives

        """
        if derivative == 0:
            interp_field = interp(x / dx, y / dy, derivative=derivative)
            return interp_field
        elif derivative == 1:
            interp_field, interp_derx, interp_dery = \
                interp(x / dx, y / dy, derivative=derivative)
            # back to our original grid spacing
            interp_derx = interp_derx / dx
            interp_dery = interp_dery / dy
            return interp_field, interp_derx, interp_dery
        elif derivative == 2:
            interp_field, interp_derx, interp_dery, interp_derxx, interp_deryy, interp_derxy = \
                interp(x / dx, y / dy, derivative=derivative)

            interp_derx = interp_derx / dx
            interp_dery = interp_dery / dy

            interp_derxx = interp_derxx / dx ** 2
            interp_deryy = interp_deryy / dy ** 2
            interp_derxy = interp_derxx / dx / dy

            return interp_field, interp_derx, interp_dery, interp_derxx, interp_deryy, interp_derxy

    return bicubic_interpolator_topography


def interpolate_fourier(self, nb_grid_pts):
    r"""
    Interpolates the heights of `topography` on a finer grid given by
    `nb_grid_pts` by padding the fourier spectrum with zeros.

    Only 2D implemented.

    Note: at the nyquist frequency we use the "minimal-oscillation"
    interpolation, i.e. we assume a cosine wave
    (https://math.mit.edu/~stevenj/fft-deriv.pdf).

    Parameters
    ----------
    self: Topography
    nb_grid_pts: tuple

    Returns
    -------
    SurfaceTopography with interolated values

    """
    bigspectrum = np.zeros((nb_grid_pts[0], nb_grid_pts[1] // 2 + 1),
                           dtype=complex)
    smallspectrum = np.fft.rfft2(self.heights())
    snx, sny = smallspectrum.shape
    nx, ny = self.nb_grid_pts

    # the entries at the nyquist frequency are the superposition of the
    # positive and negative frequency. When we increase the fourier domain, we
    # will split that again between positive and negative, therefore we divide
    # the value by two and copy it to the corresponding negative vector
    if ny % 2 == 0:
        # the nyquist frequency also contains the symmetric and will be
        # twice too big when the symmetric will be included
        smallspectrum[:, -1] = smallspectrum[:, -1] / 2

    if nx % 2 == 0:
        # the nyquist frequency also contains the symmetric and will be
        # twice too big when the symmetric will be included
        smallspectrum[int(nx / 2), :] = smallspectrum[int(nx / 2), :] / 2

    i = snx // 2 if snx % 2 == 0 else (snx - 1) // 2 + 1
    bigspectrum[:i, :sny] = smallspectrum[:i, :sny]
    bigspectrum[-(snx - i):, :sny] = smallspectrum[-(snx - i):, :sny]

    # in the y direction, the entries at the nyquist frequency are
    # automatically symmetrised by the rfft, corresponding to a cosinusoidal
    # wave, i.e. the "minimal-oscillation" trigonometric interpolation
    # (https://math.mit.edu/~stevenj/fft-deriv.pdf).
    # To ensure the same behaviour in the x direction
    # (see test_fourier_interpolate_transpose_symmetry`)`
    # we have to mirror the entries at the niquist frequency by hand
    if snx % 2 == 0:
        bigspectrum[i, :sny] = smallspectrum[i, :sny]

    return Topography(np.fft.irfft2(bigspectrum, s=nb_grid_pts)
                      * np.prod(nb_grid_pts) / np.prod(self.nb_grid_pts),
                      # normalization
                      physical_sizes=self.physical_sizes)


class MirrorStichedTopography(DecoratedUniformTopography):
    """
    adds mirror images at the top (big y) and at the right (big x) of the
    topography, making it kind of periodic

    """

    def __init__(self, parent_topography, info={}):
        if parent_topography.communicator.Get_size() > 1:
            raise (NotImplementedError("MirrorStichedTopography "
                                       "not domain decomposable"))
        super().__init__(parent_topography, info)

    @property
    def is_periodic(self):
        return True

    @property
    def physical_sizes(self):
        return [2 * s for s in self.parent_topography.physical_sizes]

    @property
    def nb_grid_pts(self):
        return [2 * s for s in self.parent_topography.nb_grid_pts]

    def heights(self):
        h = self.parent_topography.heights()
        return np.block([[h[:, :], h[:, ::-1]],
                         [h[:: -1, :], h[::-1, :: -1]]])

    def positions(self):
        nx, ny = self.nb_grid_pts
        lnx, lny = self.nb_subdomain_grid_pts
        sx, sy = self.physical_sizes
        return np.meshgrid(
            (self.subdomain_locations[0] + np.arange(lnx)) * sx / nx,
            (self.subdomain_locations[1] + np.arange(lny)) * sy / ny,
            indexing='ij')


Topography.register_function("mirror_stitch", MirrorStichedTopography)
Topography.register_function("interpolate_linear", interpolate_linear)
Topography.register_function("interpolate_bicubic", interpolate_bicubic)
UniformTopographyInterface.register_function('interpolate_fourier', interpolate_fourier)
