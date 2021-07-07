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
from ..Interpolation import Bicubic
from ..UniformLineScanAndTopography import Topography
from ..UniformLineScanAndTopography import DecoratedUniformTopography


def bicubic_interpolator(topography):
    r"""
    Returns a bicubic interpolation function based on the topography's heights
    and slopes (fourier_derivative)
    """
    if not topography.is_periodic:
        raise ValueError(
            "Bicubic interpolation only implemented for periodic surfaces")

    dx, dy = topography.pixel_size
    derx, dery = topography.fourier_derivative()
    interp = Bicubic(topography.heights(),
                     derx * dx,  # Bicubic assumes a grid spacing of 1
                     dery * dy
                     )

    def wrapped_bicubic(x, y, derivative=0):
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
            interp_field, interp_derx, interp_dery, \
                interp_derxx, interp_deryy, interp_derxy = \
                interp(x / dx, y / dy, derivative=derivative)

            interp_derx = interp_derx / dx
            interp_dery = interp_dery / dy

            interp_derxx = interp_derxx / dx ** 2
            interp_deryy = interp_deryy / dy ** 2
            interp_derxy = interp_derxx / dx / dy

            return interp_field, interp_derx, interp_dery, \
                interp_derxx, interp_deryy, interp_derxy

    return wrapped_bicubic


def interpolate_fourier(topography, nb_grid_pts):
    r"""
    Interpolates the heights of `topography` on a finer grid given by
    `nb_grid_pts` by padding the fourier spectrum with zeros.

    Only 2D implemented.

    Note: at the nyquist frequenca we use the "minimal-oscillation"
    interpolation, i.e. we assume a cosinus wave
    (https://math.mit.edu/~stevenj/fft-deriv.pdf).

    Parameters
    ----------
    topography: Topography
    nb_grid_pts: tuple

    Returns
    -------
    SurfaceTopography with interolated values

    """
    bigspectrum = np.zeros((nb_grid_pts[0], nb_grid_pts[1] // 2 + 1),
                           dtype=complex)
    smallspectrum = np.fft.rfft2(topography.heights())
    snx, sny = smallspectrum.shape
    nx, ny = topography.nb_grid_pts

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
                      * np.prod(nb_grid_pts) / np.prod(topography.nb_grid_pts),
                      # normalization
                      physical_sizes=topography.physical_sizes)


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
Topography.register_function("interpolate_bicubic", bicubic_interpolator)
UniformTopographyInterface.register_function('interpolate_fourier', interpolate_fourier)
