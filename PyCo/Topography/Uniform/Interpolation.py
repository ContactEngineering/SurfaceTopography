
import numpy as np

from ..UniformLineScanAndTopography import Topography
from ..HeightContainer import UniformTopographyInterface
from PyCo.Tools.Interpolation import Bicubic

def bicubic_interpolator(topography):
    r"""
    Returns a bicubic interpolation function based on the topography's heights
    and slopes (fourier_derivative)
    """
    if not topography.is_periodic:
        raise ValueError("Bicubic interpolation only implemented for periodic surfaces")

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
            if > 0 returns also derivatives

        Returns
        -------
        array of interpolated values or tuple with arrays of values and derivatives

        """
        if derivative == 0:
            interp_field, = interp(x / dx, y / dy, derivative=derivative)
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
    Interpolates the heights of `topography` on a finer grid given by `nb_grid_pts`
    by padding the fourier spectrum with zeros.

    Only 2D implemented.

    Note: at the nyquist frequenca we use the "minimal-oscillation" interpolation,
    i.e. we assume a cosinus wave (https://math.mit.edu/~stevenj/fft-deriv.pdf).

    Parameters
    ----------
    topography: Topography
    nb_grid_pts: tuple

    Returns
    -------
    Topography with interolated values

    """
    bigspectrum = np.zeros((nb_grid_pts[0], nb_grid_pts[1] // 2 + 1),
                           dtype=complex)
    smallspectrum = np.fft.rfft2(topography.heights())
    snx, sny = smallspectrum.shape

    i = snx // 2 if snx % 2 == 0 else (snx - 1) // 2 + 1
    bigspectrum[:i, :sny] = smallspectrum[:i, :sny]
    bigspectrum[-(snx - i):, :sny] = smallspectrum[-(snx - i):, :sny]

    # in the y direction, the entries at the nyquist frequency are automatically
    # symmetrised by the rfft, corresponding to a cosinusoidal wave, i.e. the
    # "minimal-oscillation" trigonometric interpolation
    # (https://math.mit.edu/~stevenj/fft-deriv.pdf).
    # To ensure the same behaviour in the x direction
    # (see test_fourier_interpolate_transpose_symmetry`)`
    # we have to mirror the entries at the niquist frequency by hand
    if snx%2 == 0:
        bigspectrum[i, :sny] = smallspectrum[i, :sny]

    return Topography(np.fft.irfft2(bigspectrum, s=nb_grid_pts)
                      * np.prod(nb_grid_pts) / np.prod(topography.nb_grid_pts),  # normalization
                      physical_sizes=topography.physical_sizes)

Topography.register_function("interpolate_bicubic", bicubic_interpolator)
UniformTopographyInterface.register_function('interpolate_fourier', interpolate_fourier)
