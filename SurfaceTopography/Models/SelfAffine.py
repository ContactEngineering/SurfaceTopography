import numpy as np
from SurfaceTopography.Generation import fourier_synthesis


# TODO abstract class for discrete and model PSDs
# TODO all wavelengths and prefactors as properties
#
# TODO: think about abstract Statistical roughness hierarchy

class AbstractStatisticalRoughness(object):
    def power_spectrum(seld):
        raise NotImplementedError

    def power_spectrum_profile(seld):
        raise NotImplementedError


class AbstractIsotropicRoughness(AbstractStatisticalRoughness):
    def power_spectrum_isotropic(self):
        raise NotImplementedError

    def variance_derivatice(self):
        raise NotImplementedError


class SelfAffine(AbstractIsotropicRoughness):
    def __init__(self,
                 cr,
                 rolloff_wavelength,
                 hurst_exponent,
                 longcut_wavelength=np.infty,
                 shortcut_wavelength=0, unit=None):
        self.cr = cr  # Ampliture of the PSD at and below the rolloff wavevector
        self.rolloff_wavelength = rolloff_wavelength
        self.longcut_wavelength = longcut_wavelength
        self.shortcut_wavelength = shortcut_wavelength
        self.hurst_exponent = hurst_exponent

        if self.shortcut_wavelength == 0:
            self.shortcut_wavevector = np.infty
        else:
            self.shortcut_wavevector = 2 * np.pi / self.shortcut_wavelength

        self.longcut_wavevector = 2 * np.pi / self.longcut_wavelength
        self.rolloff_wavevector = 2 * np.pi / self.rolloff_wavelength

        self.unit = unit

    def power_spectrum_isotropic(self, q):
        qs = self.shortcut_wavevector
        qL = self.longcut_wavevector
        qr = self.rolloff_wavevector

        return np.where(
            np.logical_and(q < qs, q > qL),
            np.where(q < qr,
                     self.cr,
                     self.cr * (q / qr) ** (-2 - 2 * self.hurst_exponent)),
            0)

    def power_spectrum_profile(self, q):
        # TODO:  I think this is only the simplified formula.
        # I think I could implement the Fresnel integral

        qs = self.shortcut_wavevector
        qL = self.longcut_wavevector
        qr = self.rolloff_wavevector

        return np.where(
            np.logical_and(q < qs, q > qL),
            np.where(q < qr,
                     self.cr * qr / np.pi,
                     self.cr * (q / qr) ** (-2 - 2 * self.hurst_exponent)
                     * q / np.pi * np.sqrt(1 - (q / qs) ** 2)
                     ),
            0)

    def rms_height(self, shortcut_wavelength=None, longcut_wavelength=None, ):
        r"""
        See 10.1088/2051-672X/aa51f8 equations 4, 10 and 11

        .. math ::

            h^2_\mathrm{rms} = \frac{C_r q_r^2}{4 \pi}
            \left[ 1 + \frac{1}{H} - \left( \frac{q_l}{q_r}\right)^2 - \left( \frac{q_s}{q_r}\right)^{-2H}\right]

        Note that :math:`C_r q_r^2` can be interchanged with :math:`C_0 q_r^{-2H}`

        Parameters
        ----------
        shortcut_wavelength: float or array, optional
            makes the rms-height scale dependent by overriding the shortcut
        longcut_wavelength
            makes the rms-height scale dependent by overriding the longcut

        Returns
        -------
            float or array

        """

        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength

        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength

        shortcut_wavevector = (2 * np.pi) / shortcut_wavelength if shortcut_wavelength > 0 else np.infty
        longcut_wavevector = (2 * np.pi) / longcut_wavelength

        # see labbook of 220304
        return np.sqrt(
            self.cr * self.rolloff_wavevector ** 2 / (4 * np.pi)
            * (1 + 1 / self.hurst_exponent
               - (longcut_wavevector / self.rolloff_wavevector) ** 2
               - (shortcut_wavevector / self.rolloff_wavevector) ** (- 2 * self.hurst_exponent)
               )
        )

    def variance_derivative(self, order, shortcut_wavelength=None, longcut_wavelength=None, ):
        r"""
         Variance of a derivative of arbitrary (fractional) order

        The contribution for an isotropic flat PSD with
        amplite :math:`C_r` between the wavevectors :math:`[q_L, q_s]` is

        .. math ::

            \left[h^{(\alpha)}_\mathrm{rms}\right]^2 = \frac{C_r}{2 \pi}
                \left[ q_s^{2 + 2\alpha } - q_L ^{2 + 2\alpha} \right] / (2 + 2 \alpha)

        The contribution for a power-law PSD between :math:`[q_L, q_s]`

        For  :math:`\alpha \neq H`

        .. math ::


                \left[h^{(\alpha)}_\mathrm{rms}\right]^2 = \frac{C_0}{2 \pi}
                            \left[ q_s^{ 2\alpha - 2H} - q_L ^{2\alpha - 2H} \right] / (2\alpha - 2 H)

        and for  :math:`\alpha = H`

        .. math ::

            \left[h^{(\alpha)}_\mathrm{rms}\right]^2 = \frac{C_0}{2 \pi}
                \ln\left( q_s/ q_L \right)

        Note that :math:`C_r q_r^2` can be interchanged with :math:`C_0 q_r^{-2H}`


        Parameters
        ----------
        order: float
             order of the derivative
        shortcut_wavelength: float or array, optional
            makes the result scale dependent by overriding the shortcut
        longcut_wavelength
            makes the result scale dependent by overriding the longcut

        Returns
        -------
            float or array

        """
        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength
        else:
            shortcut_wavelength = max(shortcut_wavelength, self.shortcut_wavelength)
        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength
        else:
            longcut_wavelength = min(longcut_wavelength, self.longcut_wavelength)

        shortcut_wavevector = (2 * np.pi) / shortcut_wavelength if shortcut_wavelength > 0 else np.infty
        longcut_wavevector = (2 * np.pi) / longcut_wavelength
        rolloff_wavevector = self.rolloff_wavevector
        # prefactor of the self-afine region:
        c0 = self.cr * self.rolloff_wavevector ** (2 + 2 * self.hurst_exponent)

        if longcut_wavevector < min(rolloff_wavevector, shortcut_wavevector):
            # We split the contributions to the integral between the rolloff and the self-affine region.
            # rolloff region
            rolloff = self.cr / (2 * np.pi) * (
                    min(rolloff_wavevector, shortcut_wavevector) ** (2 + 2 * order)
                    - longcut_wavevector ** (2 + 2 * order)
            ) / (2 + 2 * order)
        else:
            rolloff = 0
        if shortcut_wavevector > rolloff_wavevector and longcut_wavevector < shortcut_wavevector:
            # self-affine region
            if self.hurst_exponent == order:
                self_affine = c0 / (2 * np.pi) * np.log(shortcut_wavevector / rolloff_wavevector)
            else:
                self_affine = (c0 / (2 * np.pi)
                               * (shortcut_wavevector ** (2 * (order - self.hurst_exponent))
                                  - max(rolloff_wavevector, longcut_wavevector) ** (2 * (order - self.hurst_exponent))
                                  ) /
                               (2 * (order - self.hurst_exponent))
                               )
        else:
            self_affine = 0
        return self_affine + rolloff

    # def rms_slope(self):
    # See 39fa2d84-e57d-4fc2-a0f1-31bba9233c46/Scale_dependent_slope/__init__.py
    #
    # def self_affine_scale_dependent_rms_slope(q_max, hurst=0.6,c0=1., q_rolloff = 1e-10):
    #
    #     return np.sqrt(np.where(xa
    #     q_max < q_rolloff,
    #     1 / (3 * np.pi) * c0 * q_rolloff ** (-1 - 2*hurst) * q_max **3,
    #     1 / (3 * np.pi) * c0 * q_rolloff ** (2 - 2*hurst)
    #     + (c0 / (2-2 * hurst) / np.pi) * (q_max**(2-2*hurst) - q_rolloff**(2-2*hurst))
    #     ))
    #

    def variance_half_derivative(self, shortcut_wavelength=None,
                                 longcut_wavelength=None):
        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength

        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength

        return _variance_half_derivative(
            self.cr,
            self.rolloff_wavelength,
            self.hurst_exponent,
            longcut_wavelength=longcut_wavelength,
            shortcut_wavelength=shortcut_wavelength,
        )

    def generate_roughness(self,
                           pixel_size,
                           n_pixels,
                           seed,
                           shortcut_wavelength=None,
                           longcut_wavelength=None,
                           **kwargs):
        r"""
        Generates a discrete random realisation of Gaussian noise with variances of the waves given by the PSD

        The `shortcut_wavelength` and `longcut_wavelength` parameters allow to overwrite the
        values paramtetrized in the model, since a discrete realization is often shorter.

        See fourier synthesis for more arguments
        """

        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength

        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength

        nx = n_pixels
        dx = pixel_size
        sx = nx * dx

        c0 = self.cr * (2 * np.pi / self.rolloff_wavelength) ** (2 + 2 * self.hurst_exponent)

        np.random.seed(seed)
        roughness = fourier_synthesis((n_pixels, n_pixels), (sx, sx),
                                      hurst=self.hurst_exponent,
                                      c0=c0,
                                      short_cutoff=self.shortcut_wavelength,
                                      long_cutoff=self.rolloff_wavelength,
                                      unit=self.unit,
                                      **kwargs
                                      )
        return roughness


def _variance_half_derivative(
        cr,
        rolloff_wavelength,
        hurst_exponent,
        longcut_wavelength=np.infty,
        shortcut_wavelength=0,
        **kwargs):
    r"""
    computes the variance of the half-derivative for the idealized isotropic self-affine PSD with flat rolloff region

    :: math ..

        \left(h_\mathrm{rms}^{(1/2)}\right)^{2}
        &= \frac{1}{4\pi^{2}}\iint \dif q_{x}\dif q_{y} C^{\mathrm{2D}}(q_{x}, q_{y}) |q|


    Parameters:
    -----------


    """
    if longcut_wavelength < rolloff_wavelength:
        raise ValueError

    fac = cr / (2 * np.pi)

    H = hurst_exponent

    q_r = 2 * np.pi / rolloff_wavelength
    q_L = 2 * np.pi / longcut_wavelength
    rolloff_contrib = fac / 3 * (q_r ** 3 - q_L ** 3)

    if hurst_exponent <= 0.5:
        if shortcut_wavelength == 0:
            return np.infty
        q_s = 2 * np.pi / shortcut_wavelength
        if hurst_exponent == 0.5:
            return rolloff_contrib + fac * np.log(q_s / q_r) * q_r ** 3
        else:
            return rolloff_contrib + fac * q_r ** 3 / (1 - 2 * H) * ((q_s / q_r) ** (1 - 2 * H) - 1)
    if shortcut_wavelength == 0:
        return rolloff_contrib - fac * q_r ** 3 / (1 - 2 * H)
    else:
        q_s = 2 * np.pi / shortcut_wavelength
        return rolloff_contrib + fac * q_r ** 3 / (1 - 2 * H) * ((q_s / q_r) ** (1 - 2 * H) - 1)
