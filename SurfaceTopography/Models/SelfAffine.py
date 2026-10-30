#
# Copyright 2023 antoine.sanner@ifb.baug.ethz.ch
#           2022-2023 Antoine Sanner
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
Analytic model of the power-spectral density of a self-affine topography
with roll-off.
"""

import numpy as np

from SurfaceTopography.Generation import fourier_synthesis

# TODO abstract class for discrete and model PSDs
# TODO all wavelengths and prefactors as properties
#
# TODO: think about abstract Statistical roughness hierarchy


class AbstractStatisticalRoughness(object):
    """Interface of statistical roughness models."""

    def power_spectrum(seld):
        raise NotImplementedError

    def power_spectrum_profile(seld):
        raise NotImplementedError


class AbstractIsotropicRoughness(AbstractStatisticalRoughness):
    """Interface of statistical models of isotropic roughness."""

    def power_spectrum_isotropic(self):
        raise NotImplementedError

    def variance_derivatice(self):
        raise NotImplementedError


class SelfAffine(AbstractIsotropicRoughness):
    r"""
    Isotropic self-affine roughness with roll-off. The isotropic PSD is

    .. math ::

        C^\text{iso}(q) = \left\{ \begin{array}{ll}
            C_r & \text{for } q_L < q < q_r \\
            C_r (q/q_r)^{-2-2H} & \text{for } q_r \le q < q_s \\
            0 & \text{else}
        \end{array} \right.

    with :math:`q_r = 2\pi/\lambda_r`, :math:`q_L = 2\pi/\lambda_L` and
    :math:`q_s = 2\pi/\lambda_s`. In the self-affine region,
    :math:`C^\text{iso}(q) = C_0 q^{-2-2H}` with
    :math:`C_0 = C_r q_r^{2+2H}`.

    Parameters
    ----------
    cr : float
        Value :math:`C_r` of the PSD at and below the roll-off wavevector.
    rolloff_wavelength : float
        Roll-off wavelength :math:`\lambda_r`.
    hurst_exponent : float
        Hurst exponent :math:`H`.
    longcut_wavelength : float, optional
        Long-wavelength cutoff :math:`\lambda_L`. (Default: np.inf)
    shortcut_wavelength : float, optional
        Short-wavelength cutoff :math:`\lambda_s`; zero means no cutoff.
        (Default: 0)
    unit : str, optional
        Length unit. (Default: None)
    """

    def __init__(self,
                 cr,
                 rolloff_wavelength,
                 hurst_exponent,
                 longcut_wavelength=np.inf,
                 shortcut_wavelength=0, unit=None):
        self.cr = cr  # Ampliture of the PSD at and below the rolloff wavevector
        self.rolloff_wavelength = rolloff_wavelength
        self.longcut_wavelength = longcut_wavelength
        self.shortcut_wavelength = shortcut_wavelength
        self.hurst_exponent = hurst_exponent

        if self.shortcut_wavelength == 0:
            self.shortcut_wavevector = np.inf
        else:
            self.shortcut_wavevector = 2 * np.pi / self.shortcut_wavelength

        self.longcut_wavevector = 2 * np.pi / self.longcut_wavelength
        self.rolloff_wavevector = 2 * np.pi / self.rolloff_wavelength

        self.unit = unit

    def power_spectrum_isotropic(self, q):
        r"""
        Isotropic PSD :math:`C^\text{iso}(q)` (units length^4).

        Parameters
        ----------
        q : float or np.ndarray
            Magnitude of the wavevector.

        Returns
        -------
        psd : np.ndarray
            PSD at `q`.
        """
        qs = self.shortcut_wavevector
        qL = self.longcut_wavevector
        qr = self.rolloff_wavevector

        # Note: only evaluate the power law where it applies; evaluating it
        # at q = 0 (e.g. through np.where) would produce division warnings
        # and infinities
        q = np.asarray(q, dtype=float)
        C = np.zeros_like(q)
        inside = np.logical_and(q < qs, q > qL)
        plateau = np.logical_and(inside, q < qr)
        decay = np.logical_and(inside, q >= qr)
        C[plateau] = self.cr
        C[decay] = self.cr * (q[decay] / qr) ** (-2 - 2 * self.hurst_exponent)
        return C

    def power_spectrum_profile(self, q):
        r"""
        Approximate one-dimensional PSD :math:`C^\text{1D}(q)` (units
        length^3) of profiles through the isotropic topography,

        .. math ::

            C^\text{1D}(q) \approx \frac{q}{\pi} C^\text{iso}(q)
            \sqrt{1 - (q/q_s)^2}

        in the self-affine region and :math:`C_r q_r/\pi` in the roll-off
        region. The exact projection,
        :math:`C^\text{1D}(q_x) = \frac{1}{2\pi}\int dq_y\,
        C^\text{iso}(\sqrt{q_x^2+q_y^2})`, gives for a pure power law
        :math:`C_0 q^{-1-2H}\,\Gamma(H+1/2)/(2\sqrt{\pi}\,\Gamma(H+1))`;
        the approximation coincides with it for :math:`H=1/2` and
        :math:`q_s \to \infty`.

        Parameters
        ----------
        q : float or np.ndarray
            Wavevector.

        Returns
        -------
        psd : np.ndarray
            PSD at `q`.
        """
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
        Root mean-squared height. Integrating the isotropic PSD,
        :math:`h^2_\mathrm{rms} = \frac{1}{2\pi}\int_{q_L}^{q_s} q C^\text{iso}(q) dq`
        (see 10.1088/2051-672X/aa51f8 equations 4, 10 and 11), gives for
        :math:`q_L \le q_r \le q_s`

        .. math ::

            h^2_\mathrm{rms} = \frac{C_r q_r^2}{4 \pi}
            \left[ 1 + \frac{1}{H} - \left( \frac{q_L}{q_r}\right)^2
            - \frac{1}{H}\left( \frac{q_s}{q_r}\right)^{-2H}\right]

        Note that :math:`C_r q_r^2` can be interchanged with :math:`C_0 q_r^{-2H}`.
        This is :func:`variance_derivative` of order zero, which also covers
        the cases outside :math:`q_L \le q_r \le q_s`.

        Parameters
        ----------
        shortcut_wavelength: float, optional
            Makes the rms height scale dependent by overriding the short
            cutoff (only values larger than the model's short cutoff take
            effect). (Default: None, i.e. the model's value)
        longcut_wavelength: float, optional
            Makes the rms height scale dependent by overriding the long
            cutoff (only values smaller than the model's long cutoff take
            effect). (Default: None, i.e. the model's value)

        Returns
        -------
        rms_height : float

        """
        return np.sqrt(self.variance_derivative(0, shortcut_wavelength=shortcut_wavelength,
                                                longcut_wavelength=longcut_wavelength))

    def variance_derivative(self, order, shortcut_wavelength=None, longcut_wavelength=None, ):
        r"""
        Variance of a derivative of arbitrary (fractional) order
        :math:`\alpha`,

        .. math ::

            \left[h^{(\alpha)}_\mathrm{rms}\right]^2 = \frac{1}{2\pi}
                \int_{q_L}^{q_s} q^{1+2\alpha} C^\text{iso}(q) dq.

        The roll-off region and the self-affine region are integrated
        separately. The contribution for an isotropic flat PSD with
        amplitude :math:`C_r` between the wavevectors :math:`[q_L, q_s]` is

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
             Order :math:`\alpha` of the derivative.
        shortcut_wavelength: float, optional
            Makes the result scale dependent by overriding the short cutoff
            (only values larger than the model's short cutoff take effect).
        longcut_wavelength: float, optional
            Makes the result scale dependent by overriding the long cutoff
            (only values smaller than the model's long cutoff take effect).

        Returns
        -------
        variance : float

        """
        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength
        else:
            shortcut_wavelength = max(shortcut_wavelength, self.shortcut_wavelength)
        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength
        else:
            longcut_wavelength = min(longcut_wavelength, self.longcut_wavelength)

        shortcut_wavevector = (2 * np.pi) / shortcut_wavelength if shortcut_wavelength > 0 else np.inf
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
                # The self-affine region spans from the larger of the
                # rolloff and longcut wavevectors to the shortcut
                # wavevector, like in the general-order branch below
                self_affine = c0 / (2 * np.pi) * np.log(
                    shortcut_wavevector / max(rolloff_wavevector, longcut_wavevector))
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

    def generate_roughness(self,
                           pixel_size,
                           n_pixels,
                           seed,
                           shortcut_wavelength=None,
                           longcut_wavelength=None,
                           **kwargs):
        r"""
        Generate a square, periodic topography map whose expected PSD is the
        PSD of this model, using
        :func:`SurfaceTopography.Generation.fourier_synthesis`.

        The `shortcut_wavelength` and `longcut_wavelength` parameters allow
        to override the values parametrized in the model, since a discrete
        realization is often shorter.

        Parameters
        ----------
        pixel_size : float
            Grid spacing :math:`\Delta x = \Delta y`.
        n_pixels : int
            Number of grid points per direction.
        seed : int
            Seed for `np.random.seed`.
        shortcut_wavelength : float, optional
            Short-wavelength cutoff. (Default: the model's value)
        longcut_wavelength : float, optional
            Long-wavelength cutoff. (Default: the model's value)
        **kwargs
            Further arguments passed to `fourier_synthesis`.

        Returns
        -------
        topography : Topography
            The topography map.
        """

        if shortcut_wavelength is None:
            shortcut_wavelength = self.shortcut_wavelength

        if longcut_wavelength is None:
            longcut_wavelength = self.longcut_wavelength

        nx = n_pixels
        dx = pixel_size
        sx = nx * dx

        # Build a model with the (possibly overridden) cutoffs and hand its
        # PSD to the Fourier synthesis. This honors shortcut, rolloff *and*
        # longcut; passing `c0`/`long_cutoff` instead would only describe
        # the rolloff knee and silently ignore `longcut_wavelength`.
        model = SelfAffine(cr=self.cr,
                           rolloff_wavelength=self.rolloff_wavelength,
                           hurst_exponent=self.hurst_exponent,
                           longcut_wavelength=longcut_wavelength,
                           shortcut_wavelength=shortcut_wavelength,
                           unit=self.unit)

        np.random.seed(seed)
        roughness = fourier_synthesis((n_pixels, n_pixels), (sx, sx),
                                      psd=model.power_spectrum_isotropic,
                                      short_cutoff=shortcut_wavelength if shortcut_wavelength > 0 else None,
                                      unit=self.unit,
                                      **kwargs
                                      )
        return roughness
