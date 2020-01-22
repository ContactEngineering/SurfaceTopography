#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
Helper functions for the generation of random fractal surfaces
"""

import numpy as np
import scipy.stats as stats
from PyCo.Topography import Topography, UniformLineScan
from PyCo.Tools.common import compute_wavevectors, ifftn


# FIXME: Not sure topography generation should be classes. These should probably
# be turned into individual functions.

# FIXME: In contrast to what is explained in docstrings, this functions don't work for the generation of 1D data or 2D data with nx and ny diferent, tests are only for square nb_grid_pts
class RandomSurfaceExact(object):
    """ Metasurface with exact power spectrum"""
    Error = Exception

    def __init__(self, nb_grid_pts, physical_sizes, hurst, rms_height=None,
                 rms_slope=None, seed=None, lambda_min=None, lambda_max=None):
        """
        Generates a surface with an exact power spectrum (deterministic
        amplitude)
        Keyword Arguments:
        nb_grid_pts -- Tuple containing number of points in spatial directions.
                      The length of the tuple determines the spatial dimension
                      of the problem (for the time being, only 1D or square 2D)
        physical_sizes       -- domain physical_sizes. For multidimensional problems,
                      a tuple can be provided to specify the length per
                      dimension. If the tuple has less entries than dimensions,
                      the last value in repeated.
        hurst      -- Hurst exponent
        rms_height -- root mean square height of surface
        rms_slope  -- root mean square slope of surface
        seed       -- (default hash(None)) for repeatability, the random number
                      generator is seeded previous to outputting the generated
                      surface
        lambda_min -- (default None) min wavelength to consider when scaling
                      power spectral density
        lambda_max -- (default None) max wavelength to consider when scaling
                      power spectral density
        """
        if seed is not None:
            np.random.seed(hash(seed))
        if not hasattr(nb_grid_pts, "__iter__"):
            nb_grid_pts = (nb_grid_pts,)
        if not hasattr(physical_sizes, "__iter__"):
            physical_sizes = (physical_sizes,)

        self.dim = len(nb_grid_pts)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.nb_grid_pts = nb_grid_pts
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(physical_sizes[min(i, len(physical_sizes) - 1)])
        self.size = tuple(tmpsize)

        self.hurst = hurst

        if rms_height is None and rms_slope is None:
            raise self.Error('Please specify either rms height or rms slope.')
        if rms_height is not None and rms_slope is not None:
            raise self.Error('Please specify either rms height or rms slope, '
                             'not both.')

        self.rms_height = rms_height
        self.rms_slope = rms_slope
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        if lambda_max is not None:
            self.q_min = 2 * np.pi / lambda_max
        else:
            self.q_min = 2 * np.pi * max(1 / self.size[0], 1 / self.size[1])

        max_pixelsize = max(
            (siz / res for siz, res in zip(self.size, self.nb_grid_pts)))
        self.q_max = np.pi / max_pixelsize

        self.prefactor = self.compute_prefactor()

        self.q = compute_wavevectors(  # pylint: disable=invalid-name
            self.nb_grid_pts, self.size, self.dim)
        self.coeffs = self.generate_phases()
        self.generate_amplitudes()
        self.distribution = self.amplitude_distribution()
        self.active_coeffs = None

    def get_negative_frequency_iterator(self):
        " frequency complement"

        def iterator():  # pylint: disable=missing-docstring
            for i in range(self.nb_grid_pts[0]):
                for j in range(self.nb_grid_pts[1] // 2 + 1):
                    yield (i, j), (-i, -j)

        return iterator()

    def amplitude_distribution(self):  # pylint: disable=no-self-use
        """
        returns a multiplicative factor to apply to the fourier coeffs before
        computing the inverse transform (trivial in this case, since there's no
        stochastic distro in this case)
        """
        return 1.

    @property
    def C0(self):
        " prefactor of psd"
        return (self.compute_prefactor() / np.sqrt(np.prod(self.size))) ** 2

    @property
    def abs_q(self):
        " radial distances in q-space"
        q_norm = np.sqrt((self.q[0] ** 2).reshape((-1, 1)) + self.q[1] ** 2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return q_norm.flatten()[order][1:]

    @property
    def lambdas(self):
        " radial wavelengths in grid"
        return 2 * np.pi / self.abs_q

    def compute_prefactor(self):
        """
        computes the proportionality factor that determines the root mean
        square height assuming that the largest wave length is the full
        domain. This is described for the square of the factor on p R7
        """
        if self.lambda_min is not None:
            q_max = 2 * np.pi / self.lambda_min
        else:
            q_max = np.pi * min(self.nb_grid_pts[0] / self.size[0],
                                self.nb_grid_pts[1] / self.size[1])
        area = np.prod(self.size)
        if self.rms_height is not None:
            return 2 * self.rms_height / np.sqrt(
                self.q_min ** (-2 * self.hurst) - q_max ** (-2 * self.hurst)) * \
                   np.sqrt(self.hurst * np.pi * area)
        elif self.rms_slope is not None:
            return 2 * self.rms_slope / np.sqrt(
                q_max ** (2 - 2 * self.hurst) - self.q_min ** (2 - 2 * self.hurst)) * \
                   np.sqrt((1 - self.hurst) * np.pi * area)
        else:
            self.Error('Neither rms height nor rms slope is defined!')

    def generate_phases(self):
        """
        generates appropriate random phases (φ(-q) = -φ(q))
        """
        rand_phase = np.random.rand(*self.nb_grid_pts) * 2 * np.pi
        coeffs = np.exp(1j * rand_phase)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            if pos_it != (0, 0):
                coeffs[neg_it] = coeffs[pos_it].conj()
        if self.nb_grid_pts[0] % 2 == 0:
            r_2 = self.nb_grid_pts[0] // 2
            coeffs[r_2, 0] = coeffs[r_2, r_2] = coeffs[0, r_2] = 1
        return coeffs

    def generate_amplitudes(self):
        "compute an amplitude distribution"
        q_2 = self.q[0].reshape(-1, 1) ** 2 + self.q[1] ** 2
        q_2[0, 0] = 1  # to avoid div by zeros, needs to be fixed after
        self.coeffs *= (q_2) ** (-(1 + self.hurst) / 2) * self.prefactor
        self.coeffs[0, 0] = 0  # et voilà
        # Fix Shannon limit:
        self.coeffs[q_2 > self.q_max ** 2] = 0

    def get_topography(self, lambda_max=None, lambda_min=None, roll_off=1):
        """
        Computes and returs a NumpySurface object with the specified
        properties. This follows appendices A and B of Persson et al. (2005)

        Persson et al., On the nature of surface roughness with application to
        contact mechanics, sealing, rubber friction and adhesion, J. Phys.:
        Condens. Matter 17 (2005) R1-R62, http://arxiv.org/abs/cond-mat/0502419

        Keyword Arguments:
        lambda_max -- (default None) specifies a cutoff value for the longest
                      wavelength. By default, this is the domain physical_sizes in the
                      smallest dimension
        lambda_min -- (default None) specifies a cutoff value for the shortest
                      wavelength. by default this is determined by Shannon's
                      Theorem.
        """
        if lambda_max is None:
            lambda_max = self.lambda_max
        if lambda_min is None:
            lambda_min = self.lambda_min

        active_coeffs = self.coeffs.copy()
        q_square = self.q[0].reshape(-1, 1) ** 2 + self.q[1] ** 2
        if lambda_max is not None:
            q2_min = (2 * np.pi / lambda_max) ** 2
            # ampli_max = (self.prefactor*2*np.pi/self.physical_sizes[0] *
            #             q2_min**((-1-self.hurst)/2))
            ampli_max = (q2_min) ** (-(1 + self.hurst) / 2) * self.prefactor
            sl = q_square < q2_min
            ampli = abs(active_coeffs[sl])
            ampli[0] = 1
            active_coeffs[sl] *= roll_off * ampli_max / ampli
        if lambda_min is not None:
            q2_max = (2 * np.pi / lambda_min) ** 2
            active_coeffs[q_square > q2_max] = 0
        active_coeffs *= self.distribution
        area = np.prod(self.size)
        profile = ifftn(active_coeffs, area).real
        self.active_coeffs = active_coeffs
        return Topography(profile, self.size)


class RandomSurfaceGaussian(RandomSurfaceExact):
    """ Metasurface with Gaussian height distribution"""

    def __init__(self, nb_grid_pts, physical_sizes, hurst, rms_height=None,
                 rms_slope=None, seed=None, lambda_min=None, lambda_max=None):
        """
        Generates a surface with an Gaussian amplitude distribution
        Keyword Arguments:
        nb_grid_pts -- Tuple containing number of points in spatial directions.
                      The length of the tuple determines the spatial dimension
                      of the problem (for the time being, only 1D or square 2D)
        physical_sizes       -- domain physical_sizes. For multidimensional problems,
                      a tuple can be provided to specify the lenths per
                      dimension. If the tuple has less entries than dimensions,
                      the last value in repeated.
        hurst      -- Hurst exponent
        rms_height -- root mean square height of surface
        rms_slope  -- root mean square slope of surface
        seed       -- (default hash(None)) for repeatability, the random number
                      generator is seeded previous to outputting the generated
                      surface
        lambda_min -- (default None) min wavelength to consider when scaling
                      power spectral density
        lambda_max -- (default None) max wavelength to consider when scaling
                      power spectral density
        """
        super().__init__(nb_grid_pts, physical_sizes, hurst, rms_height=rms_height,
                         rms_slope=rms_slope, seed=seed, lambda_min=lambda_min,
                         lambda_max=lambda_max)

    def amplitude_distribution(self):
        """
        updates the amplitudes to be a Gaussian distribution around B(q) from
        Appendix B.
        """
        distr = stats.norm.rvs(size=self.coeffs.shape)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            distr[neg_it] = distr[pos_it]
        return distr


class CapillaryWavesExact(object):
    """Frozen capillary waves"""
    Error = Exception

    def __init__(self, nb_grid_pts, physical_sizes, mass_density, surface_tension,
                 bending_stiffness, seed=None):
        """
        Generates a surface with an exact power spectrum (deterministic
        amplitude)
        Keyword Arguments:
        nb_grid_pts        -- Tuple containing number of points in spatial directions.
                             The length of the tuple determines the spatial dimension
                             of the problem (for the time being, only 1D or square 2D)
        physical_sizes              -- domain physical_sizes. For multidimensional problems,
                             a tuple can be provided to specify the length per
                             dimension. If the tuple has less entries than dimensions,
                             the last value in repeated.
        mass_density      -- Mass density
        surface_tension   -- Topography tension
        bending_stiffness -- Bending stiffness
        rms_height        -- root mean square height of surface
        rms_slope         -- root mean square slope of surface
        seed              -- (default hash(None)) for repeatability, the random number
                             generator is seeded previous to outputting the generated
                             surface
        """
        if seed is not None:
            np.random.seed(hash(seed))
        if not hasattr(nb_grid_pts, "__iter__"):
            nb_grid_pts = (nb_grid_pts,)
        if not hasattr(physical_sizes, "__iter__"):
            physical_sizes = (physical_sizes,)

        self.dim = len(nb_grid_pts)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        self.nb_grid_pts = nb_grid_pts
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(physical_sizes[min(i, len(physical_sizes) - 1)])
        self.size = tuple(tmpsize)

        self.mass_density = mass_density
        self.surface_tension = surface_tension
        self.bending_stiffness = bending_stiffness

        max_pixelsize = max(
            (siz / res for siz, res in zip(self.size, self.nb_grid_pts)))

        self.q = compute_wavevectors(  # pylint: disable=invalid-name
            self.nb_grid_pts, self.size, self.dim)
        self.coeffs = self.generate_phases()
        self.generate_amplitudes()
        self.distribution = self.amplitude_distribution()
        self.active_coeffs = None

    def get_negative_frequency_iterator(self):
        " frequency complement"

        def iterator():  # pylint: disable=missing-docstring
            for i in range(self.nb_grid_pts[0]):
                for j in range(self.nb_grid_pts[1] // 2 + 1):
                    yield (i, j), (-i, -j)

        return iterator()

    def amplitude_distribution(self):  # pylint: disable=no-self-use
        """
        returns a multiplicative factor to apply to the fourier coeffs before
        computing the inverse transform (trivial in this case, since there's no
        stochastic distro in this case)
        """
        return 1.

    @property
    def abs_q(self):
        " radial distances in q-space"
        q_norm = np.sqrt((self.q[0] ** 2).reshape((-1, 1)) + self.q[1] ** 2)
        order = np.argsort(q_norm, axis=None)
        # The first entry (for |q| = 0) is rejected, since it's 0 by construct
        return q_norm.flatten()[order][1:]

    def generate_phases(self):
        """
        generates appropriate random phases (φ(-q) = -φ(q))
        """
        rand_phase = np.random.rand(*self.nb_grid_pts) * 2 * np.pi
        coeffs = np.exp(1j * rand_phase)
        for pos_it, neg_it in self.get_negative_frequency_iterator():
            if pos_it != (0, 0):
                coeffs[neg_it] = coeffs[pos_it].conj()
        if self.nb_grid_pts[0] % 2 == 0:
            r_2 = self.nb_grid_pts[0] // 2
            coeffs[r_2, 0] = coeffs[r_2, r_2] = coeffs[0, r_2] = 1
        return coeffs

    def generate_amplitudes(self):
        "compute an amplitude distribution"
        q_2 = self.q[0].reshape(-1, 1) ** 2 + self.q[1] ** 2
        q_2[0, 0] = 1  # to avoid div by zeros, needs to be fixed after
        self.coeffs *= 1 / (self.mass_density + self.surface_tension * q_2 +
                            self.bending_stiffness * q_2 * q_2)
        self.coeffs[0, 0] = 0  # et voilà
        # Fix Shannon limit:
        # self.coeffs[q_2 > self.q_max**2] = 0

    def get_topography(self):
        """
        Computes and returs a NumpySurface object with the specified
        properties. This follows appendices A and B of Persson et al. (2005)

        Persson et al., On the nature of surface roughness with application to
        contact mechanics, sealing, rubber friction and adhesion, J. Phys.:
        Condens. Matter 17 (2005) R1-R62, http://arxiv.org/abs/cond-mat/0502419

        Keyword Arguments:
        lambda_max -- (default None) specifies a cutoff value for the longest
                      wavelength. By default, this is the domain physical_sizes in the
                      smallest dimension
        lambda_min -- (default None) specifies a cutoff value for the shortest
                      wavelength. by default this is determined by Shannon's
                      Theorem.
        """
        active_coeffs = self.coeffs.copy()
        active_coeffs *= self.distribution
        area = np.prod(self.size)
        profile = ifftn(active_coeffs, area).real
        self.active_coeffs = active_coeffs
        return Topography(profile, self.size)


###

def _irfft2(karr, rarr, progress_callback=None):
    """
    Inverse 2d real-to-complex FFT with callback that allows to report status
    of the computation to user.

    Parameters
    ----------
    karr : array_like
        Fourier-space representation
    rarr : array_like
        Real-space representation
    progress_callback : function(i, n)
        Function that is called to report progress.
    """
    nrows, ncolumns = karr.shape
    for i in range(ncolumns):
        if progress_callback is not None:
            progress_callback(i, ncolumns + nrows - 1)
        karr[:, i] = np.fft.ifft(karr[:, i])
    for i in range(nrows):
        if progress_callback is not None:
            progress_callback(i + ncolumns, ncolumns + nrows - 1)
        if rarr.shape[1] % 2 == 0:
            rarr[i, :] = np.fft.irfft(karr[i, :])
        else:
            rarr[i, :] = np.fft.irfft(karr[i, :], n=rarr.shape[1])


def self_affine_prefactor(nb_grid_pts, physical_sizes, Hurst, rms_height=None,
                          rms_slope=None, short_cutoff=None, long_cutoff=None):
    r"""
    Compute prefactor :math:`C_0` for the power-spectrum density of an ideal
    self-affine topography given by

    .. math ::

        C(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.

    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map.
    physical_sizes : array_like
        Physical physical_sizes of the topography map.
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.

    Returns
    -------
    prefactor : float
        Prefactor :math:`C_0`
    """
    nb_grid_pts = np.asarray(nb_grid_pts)
    physical_sizes = np.asarray(physical_sizes)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(nb_grid_pts / physical_sizes)

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = 2 * np.pi * np.max(1 / physical_sizes)

    area = np.prod(physical_sizes)
    if rms_height is not None:
        fac = 2 * rms_height / np.sqrt(q_min ** (-2 * Hurst) -
                                       q_max ** (-2 * Hurst)) * np.sqrt(Hurst * np.pi)
    elif rms_slope is not None:
        fac = 2 * rms_slope / np.sqrt(q_max ** (2 - 2 * Hurst) -
                                      q_min ** (2 - 2 * Hurst)) * np.sqrt((1 - Hurst) * np.pi)
    else:
        raise ValueError('Neither rms height nor rms slope is defined!')
    return fac * np.prod(nb_grid_pts) / np.sqrt(area)


def fourier_synthesis(nb_grid_pts, physical_sizes, hurst,
                      rms_height=None, rms_slope=None, c0=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0,
                      amplitude_distribution=lambda n: np.random.normal(size=n),
                      rfn=None, kfn=None, progress_callback=None):
    r"""
    Create a self-affine, randomly rough surface using a Fourier filtering
    algorithm. The algorithm is described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map.
    physical_sizes : array_like
        Physical physical_sizes of the topography map.
    hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    c0: float
        self affine prefactor :math:`C_0`:
        :math:`C(q) = C_0 q^{-2-2H}`
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.
    rolloff : float
        Value for the power-spectral density (PSD) below the long-wavelength
        cutoff. This multiplies the value at the cutoff, i.e. unit will give a
        PSD that is flat below the cutoff, zero will give a PSD that is vanishes
        below cutoff. (Default: 1.0)
    amplitude_distribution : function
        Function that generates the distribution of amplitudes.
        (Default: np.random.normal)
    rfn : str
        Name of file that stores the real-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    kfn : str
        Name of file that stores the Fourie-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    progress_callback : function(i, n)
        Function that is called to report progress.

    Returns
    -------
    topography : UniformTopography or UniformLineScan
        The topography.
    """
    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts) / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    if c0 is None:
        fac = self_affine_prefactor(nb_grid_pts, physical_sizes, hurst, rms_height=rms_height,
                                    rms_slope=rms_slope, short_cutoff=short_cutoff,
                                    long_cutoff=long_cutoff)
    else:
        fac = c0 * np.prod(nb_grid_pts) / np.sqrt(np.prod(physical_sizes))
        #          ^                       ^ C(q) = c0 q^(-2-2H) = 1 / A |fh(q)|^2
        #          |                         and h(x,y) = sum(1/A fh(q) e^(iqx)))
        #          compensate for the np.fft normalisation

    if len(nb_grid_pts) == 2:
        nx, ny = nb_grid_pts
        sx, sy = physical_sizes
        kny = ny // 2 + 1
        kshape = (nx, kny)
    else:
        nx = 1
        ny, = nb_grid_pts
        sx = 1
        sy, = physical_sizes
        kny = ny // 2 + 1
        kshape = (kny,)

    # Create in-memory or memory-mapped arrays as storage buffers
    if rfn is None:
        rarr = np.empty(nb_grid_pts, dtype=np.float64)
    else:
        rarr = np.memmap(rfn, np.float64, 'w+', shape=nb_grid_pts)
    if kfn is None:
        karr = np.empty(kshape, dtype=np.complex128)
    else:
        karr = np.memmap(kfn, np.complex128, 'w+', shape=kshape)

    qy = 2 * np.pi * np.arange(kny) / sy
    for x in range(nx):
        if progress_callback is not None:
            progress_callback(x, nx - 1)
        if x > nx // 2:
            qx = 2 * np.pi * (nx - x) / sx
        else:
            qx = 2 * np.pi * x / sx
        q_sq = qx ** 2 + qy ** 2
        if x == 0:
            q_sq[0] = 1.
        phase = np.exp(2 * np.pi * np.random.rand(kny) * 1j)
        ran = fac * phase * amplitude_distribution(kny)
        if len(nb_grid_pts) == 2:
            karr[x, :] = ran * q_sq ** (-(1 + hurst) / 2)
            karr[x, q_sq > q_max ** 2] = 0.
        else:
            karr[:] = ran * q_sq ** (-(0.5 + hurst) / 2)
            karr[q_sq > q_max ** 2] = 0.
        if q_min is not None:
            mask = q_sq < q_min ** 2
            karr[x, mask] = rolloff * ran[mask] * q_min ** (-(1 + hurst))
    if len(nb_grid_pts) == 2:
        for iy in [0, -1] if ny % 2 == 0 else [0]:
            # Enforce symmetry
            if nx % 2 == 0:
                karr[0, iy] = np.real(karr[0, iy])
                karr[nx // 2, iy] = np.real(karr[nx // 2, iy])
                karr[1:nx // 2, iy] = karr[-1:nx // 2:-1, iy].conj()
            else:
                karr[0, iy] = np.real(karr[0, iy])
                karr[1:nx // 2 + 1, iy] = karr[-1:nx // 2:-1, iy].conj()
        _irfft2(karr, rarr, progress_callback)
        return Topography(rarr, physical_sizes, periodic=True)
    else:
        karr[0] = np.real(karr[0])
        rarr[:] = np.fft.irfft(karr)
        return UniformLineScan(rarr, sy, periodic=True)
