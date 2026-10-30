#
# Copyright 2015-2016, 2018-2022 Lars Pastewka
#           2018-2021 Antoine Sanner
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
Generation of random self-affine (fractal) topographies by Fourier synthesis
"""

import numpy as np
from . import Topography, UniformLineScan


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
                          rms_slope=None, short_cutoff=None, long_cutoff=None, rolloff=1.0):
    r"""
    Compute the amplitude prefactor used by `fourier_synthesis` for a
    self-affine topography with power-spectral density (PSD)

    .. math ::

        C^\text{2D}(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C^\text{1D}(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.
    The PSD is zero above the short-wavelength cutoff :math:`q_s` and equal
    to :math:`\text{rolloff}^2\, C(q_L)` below the long-wavelength cutoff
    :math:`q_L`, as in `fourier_synthesis`.

    :math:`C_0` is determined such that the expectation value of the mean
    square height (or of the mean square slope) of the generated topography
    equals the requested value. This is the sum over the wavevectors of the
    grid,

    .. math ::

        \langle h^2 \rangle = \frac{1}{A} \sum_{\mathbf q} C(q),
        \qquad
        \langle |\nabla h|^2 \rangle = \frac{1}{A} \sum_{\mathbf q} q^2 C(q),

    with :math:`A = L_x L_y` (1D: :math:`L`), not the continuum integral
    :math:`\frac{1}{2\pi}\int q\,C(q)\,dq` that it approaches for
    :math:`q_L \gg 2\pi/L`. The two differ most where few wavevectors carry
    most of the variance, i.e. for the rms height near the long-wavelength
    cutoff. Coefficients that must be real (the Nyquist wavevectors) carry
    on average half of the power of the others; this is included in the
    sum. The slope is the spectral (not a finite-difference) slope.

    Parameters
    ----------
    nb_grid_pts : array_like
        Number of grid points :math:`(N_x, N_y)` or :math:`(N,)`.
    physical_sizes : array_like
        Physical size :math:`(L_x, L_y)` or :math:`(L,)`.
    Hurst : float
        Hurst exponent :math:`H`.
    rms_height : float, optional
        Root mean-squared height. (Default: None)
    rms_slope : float, optional
        Root mean-squared slope. Used if `rms_height` is None.
        (Default: None)
    short_cutoff : float, optional
        Short-wavelength cutoff :math:`\lambda_s = 2\pi/q_s`.
        (Default: None, i.e. :math:`q_s = \pi/\Delta x`, using the smallest
        Nyquist wavevector of the two directions)
    long_cutoff : float, optional
        Long-wavelength cutoff :math:`\lambda_L = 2\pi/q_L`.
        (Default: None, i.e. power law down to the smallest nonzero
        wavevector)
    rolloff : float, optional
        Amplitude below the long-wavelength cutoff relative to the
        amplitude at the cutoff, see `fourier_synthesis`. (Default: 1.0)

    Returns
    -------
    prefactor : float
        Prefactor :math:`\sqrt{C_0}\, N_x N_y / \sqrt{L_x L_y}` (1D:
        :math:`\sqrt{C_0}\, N / \sqrt{L}`) of the Fourier amplitudes in
        numpy's unnormalized FFT convention.

    References
    ----------
    [1]: Jacobs, Junge, Pastewka,
    Surf. Topogr.: Metrol. Prop. 5, 013001 (2017)
    """
    nb_grid_pts = tuple(int(n) for n in np.atleast_1d(nb_grid_pts))
    physical_sizes = tuple(float(s) for s in np.atleast_1d(physical_sizes))

    if rms_height is not None:
        target = rms_height
        slope_order = 0
    elif rms_slope is not None:
        target = rms_slope
        slope_order = 1
    else:
        raise ValueError('Neither rms height nor rms slope is defined!')

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts) / np.asarray(physical_sizes))
    q_min = None if long_cutoff is None else 2 * np.pi / long_cutoff

    # Exponent of the PSD: C^2D ~ q^(-2-2H), C^1D ~ q^(-1-2H)
    exponent = -len(nb_grid_pts) - 2 * Hurst

    if len(nb_grid_pts) == 2:
        nx, ny = nb_grid_pts
        sx, sy = physical_sizes
    else:
        nx, sx = 1, 1.0
        ny, = nb_grid_pts
        sy, = physical_sizes
    kny = ny // 2 + 1

    # Each coefficient of the half-complex array (along y) stands for two
    # wavevectors, +q and -q, except the q_y = 0 column and, for even ny, the
    # Nyquist column. Coefficients that are their own complex conjugate are
    # made real, which halves their expected power.
    column_weight = np.full(kny, 2.0)
    column_weight[0] = 1
    if ny % 2 == 0:
        column_weight[-1] = 1
    self_conjugate_columns = [0, kny - 1] if ny % 2 == 0 else [0]
    self_conjugate_rows = [0, nx // 2] if nx % 2 == 0 else [0]

    qy = 2 * np.pi * np.arange(kny) / sy
    weighted_sum = 0.0
    for x in range(nx):
        qx = 2 * np.pi * min(x, nx - x) / sx
        q_sq = qx ** 2 + qy ** 2
        psd = np.zeros(kny)
        inside = np.logical_and(q_sq > 0, q_sq <= q_max ** 2)
        psd[inside] = q_sq[inside] ** (exponent / 2)
        if q_min is not None:
            below = np.logical_and(inside, q_sq < q_min ** 2)
            psd[below] = rolloff ** 2 * q_min ** exponent
        weight = column_weight.copy()
        if x in self_conjugate_rows:
            weight[self_conjugate_columns] /= 2
        weighted_sum += np.sum(weight * psd * q_sq ** slope_order)

    # <h^2> = fac^2 / (N_x N_y)^2 * weighted_sum for unit amplitudes
    return target * nx * ny / np.sqrt(weighted_sum)


def fourier_synthesis(nb_grid_pts, physical_sizes,
                      hurst=None,
                      rms_height=None, rms_slope=None, c0=None,
                      psd=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0,
                      amplitude_distribution=None,
                      periodic=True, rfn=None, kfn=None,
                      progress_callback=None, unit=None, info={}, rng=None):
    r"""
    Create a randomly rough topography with prescribed power-spectral
    density (PSD) by Fourier filtering of random noise. The algorithm is
    described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topogr.: Metrol. Prop. 5, 013001 (2017)

    The Fourier coefficient at each wavevector :math:`\mathbf q` of the
    grid is

    .. math ::

        \tilde h(\mathbf q) = \sqrt{A\, C(q)}\; a_\mathbf{q}\; e^{i\phi_\mathbf{q}}

    with :math:`A = L_x L_y` (1D: :math:`L`), random amplitudes
    :math:`a_\mathbf{q}` drawn from `amplitude_distribution` and uniformly
    distributed random phases :math:`\phi_\mathbf{q} \in [0, 2\pi)`. For
    :math:`\langle a^2 \rangle = 1` the expectation value of the PSD of the
    result is :math:`C(q)`. The PSD is either self-affine,
    :math:`C(q) = C_0 q^{-2-2H}` (1D: :math:`C_0 q^{-1-2H}`) with
    :math:`C_0` given or computed from `rms_height` or `rms_slope` (see
    `self_affine_prefactor`), or the function `psd`. It is zero for
    :math:`q > q_s` and, if `long_cutoff` is given, constant and equal to
    :math:`\text{rolloff}^2\, C(q_L)` for :math:`q < q_L`. For the
    self-affine PSD the :math:`q = 0` coefficient is zero, i.e. the mean
    height vanishes.

    The real-space heights are obtained by an inverse real-to-complex FFT;
    the cost is :math:`O(N_x N_y \log(N_x N_y))`. The result is always
    periodic by construction.

    Parameters
    ----------
    nb_grid_pts : array_like
        Number of grid points, :math:`(N_x, N_y)` for a topography map or
        :math:`(N,)` for a line scan.
    physical_sizes : array_like
        Physical size, :math:`(L_x, L_y)` or :math:`(L,)`.
    hurst : float, optional
        Hurst exponent :math:`H`. Required unless `psd` is given.
        (Default: None)
    rms_height : float, optional
        Root mean-squared height. (Default: None)
    rms_slope : float, optional
        Root mean-squared slope. Used if `rms_height` is None.
        (Default: None)
    c0 : float, optional
        Self-affine prefactor :math:`C_0`. Used if neither `rms_height` nor
        `rms_slope` is given. (Default: None)
    psd : callable, optional
        Function returning the PSD :math:`C(q)` (2D: :math:`C^\text{2D}`,
        1D: :math:`C^\text{1D}`) as a function of the magnitude of the
        wavevector. It is also evaluated at :math:`q = 0`. Cannot be
        combined with `hurst`, `rms_height`, `rms_slope`, `c0` or
        `long_cutoff`. (Default: None)
    short_cutoff : float, optional
        Short-wavelength cutoff :math:`\lambda_s = 2\pi/q_s`.
        (Default: None, i.e. :math:`q_s = \pi/\Delta x`)
    long_cutoff : float, optional
        Long-wavelength cutoff :math:`\lambda_L = 2\pi/q_L`. (Default: None,
        i.e. power law down to the smallest nonzero wavevector)
    rolloff : float, optional
        Amplitude of the Fourier coefficients below the long-wavelength
        cutoff relative to the amplitude at the cutoff; the PSD for
        :math:`q < q_L` is :math:`\text{rolloff}^2\, C(q_L)`. Unity gives a
        PSD that is flat below the cutoff, zero a PSD that vanishes below
        the cutoff. (Default: 1.0)
    amplitude_distribution : function, optional
        Function that returns `n` random amplitudes :math:`a_\mathbf{q}`.
        (Default: None, i.e. standard normal distribution drawn from `rng`)
    periodic : bool, optional
        Value of the `is_periodic` flag of the returned topography. The
        heights are periodic in either case. (Default: True)
    rfn : str, optional
        Name of file that stores the real-space array. If specified, the
        real-space array will be created as a memory mapped file. This is
        useful for creating very large topography maps. (Default: None)
    kfn : str, optional
        Name of file that stores the Fourier-space array. If specified, the
        Fourier-space array will be created as a memory mapped file. This is
        useful for creating very large topography maps. (Default: None)
    progress_callback : function(i, n), optional
        Function that is called to report progress. (Default: None)
    unit : str, optional
        Length unit. (Default: None)
    info : dict, optional
        Initial info dictionary. (Default: {})
    rng : int or np.random.Generator, optional
        Seed or generator for the random phases and the default amplitudes.
        (Default: None, i.e. numpy's global random state, `np.random`)

    Returns
    -------
    topography : Topography or UniformLineScan
        The topography.
    """
    if rng is None:
        # Global random state, such that np.random.seed controls the result
        rng = np.random
    else:
        rng = np.random.default_rng(rng)
    if amplitude_distribution is None:
        def amplitude_distribution(n):
            return rng.normal(size=n)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts) / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    if psd and (long_cutoff or hurst or rms_slope or rms_height or c0):
        raise NotImplementedError("long_cutoff, hurst, rms_slope, rms_height, c0 "
                                  "cannot be combined with prescribed PSD")

    if psd is not None:
        fac = np.prod(nb_grid_pts) / np.sqrt(np.prod(physical_sizes))
    elif c0 is None:
        fac = self_affine_prefactor(nb_grid_pts, physical_sizes, hurst, rms_height=rms_height,
                                    rms_slope=rms_slope, short_cutoff=short_cutoff,
                                    long_cutoff=long_cutoff, rolloff=rolloff)
    else:
        # prefactor for the fourier heights
        fac = np.sqrt(c0) * np.prod(nb_grid_pts) / np.sqrt(np.prod(physical_sizes))
        #                   ^                       ^ C(q) = c0 q^(-2-2H) = 1 / A |fh(q)|^2
        #                   |                         and h(x,y) = sum(1/A fh(q) e^(iqx)))
        #                   compensate for the np.fft normalisation

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
        rarr = np.empty(nb_grid_pts, dtype=float)
    else:
        rarr = np.memmap(rfn, float, 'w+', shape=nb_grid_pts)
    if kfn is None:
        karr = np.empty(kshape, dtype=complex)
    else:
        karr = np.memmap(kfn, complex, 'w+', shape=kshape)

    qy = 2 * np.pi * np.arange(kny) / sy
    for x in range(nx):
        if progress_callback is not None:
            progress_callback(x, nx - 1)
        if x > nx // 2:
            qx = 2 * np.pi * (nx - x) / sx
        else:
            qx = 2 * np.pi * x / sx
        q_sq = qx ** 2 + qy ** 2
        if x == 0 and not psd:
            q_sq[0] = 1.
        phase = np.exp(2 * np.pi * rng.random(kny) * 1j)
        ran = fac * phase * amplitude_distribution(kny)
        if psd is None:
            if len(nb_grid_pts) == 2:
                karr[x, :] = ran * q_sq ** (-(1 + hurst) / 2)
                karr[x, q_sq > q_max ** 2] = 0.
            else:
                karr[:] = ran * q_sq ** (-(0.5 + hurst) / 2)
                karr[q_sq > q_max ** 2] = 0.

        else:
            if len(nb_grid_pts) == 2:
                karr[x, :] = ran * np.sqrt(psd(np.sqrt(q_sq)))
                karr[x, q_sq > q_max ** 2] = 0.
            else:
                karr[:] = ran * np.sqrt(psd(np.sqrt(q_sq)))
                karr[q_sq > q_max ** 2] = 0.

        if q_min is not None:
            mask = q_sq < q_min ** 2
            if len(nb_grid_pts) == 2:
                karr[x, mask] = rolloff * ran[mask] * q_min ** (-(1 + hurst))
            else:
                karr[mask] = rolloff * ran[mask] * q_min ** (-(0.5 + hurst))
        if x == 0 and psd is None:
            # The q=0 mode only controls the mean height, which we fix to
            # zero. (Setting q_sq[0] to 1 above merely avoids a
            # divide-by-zero; without zeroing this mode the surface would
            # acquire a random mean offset of magnitude C(q=1), which is
            # unit-dependent and typically dwarfs the rms height.)
            if len(nb_grid_pts) == 2:
                karr[0, 0] = 0.
            else:
                karr[0] = 0.
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
        return Topography(rarr, physical_sizes, periodic=periodic, unit=unit, info=info)
    else:
        karr[0] = np.real(karr[0])
        rarr[:] = np.fft.irfft(karr, n=ny)
        return UniformLineScan(rarr, sy, periodic=periodic, unit=unit, info=info)
