#
# Copyright 2020 Lars Pastewka
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

from SurfaceTopography import Topography
import numpy as np


def highcut(topography, cutoff_wavevector=None, cutoff_wavelength=None,
            kind="circular step"):
    r"""Applies a highcut filter to the topography using fft.

    for `kind=="circular step"` (default), parts of the spectrum with
    `|q| > cutoff_wavevector` are set to zero

    for `kind=="square step"`, parts of the spectrum with
    `q_x > cutoff_wavevector or q_y > cutoff_wavevector ` are set to zero

    either `cutoff_wavelength` or
    `cutoff_wavevector` :math:`= 2 pi /` `cutoff_wavelength`
    have to be provided.

    Parameters
    ----------
    topography: Topography
    cutoff_wavevector: float
    highest wavevector
    cutoff_wavelength: float
    shortest wavelength
    kind: {"circular step", "square step"}

    Returns
    -------
    Topography with filtered heights

    Examples
    --------
    >>> highcut(topography, cutoff_wavevector=2 * np.pi / l)
    >>> highcut(topography, cutoff_wavelength=l) # equivalent

    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    # dx, dy =  [r / s for r,s in zip(topography.nb_grid_pts,
    # topography.physical_sizes)]
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    qx = np.arange(0, nx, dtype=np.float64)
    qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
    qx *= 2 * np.pi

    qy = np.arange(0, ny // 2 + 1, dtype=np.float64)
    qy *= 2 * np.pi / sy

    q2 = (qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1)
    print(q2.shape)
    # square of the norm of the wavevector

    if cutoff_wavevector is None:
        if cutoff_wavelength is not None:
            cutoff_wavevector = 2 * np.pi / cutoff_wavelength
        else:
            raise ValueError(
                "cutoff_wavevector or cutoff_wavelength should be provided")
    elif cutoff_wavelength is not None:
        raise ValueError(
            "cutoff_wavevector or cutoff_wavelength should be provided")

    filt = np.ones_like(q2)

    if kind == "circular step":
        filt *= (q2 <= cutoff_wavevector ** 2)
    elif kind == "square step":
        filt *= (np.abs(qx.reshape(-1, 1)) <= cutoff_wavevector) * (
                np.abs(qy.reshape(1, -1)) <= cutoff_wavevector)

    h_qs = np.fft.irfftn(np.fft.rfftn(topography.heights()) * filt)

    return Topography(h_qs, physical_sizes=topography.physical_sizes)


def lowcut(topography, cutoff_wavevector=None, cutoff_wavelength=None,
           kind="circular step"):
    r"""Applies a lowcut filter to the topography using fft.

    for `kind=="circular step"` (default), parts of the spectrum with
    `|q| < cutoff_wavevector` are set to zero

    for `kind=="square step"`, parts of the spectrum with
    `q_x < cutoff_wavevector or q_y < cutoff_wavevector ` are set to zero

    either `cutoff_wavelength` or
    `cutoff_wavevector` :math:`= 2 pi /` `cutoff_wavelength`
    have to be provided.

    Parameters
    ----------
    topography: Topography
    cutoff_wavevector: float
    highest wavevector
    cutoff_wavelength: float
    shortest wavelength
    kind: {"circular step", "square step"}

    Returns
    -------
    Topography with filtered heights

    Examples
    --------
    >>> lowcut(topography, cutoff_wavevector=2 * np.pi / l)
    >>> lowcut(topography, cutoff_wavelength=l) # equivalent

    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    # dx, dy =  [r / s for r,s in zip(topography.nb_grid_pts,
    # topography.physical_sizes)]
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    qx = np.arange(0, nx, dtype=np.float64)
    qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
    qx *= 2 * np.pi

    qy = np.arange(0, ny // 2 + 1, dtype=np.float64)
    qy *= 2 * np.pi / sy

    q2 = (qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1)
    print(q2.shape)
    # square of the norm of the wavevector

    if cutoff_wavevector is None:
        if cutoff_wavelength is not None:
            cutoff_wavevector = 2 * np.pi / cutoff_wavelength
        else:
            raise ValueError(
                "cutoff_wavevector or cutoff_wavelength should be provided")
    elif cutoff_wavelength is not None:
        raise ValueError(
            "cutoff_wavevector or cutoff_wavelength should be provided")

    filt = np.ones_like(q2)

    if kind == "circular step":
        filt *= (q2 >= cutoff_wavevector ** 2)
    elif kind == "square step":
        filt *= (np.abs(qx.reshape(-1, 1)) >= cutoff_wavevector) * (
                    np.abs(qy.reshape(1, -1)) >= cutoff_wavevector)

    h_qs = np.fft.irfftn(np.fft.rfftn(topography.heights()) * filt)

    return Topography(h_qs, physical_sizes=topography.physical_sizes)


def isotropic_filter(topography, filter_function=lambda q: np.exp(-q)):
    r"""Multiplies filter_function(|q|) to the spectrum of the topography
    (q is the wavevector)

    returns

    ..math :: h^f_{ij} = FFT^-1(f(|q_{kl}|) FFT(h)_{kl})_{ij}

    with :math:`f` the `filter_function`

    Parameters
    ----------
    topography: Topography
    filter_function:
    function of the absolute value of the wavevector |q|

    Returns
    -------
    Topography with the modified heights
    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    sx, sy = topography.physical_sizes
    nx, ny = topography.nb_grid_pts
    qx = 2 * np.pi * np.fft.fftfreq(nx, sx / nx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, sy / ny).reshape(1, -1)

    q = np.sqrt(qx ** 2 + qy ** 2)
    h = topography.heights()
    h_q = np.fft.fft2(h)
    h_q_filtered = np.fft.ifft2(h_q * filter_function(q))

    # Max_imaginary = np.max(np.imag(shifted_pot))
    # assert Max_imaginary < 1e-14 *np.max(np.real(shifted_pot)) ,
    # f"{Max_imaginary}"

    return Topography(np.real(h_q_filtered),
                      physical_sizes=topography.physical_sizes)


Topography.register_function("isotropic_filter", isotropic_filter)
Topography.register_function("highcut", highcut)
Topography.register_function("lowcut", lowcut)
