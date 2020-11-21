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
from SurfaceTopography.UniformLineScanAndTopography import DecoratedUniformTopography
from SurfaceTopography.HeightContainer import UniformTopographyInterface
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
    q = np.sqrt((qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1))
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

    #filt = np.ones_like(q2)

    if kind == "circular step":
        filt = (q <= cutoff_wavevector)
    elif kind == "square step":
        filt = (np.abs(qx.reshape(-1, 1)) <= cutoff_wavevector) * (
                np.abs(qy.reshape(1, -1)) <= cutoff_wavevector)
    else:
        raise ValueError
    h_qs = np.fft.irfftn(np.fft.rfftn(topography.heights()) * filt)

    return Topography(h_qs, physical_sizes=topography.physical_sizes, periodic=True)

class LongCutTopography(DecoratedUniformTopography):
    name = 'longcut_filtered_topography'

    def __init__(self, topography,
                 cutoff_wavevector=None, cutoff_wavelength=None,
                 kind="circular step",
                 info={}):
        r"""Applies a long wavelength cut filter to the topography using fft.

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
        >>> topography.lowcut(cutoff_wavevector=2 * np.pi / l)
        >>> topography.lowcut(cutoff_wavelength=l) # equivalent

        """
        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")
        super().__init__(topography, info=info)

        if cutoff_wavelength is None:
            if cutoff_wavevector is not None:
                cutoff_wavelength = 2 * np.pi / cutoff_wavevector
            else:
                raise ValueError(
                    "cutoff_wavevector or cutoff_wavelength should be provided")
        elif cutoff_wavevector is not None:
            raise ValueError(
                "cutoff_wavevector or cutoff_wavelength should be provided")

        self._cutoff_wavelength = cutoff_wavelength
        self._kind = kind

    @property
    def cutoff_wavevector(self):
        return 2 * np.pi / self._cutoff_wavelength

    @property
    def cutoff_wavelength(self):
        return self._cutoff_wavelength

    @property
    def filter_function(self, qx, qy):
        if self._kind == "circular step":
            return ((qx**2 + qy**2) >= self.cutoff_wavevector ** 2)
        elif self._kind == "square step":
            return (np.abs(qx) >= self.cutoff_wavevector) * (
                        np.abs(qy) >= self.cutoff_wavevector)

    def heights(self):
        # dx, dy =  [r / s for r,s in zip(topography.nb_grid_pts,
        # topography.physical_sizes)]
        nx, ny = self.parent_topography.nb_grid_pts
        sx, sy = self.parent_topography.physical_sizes

        qx = np.arange(0, nx, dtype=np.float64).reshape(-1, 1)
        qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
        qx *= 2 * np.pi

        qy = np.arange(0, ny // 2 + 1, dtype=np.float64).reshape(1, -1)
        qy *= 2 * np.pi / sy

        h_qs = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) * self.filter_function(qx, qy))

        return h_qs


class FilteredTopography(DecoratedUniformTopography):
    name = 'filtered_topography'

    def __init__(self, topography,
                 filter_function=lambda qx, qy: (np.abs(qx) <= 1 )* np.abs(qy) <= 1,
                 isotropic=True,
                 info={}):

        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")
        super().__init__(topography, info=info)

        self._filter_function = filter_function
        self._is_filter_isotropic = isotropic # TODO: should be deductible from the filter function signature

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), \
            self._filter_function, self._is_filter_isotropic
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._filter_function, self._is_filter_isotropic = state
        super().__setstate__(superstate)

    @property
    def is_filter_isotropic(self):
        return self._is_filter_isotropic

    def filter_function(self, *args):
        """

        Parameters
        ----------
        if dim = 2 and filter is not isotropic
            qx, qy
        if dim = 1
            q
        """

        if self.dim==2 and not self.is_filter_isotropic\
            and len(args) != 2:
            raise("ValueError: qx, qy expected")
        elif self.dim == 1 and len(args) != 1:
            raise("ValueError: q expected")

        return self._filter_function(*args)

    def heights(self):
        if self.dim==2:
            nx, ny = self.parent_topography.nb_grid_pts
            sx, sy = self.parent_topography.physical_sizes

            qx = np.arange(0, nx, dtype=np.float64).reshape(-1, 1)
            qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
            qx *= 2 * np.pi

            qy = np.arange(0, ny // 2 + 1, dtype=np.float64).reshape(1, -1)
            qy *= 2 * np.pi / sy

            if self.is_filter_isotropic:
                h_qs = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) * self.filter_function(np.sqrt(qx**2 + qy**2)))
            else:
                h_qs = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) * self.filter_function(qx, qy))

            return h_qs
        elif self.dim == 1:
            s, = self.parent_topography.physical_sizes
            n, = self.parent_topography.nb_grid_pts
            q = abs(2 * np.pi * np.fft.rfftfreq(n, s / n))

            h = self.parent_topography.heights()
            h_q = np.fft.rfft(h)
            h_q_filtered = np.fft.irfft(h_q * self.filter_function(q))

            # Max_imaginary = np.max(np.imag(shifted_pot))
            # assert Max_imaginary < 1e-14 *np.max(np.real(shifted_pot)) ,
            # f"{Max_imaginary}"

            return np.real(h_q_filtered)


# TODO: remove
class IsotropicFilteredTopography(DecoratedUniformTopography):
    name = 'isotropic_filtered_topography'
    def __init__(self, topography,
                 filter_function=lambda q: np.exp(-q),
                 info={}):
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

        """
        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")
        super().__init__(topography, info=info)
        self._filter_function = filter_function


    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._filter_function
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._filter_function = state
        super().__setstate__(superstate)

    def filter_function(self, q):
        return self._filter_function(q)

    def heights(self):

        if self.dim == 2:
            sx, sy = self.parent_topography.physical_sizes
            nx, ny = self.parent_topography.nb_grid_pts
            qx = np.arange(0, nx, dtype=np.float64)
            qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
            qx *= 2 * np.pi

            qy = np.arange(0, ny // 2 + 1, dtype=np.float64)
            qy *= 2 * np.pi / sy

            q = np.sqrt((qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1))

            # h_q = np.fft.fft2(h)
            # h_q_filtered = np.fft.ifft2(h_q * self.filter_function(q))
            h_q_filtered = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) * self.filter_function(q))
            # Max_imaginary = np.max(np.imag(shifted_pot))
            # assert Max_imaginary < 1e-14 *np.max(np.real(shifted_pot)) ,
            # f"{Max_imaginary}"

            return h_q_filtered
        elif self.dim == 1:
            s, = self.parent_topography.physical_sizes
            n, = self.parent_topography.nb_grid_pts
            q = abs(2 * np.pi * np.fft.fftfreq(n, s / n))

            h = self.parent_topography.heights()
            h_q = np.fft.fft(h)
            h_q_filtered = np.fft.ifft(h_q * self.filter_function(q))

            # Max_imaginary = np.max(np.imag(shifted_pot))
            # assert Max_imaginary < 1e-14 *np.max(np.real(shifted_pot)) ,
            # f"{Max_imaginary}"

            return np.real(h_q_filtered)


UniformTopographyInterface.register_function("isotropic_filter",
                                             FilteredTopography)
UniformTopographyInterface.register_function("highcut", highcut)
UniformTopographyInterface.register_function("longcut", LongCutTopography)
