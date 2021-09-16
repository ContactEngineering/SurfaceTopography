#
# Copyright 2020-2021 Lars Pastewka
#           2020-2021 Antoine Sanner
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
from scipy.signal import get_window

from ..FFTTricks import get_window_2D
from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography


class WindowedUniformTopography(DecoratedUniformTopography):
    """
    Construct a topography with a window function applied to it.
    """

    name = 'windowed_topography'

    def __init__(self, topography, window=None, direction=None, info={}):
        """
        window : str, optional
            Window for eliminating edge effect. See scipy.signal.get_window.
            (Default: no window for periodic Topographies, "hann" window for
            nonperiodic Topographies)
        direction : str, optional
            Direction in which the window is applied. Possible options are
            'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
            scans and 'radial' for topographies. (Default: None)
        """
        super().__init__(topography, info=info)

        self._window_name = window
        self._direction = direction

        self._window_data = None

    def _make_window(self):
        self._window_data = None

        n = self.parent_topography.nb_grid_pts

        try:
            nx, ny = n
        except ValueError:
            nx, = n

        window_name = self._window_name
        if not self.parent_topography.is_periodic and window_name is None:
            window_name = "hann"

        direction = self._direction
        if direction is None:
            direction = 'x' if self.parent_topography.dim == 1 else 'radial'

        # Construct window
        if window_name is not None and window_name != 'None':
            if direction == 'x':
                # Get window from scipy.signal
                win = get_window(window_name, nx)
                # Normalize window
                win *= np.sqrt(nx / (win ** 2).sum())
            elif direction == 'y':
                if self.parent_topography.dim == 1:
                    raise ValueError("Direction 'y' does not make sense for line scans.")
                # Get window from scipy.signal
                win = get_window(window_name, ny)
                # Normalize window
                win *= np.sqrt(ny / (win ** 2).sum())
            elif direction == 'radial':
                if self.parent_topography.dim == 1:
                    raise ValueError("Direction 'radial' does not make sense for line scans.")
                win = get_window_2D(window_name, nx, ny,
                                    self.parent_topography.physical_sizes)
                # Normalize window
                win *= np.sqrt(nx * ny / (win ** 2).sum())
            else:
                raise ValueError(f"Unknown direction '{self._direction}'.")

            self._window_data = win

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), \
            self._window_name, self._direction
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._window_name, self._direction = state
        super().__setstate__(superstate)

    @property
    def window_data(self):
        if self._window_data is None:
            self._make_window()
        return self._window_data

    def heights(self):
        """ Computes the windowed topography.
        """
        if self.window_data is None:
            return self.parent_topography.heights()
        else:
            direction = self._direction
            if direction is None:
                direction = 'x' if self.parent_topography.dim == 1 else 'radial'
            if direction == 'x':
                return (self.window_data * self.parent_topography.heights().T).T
            elif direction == 'y' or direction == 'radial':
                return self.window_data * self.parent_topography.heights()
            else:
                raise ValueError(f"Unknown direction '{self._direction}'.")


class FilteredUniformTopography(DecoratedUniformTopography):
    name = 'filtered_topography'

    def __init__(self, topography,
                 filter_function=lambda qx, qy: (np.abs(qx) <= 1) * np.abs(qy) <= 1,
                 isotropic=True,
                 info={}):

        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")
        super().__init__(topography, info=info)

        self._filter_function = filter_function
        self._is_filter_isotropic = isotropic
        # TODO: should be deductible from the filter function signature

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

        if self.dim == 2 and not self.is_filter_isotropic \
                and len(args) != 2:
            raise ("ValueError: qx, qy expected")
        elif self.dim == 1 and len(args) != 1:
            raise ("ValueError: q expected")

        return self._filter_function(*args)

    def heights(self):
        if self.dim == 2:
            nx, ny = self.parent_topography.nb_grid_pts
            sx, sy = self.parent_topography.physical_sizes

            qx = np.arange(0, nx, dtype=np.float64).reshape(-1, 1)
            qx = np.where(qx <= nx // 2, qx / sx,  (qx - nx) / sx)
            qx *= 2 * np.pi

            qy = np.arange(0, ny // 2 + 1, dtype=np.float64).reshape(1, -1)
            qy *= 2 * np.pi / sy

            if self.is_filter_isotropic:
                h_qs = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) *
                                     self.filter_function(np.sqrt(qx ** 2 + qy ** 2)))
            else:
                h_qs = np.fft.irfftn(np.fft.rfftn(self.parent_topography.heights()) *
                                     self.filter_function(qx, qy))

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


class ShortCutTopography(FilteredUniformTopography):
    name = 'shortcut_filtered_topography'

    def __init__(self, topography,
                 cutoff_wavevector=None, cutoff_wavelength=None,
                 kind="circular step",
                 info={}):
        r"""Applies a short wavelength cut filter to the topography using fft.

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
        >>> topography.shortcut(cutoff_wavevector=2 * np.pi / l)
        >>> topography.shortcut(cutoff_wavelength=l) # equivalent

        """
        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")

        if cutoff_wavelength is None:
            if cutoff_wavevector is not None:
                cutoff_wavelength = 2 * np.pi / cutoff_wavevector
            else:
                raise ValueError("cutoff_wavevector "
                                 "or cutoff_wavelength should be provided")
        elif cutoff_wavevector is not None:
            raise ValueError("cutoff_wavevector "
                             "or cutoff_wavelength should be provided")

        self._cutoff_wavelength = cutoff_wavelength
        self._kind = kind

        def circular_step(q):
            return q <= self.cutoff_wavevector

        def square_step(qx, qy):
            return (np.abs(qx) <= self.cutoff_wavevector) * (
                    np.abs(qy) <= self.cutoff_wavevector)

        if self._kind == "circular step":
            super().__init__(topography, info=info,
                             filter_function=circular_step)
        elif self._kind == "square step":
            super().__init__(topography, info=info,
                             filter_function=square_step, isotropic=False)
        else:
            raise ValueError("Invalid kind")

    @property
    def cutoff_wavevector(self):
        return 2 * np.pi / self._cutoff_wavelength

    @property
    def cutoff_wavelength(self):
        return self._cutoff_wavelength

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._filter_function, \
            self._kind, self._cutoff_wavelength
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._filter_function, self._kind, \
            self._cutoff_wavelength = state
        super().__setstate__(superstate)


class LongCutTopography(FilteredUniformTopography):
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
        >>> topography.longcut(cutoff_wavevector=2 * np.pi / l)
        >>> topography.longcut(cutoff_wavelength=l) # equivalent

        """
        if not topography.is_periodic:
            raise ValueError("only implemented for periodic topographies")

        if cutoff_wavelength is None:
            if cutoff_wavevector is not None:
                cutoff_wavelength = 2 * np.pi / cutoff_wavevector
            else:
                raise ValueError("cutoff_wavevector "
                                 "or cutoff_wavelength should be provided")
        elif cutoff_wavevector is not None:
            raise ValueError("cutoff_wavevector "
                             "or cutoff_wavelength should be provided")

        self._cutoff_wavelength = cutoff_wavelength
        self._kind = kind

        def circular_step(q):
            return q >= self.cutoff_wavevector

        def square_step(qx, qy):
            return (np.abs(qx) >= self.cutoff_wavevector) * (
                    np.abs(qy) >= self.cutoff_wavevector)

        if self._kind == "circular step":
            super().__init__(topography, info=info,
                             filter_function=circular_step)
        elif self._kind == "square step":
            super().__init__(topography, info=info,
                             filter_function=square_step, isotropic=False)
        else:
            raise ValueError("Invalid kind")

    @property
    def cutoff_wavevector(self):
        return 2 * np.pi / self._cutoff_wavelength

    @property
    def cutoff_wavelength(self):
        return self._cutoff_wavelength

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._filter_function, \
            self._kind, self._cutoff_wavelength
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._filter_function, self._kind, \
            self._cutoff_wavelength = state
        super().__setstate__(superstate)


UniformTopographyInterface.register_function("window", WindowedUniformTopography)
UniformTopographyInterface.register_function("filter", FilteredUniformTopography)
UniformTopographyInterface.register_function("shortcut", ShortCutTopography)
UniformTopographyInterface.register_function("longcut", LongCutTopography)
