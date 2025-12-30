#
# Copyright 2020-2021 Lars Pastewka
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

import muGrid
import numpy as np


class NumpyFFTEngine:
    """
    A numpy-based FFT engine for 1D topographies.

    muGrid FFTEngine only supports 2D and 3D grids, so we use this wrapper
    for 1D line scans.
    """

    class Field:
        """Simple field wrapper with .p accessor for pixel data."""
        def __init__(self, data):
            self._data = data

        @property
        def p(self):
            return self._data

        @p.setter
        def p(self, value):
            self._data[...] = value

    def __init__(self, nb_grid_pts):
        self._nb_grid_pts = tuple(nb_grid_pts)
        self._n = self._nb_grid_pts[0]
        self._nb_fourier_pts = self._n // 2 + 1
        # numpy's irfft already normalizes, so normalisation factor is 1.0
        self._normalisation = 1.0
        self._real_field = None
        self._fourier_field = None

    @property
    def nb_domain_grid_pts(self):
        return self._nb_grid_pts

    @property
    def nb_fourier_grid_pts(self):
        return (self._nb_fourier_pts,)

    @property
    def normalisation(self):
        return self._normalisation

    @property
    def fftfreq(self):
        """Return normalized FFT frequencies for this engine."""
        # Return shape (1, n_fourier) to match the expected (dim, *grid_pts) format
        n = self._nb_grid_pts[0]
        return (np.arange(n // 2 + 1) / n).reshape(1, -1)

    def real_space_field(self, name, nb_components=1):
        """Create a real-space field."""
        if self._real_field is None:
            self._real_field = self.Field(np.zeros(self._n, dtype=np.float64))
        return self._real_field

    def fourier_space_field(self, name, nb_components=1):
        """Create a Fourier-space field."""
        if self._fourier_field is None:
            self._fourier_field = self.Field(np.zeros(self._nb_fourier_pts, dtype=np.complex128))
        return self._fourier_field

    def fft(self, real_field, fourier_field):
        """Forward FFT: real space -> Fourier space."""
        fourier_field.p[...] = np.fft.rfft(real_field.p)

    def ifft(self, fourier_field, real_field):
        """Inverse FFT: Fourier space -> real space."""
        real_field.p[...] = np.fft.irfft(fourier_field.p, n=self._n)


def fftfreq(fft):
    """
    Compute normalized FFT frequency grid for an FFT engine.

    Returns an array of shape (dim, *nb_fourier_grid_pts) containing the
    normalized frequencies (q/n where q is the frequency index and n is the
    grid size) for each dimension. For a half-complex (r2c) transform,
    the first dimension uses rfft_freqind and subsequent dimensions use fft_freqind.

    The normalization is required by muGrid.ConvolutionOperator.fourier() which
    expects phase values in the range [0, 1).

    Parameters
    ----------
    fft : muGrid.FFTEngine or NumpyFFTEngine
        The FFT engine object.

    Returns
    -------
    fftfreq : np.ndarray
        Array of shape (dim, *nb_fourier_grid_pts) containing normalized frequencies.
        For 1D, shape is (1, nb_fourier_pts).
        For 2D, shape is (2, nx_fourier, ny).
    """
    nb_domain_grid_pts = tuple(fft.nb_domain_grid_pts)
    dim = len(nb_domain_grid_pts)

    if dim == 1:
        # For 1D, use numpy-style rfft frequency indices, normalized by n
        # Return shape (1, n_fourier) to match the expected (dim, *grid_pts) format
        n = nb_domain_grid_pts[0]
        return (np.arange(n // 2 + 1) / n).reshape(1, -1)
    elif dim == 2:
        nx, ny = nb_domain_grid_pts
        # First axis is half-complex (rfft), second axis is full (fft)
        # Normalize by the respective grid size
        qx = np.array(muGrid.rfft_freqind(nx)) / nx
        qy = np.array(muGrid.fft_freqind(ny)) / ny
        QX, QY = np.meshgrid(qx, qy, indexing='ij')
        return np.array([QX, QY])
    else:
        raise NotImplementedError(f"fftfreq not implemented for dim={dim}")


def make_fft(topography, communicator=None):
    """
    Instantiate an FFT engine object that can compute the Fourier transform of the
    topography and has the same decomposition layout (or raise an error if
    this is not the case).

    For 1D topographies, a numpy-based FFT engine is used since muGrid only
    supports 2D and 3D grids.

    This function checks if the topography object already has an FFT engine object
    attached to it. If it does, it returns that object. If it doesn't, it
    creates a new FFT engine object and attaches it to the topography object.
    If the topography object is domain decomposed, it checks if the muGrid FFTEngine
    object's domain decomposition matches the topography's. If it doesn't,
    it raises a RuntimeError.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        Container storing the topography map.
    communicator : mpi4py communicator or NuMPI stub communicator, optional
        Communicator object. Use communicator from topography object if not
        present. (Default: None)

    Returns
    -------
    fft : muGrid.FFTEngine or NumpyFFTEngine
        The FFT engine object that can compute the Fourier transform of the topography.

    Raises
    ------
    RuntimeError
        If the muGrid FFTEngine object's domain decomposition does not match the topography's domain decomposition.
    """
    # We only initialize this once and attach it to the topography object
    if hasattr(topography, '_mufft'):
        return topography._mufft

    # Use numpy FFT for 1D topographies (muGrid doesn't support 1D)
    if topography.dim == 1:
        fft = NumpyFFTEngine(topography.nb_grid_pts)
        topography._mufft = fft
        return fft

    if topography.is_domain_decomposed:
        fft = muGrid.FFTEngine(topography.nb_grid_pts,
                               communicator=topography.communicator if communicator is None else communicator)
        if fft.subdomain_locations != topography.subdomain_locations or \
                fft.nb_subdomain_grid_pts != topography.nb_subdomain_grid_pts:
            raise RuntimeError('muGrid suggested a domain decomposition that '
                               'differs from the decomposition of the topography.')
    else:
        fft = muGrid.FFTEngine(topography.nb_grid_pts)
    topography._mufft = fft
    return fft


def get_window_2D(window, nx, ny, physical_sizes=None):
    """
    Construct a rotationally symmetric window for windowing two-dimensional
    signals (fields).

    Parameters
    ----------
    window : str or np.ndarray
        If window is an np.ndarray, it is just passed through (i.e.
        immediately returned) from this function. If window is a string,
        then a corresponding window will be constructed. Currently the only
        support window is 'hann'.
    nx : int
        Number of grid points of the signal in x-direction
    ny : int
        Number of grid points of the signal in y-direction
    physical_sizes : tuple of float
        The physical size of the underlying signal. If None, then the number
        of grid points nx, ny are used as the physical size. This parameter
        is used to adjust the aspect ratio of the signal. (Default: None)

    Returns
    -------
    window : np.ndarray
        Numerical window, multiply the signal with this window
    """

    if isinstance(window, np.ndarray):
        if window.shape != (nx, ny):
            raise TypeError('Window shape (= {0}x{1}) must match the number of points of the signal (={2}x{3})'
                            .format(*window.shape, nx, ny))
        return window

    if physical_sizes is None:
        sx, sy = nx, ny
    else:
        sx, sy = physical_sizes
    if window == 'hann':
        maxr = min(sx, sy) / 2
        r = np.sqrt((sx * (np.arange(nx).reshape(-1, 1) - nx // 2) / nx) ** 2 +
                    (sy * (np.arange(ny).reshape(1, -1) - ny // 2) / ny) ** 2)
        win = 0.5 + 0.5 * np.cos(np.pi * r / maxr)
        win[r > maxr] = 0.0
        return win
    else:
        raise ValueError("Unknown window type '{}'".format(window))
