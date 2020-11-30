import muFFT


def make_fft(topography, fft='mpi'):
    """
    Instantiate a muFFT object that can compute the Fourier transform of the
    topography and has the same decomposition layout (or raise an error if
    this is not the case).
    """

    # We only initialize this once and attach it to the topography object
    if hasattr(topography, '_mufft'):
        return topography._mufft

    if topography.is_domain_decomposed:
        fft = muFFT.FFT(topography.nb_grid_pts, fft=fft, communicator=topography.communicator)
        if fft.subdomain_locations != topography.subdomain_locations or \
                fft.nb_subdomain_grid_pts != topography.nb_subdomain_grid_pts:
            raise RuntimeError('muFFT suggested a domain decomposition that '
                               'differs from the decomposition of the topography.')
    else:
        fft = muFFT.FFT(topography.nb_grid_pts)
    topography._mufft = fft
    return fft


def get_window_2D(window, nx, ny, physical_sizes=None):
    if isinstance(window, np.ndarray):
        if window.shape != (nx, ny):
            raise TypeError(
                'Window physical_sizes (= {2}x{3}) must match signal '
                'physical_sizes (={0}x{1})'.format(nx, ny, *window.shape))
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
