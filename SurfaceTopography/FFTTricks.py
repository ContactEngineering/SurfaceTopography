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
