import numpy as np

from SurfaceTopography.Special import make_topography_from_function
from muFFT import FFT


def test_sphere(comm):
    nx = 33
    ny = 11
    sx = 6.
    sy = 7.
    R = 20.
    center = (3., 3.)
    fftengine = FFT((nx, ny), fft="mpi", communicator=comm)

    topography = make_topography_from_function(
        lambda x, y: np.sqrt(R ** 2 - (x ** 2 + y ** 2)) - R, (sx, sy), nb_grid_pts=(nx, ny), centre=center,
        nb_subdomain_grid_pts=fftengine.nb_subdomain_grid_pts,
        subdomain_locations=fftengine.subdomain_locations,
        communicator=comm)
    X, Y, Z = topography.positions_and_heights()

    np.testing.assert_allclose(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (R + Z) ** 2, R ** 2)
