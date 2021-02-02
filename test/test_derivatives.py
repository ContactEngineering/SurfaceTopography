"""
Test derivatives
"""

import numpy as np

import muFFT.Stencils2D as Stencils2D

from SurfaceTopography import UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis


def test_uniform_vs_nonuniform():
    t1 = fourier_synthesis([12], [6], 0.8, rms_slope=0.1, periodic=False)
    t2 = t1.to_nonuniform()

    d1 = t1.derivative(1)
    d2 = t2.derivative(1)

    np.testing.assert_allclose(d1, d2)


def test_analytic():
    nb_pts = 1488
    s = 4 * np.pi
    x = np.arange(nb_pts) * s / nb_pts
    h = np.sin(x)
    t1 = UniformLineScan(h, (s,))
    t2 = UniformLineScan(h, (s,), periodic=True)
    t3 = t1.to_nonuniform()

    d1 = t1.derivative(1)
    d2 = t2.derivative(1)
    d3 = t3.derivative(1)

    np.testing.assert_allclose(d1, np.cos(x[:-1] + (x[1] - x[0]) / 2),
                               atol=1e-5)
    np.testing.assert_allclose(d2, np.cos(x + (x[1] - x[0]) / 2),
                               atol=1e-5)
    np.testing.assert_allclose(d3, np.cos(x[:-1] + (x[1] - x[0]) / 2),
                               atol=1e-5)

    d1 = t1.derivative(2)
    d2 = t2.derivative(2)
    d3 = t3.derivative(2)

    np.testing.assert_allclose(d1, -np.sin(x[:-2] + (x[1] - x[0])),
                               atol=1e-5)
    np.testing.assert_allclose(d2, -np.sin(x), atol=1e-5)
    np.testing.assert_allclose(d3, -np.sin(x[:-2] + (x[1] - x[0])),
                               atol=1e-5)


def test_fourier_derivative(plot=False):
    nx, ny = [256] * 2
    sx, sy = [1.] * 2

    lc = 0.5
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.,
                                   short_cutoff=lc, long_cutoff=lc + 1e-9)
    topography = topography.scale(1 / topography.rms_height())

    # Fourier derivative
    dx, dy = topography.fourier_derivative(imtol=1e-12)

    # Finite-differences. We use central differences because this produces the
    # derivative at the same point as the Fourier derivative
    dx_num, dy_num = topography.derivative(1, operator=Stencils2D.central)

    np.testing.assert_allclose(dx, dx_num, atol=topography.rms_slope() * 1e-1)
    np.testing.assert_allclose(dy, dy_num, atol=topography.rms_slope() * 1e-1)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = topography.positions()

        ax.plot(x[:, 0], topography.heights()[:, 0])
        ax.plot(x[:, 0], dx[:, 0])
        ax.plot(x[:, 0], dx_num[:, 0])
        fig.show()

        fig, ax = plt.subplots()
        x, y = topography.positions()
        ax.plot(y[-1, :], topography.heights()[-1, :])
        ax.plot(y[-1, :], dy[-1, :])
        ax.plot(y[-1, :], dy_num[-1, :])
        fig.show()
