"""
Test derivatives
"""

import numpy as np

import muFFT.Stencils2D as Stencils2D

from SurfaceTopography import UniformLineScan
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Uniform.common import third_2d


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
    topography = topography.scale(1 / topography.rms_height_from_area())

    # Fourier derivative
    dx, dy = topography.fourier_derivative(imtol=1e-12)

    # Finite-differences. We use central differences because this produces the
    # derivative at the same point as the Fourier derivative
    dx_num, dy_num = topography.derivative(1, operator=Stencils2D.central)

    np.testing.assert_allclose(dx, dx_num, atol=topography.rms_gradient() * 1e-1)
    np.testing.assert_allclose(dy, dy_num, atol=topography.rms_gradient() * 1e-1)

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


def test_third_derivatives_fourier_vs_finite_differences(plot=False):
    nx, ny = [512] * 2
    sx, sy = [1.] * 2

    lc = 0.5
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.,
                                   short_cutoff=lc, long_cutoff=lc + 1e-9)
    topography = topography.scale(1 / topography.rms_height_from_area())

    # Fourier derivative
    dx3_topography = topography.filter(lambda qx, qy: (1j * qx) ** 3, isotropic=False)
    dx3 = dx3_topography.heights()
    dy3 = topography.filter(lambda qx, qy: (1j * qy) ** 3, isotropic=False).heights()

    # Finite-differences. We use central differences because this produces the
    # derivative at the same point as the Fourier derivative
    dx3_num, dy3_num = topography.derivative(3, operator=third_2d)

    np.testing.assert_allclose(dx3, dx3_num, atol=dx3_topography.rms_height_from_area() * 1e-1)
    np.testing.assert_allclose(dy3, dy3_num, atol=dx3_topography.rms_height_from_area() * 1e-1)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = topography.positions()

        ax.plot(x[:, 0], topography.heights()[:, 0], label="height")
        ax.plot(x[:, 0], dx3[:, 0], label="fourier der")
        ax.plot(x[:, 0], dx3_num[:, 0], label="FD")
        ax.set_title("x")
        ax.legend()

        fig.show()

        fig, ax = plt.subplots()
        ax.set_title("y")
        x, y = topography.positions()
        ax.plot(y[-1, :], topography.heights()[-1, :], label="height")
        ax.plot(y[-1, :], dy3[-1, :], label="fourier der")
        ax.plot(y[-1, :], dy3_num[-1, :], label="FD")
        ax.legend()

        fig.show()


def test_scale_factor():
    nx = 8
    sx = 1

    topography = fourier_synthesis((nx,), (sx,), 0.8, rms_height=1., periodic=True)
    topography1 = UniformLineScan(topography.heights()[::2], sx, periodic=True)
    topography2 = UniformLineScan(topography.heights()[1::2], sx, periodic=True)

    d1 = topography.derivative(1, scale_factor=1)
    d2 = topography.derivative(1, scale_factor=np.uint32(2))
    d3 = topography1.derivative(1, scale_factor=1)
    d4 = topography2.derivative(1, scale_factor=1)

    assert len(d2) == len(d1)
    np.testing.assert_allclose(d3, d2[::2])
    np.testing.assert_allclose(d4, d2[1::2])

    topography = fourier_synthesis((nx,), (sx,), 0.8, rms_height=1., periodic=False)
    topography1 = UniformLineScan(topography.heights()[::2], sx, periodic=False)
    topography2 = UniformLineScan(topography.heights()[1::2], sx, periodic=False)

    d1, d2 = topography.derivative(1, scale_factor=[1, 2])
    d3 = topography1.derivative(1, scale_factor=1)
    d4 = topography2.derivative(1, scale_factor=1)

    assert len(d2) == len(d1) - 1
    np.testing.assert_allclose(d3, d2[::2])
    np.testing.assert_allclose(d4, d2[1::2])

    ny = 10
    sy = 0.8
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1., periodic=False)

    dx, dy = topography.derivative(1, scale_factor=[1, 2, 4])
    dx1, dx2, dx4 = dx
    dy1, dy2, dy4 = dy

    assert dx1.shape[0] == nx - 1
    assert dx1.shape[1] == ny - 1
    assert dy1.shape[0] == nx - 1
    assert dy1.shape[1] == ny - 1
    assert dx2.shape[0] == nx - 2
    assert dx2.shape[1] == ny - 2
    assert dy2.shape[0] == nx - 2
    assert dy2.shape[1] == ny - 2
    assert dx4.shape[0] == nx - 4
    assert dx4.shape[1] == ny - 4
    assert dy4.shape[0] == nx - 4
    assert dy4.shape[1] == ny - 4

    dx, dy = topography.derivative(1, scale_factor=[(1, 4), (2, 1), (4, 2)])
    dxn1, dxn2, dxn4 = dx
    dyn4, dyn1, dyn2 = dy
    np.testing.assert_allclose(dxn1, dx1)
    np.testing.assert_allclose(dyn1, dy1)
    np.testing.assert_allclose(dxn2, dx2)
    np.testing.assert_allclose(dyn2, dy2)
    np.testing.assert_allclose(dxn4, dx4)
    np.testing.assert_allclose(dyn4, dy4)
