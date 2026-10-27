#
# Copyright 2026 Lars Pastewka
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
import pytest
from NuMPI import MPI

from SurfaceTopography import Topography, UniformLineScan

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


@pytest.mark.parametrize("kx,ky", [(3, 0), (0, 2), (3, 4)])
def test_sinusoid(kx, ky):
    # A sinusoid h = a cos(q.r) has <|grad^(1/2) h|^2> = |q| a^2 / 2. Pressing
    # it flat requires the energy per area E* |q| a^2 / 8 (Johnson,
    # Contact Mechanics, 1985).
    nx, ny, sx, sy, a = 64, 48, 2.0, 1.5, 0.1
    x, y = np.meshgrid(
        np.arange(nx) * sx / nx, np.arange(ny) * sy / ny, indexing="ij"
    )
    qx, qy = 2 * np.pi * kx / sx, 2 * np.pi * ky / sy
    q = np.sqrt(qx**2 + qy**2)
    t = Topography(a * np.cos(qx * x + qy * y), (sx, sy), periodic=True)
    np.testing.assert_allclose(t.variance_half_derivative(), q * a**2 / 2)
    np.testing.assert_allclose(t.elastic_energy(2.5), 2.5 * q * a**2 / 8)


def test_line_scan():
    n, s, a, k = 128, 3.0, 0.2, 5
    q = 2 * np.pi * k / s
    t = UniformLineScan(a * np.sin(q * np.arange(n) * s / n), s, periodic=True)
    np.testing.assert_allclose(t.variance_half_derivative(), q * a**2 / 2)
    np.testing.assert_allclose(t.elastic_energy(1.0), q * a**2 / 8)
