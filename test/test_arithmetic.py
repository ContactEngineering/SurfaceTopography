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

from SurfaceTopography import NonuniformLineScan, Topography, UniformLineScan

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_add_subtract_topographies():
    a = Topography(np.arange(12.0).reshape(4, 3), (4.0, 3.0), unit="µm")
    # Same domain, but given in nm
    b = Topography(np.full((4, 3), 500.0), (4000.0, 3000.0), unit="nm")
    np.testing.assert_allclose(a.add(b).heights(), a.heights() + 0.5)
    np.testing.assert_allclose(a.subtract(b).heights(), a.heights() - 0.5)
    assert a.subtract(b).unit == "µm"


def test_add_line_scans():
    a = UniformLineScan(np.array([0.0, 1.0, 2.0]), 3.0)
    b = UniformLineScan(np.array([1.0, 1.0, 1.0]), 3.0)
    np.testing.assert_allclose(a.subtract(b).heights(), [-1.0, 0.0, 1.0])


def test_different_grids():
    a = Topography(np.zeros((4, 3)), (4.0, 3.0))
    # Finer grid on the same domain; heights are the x-position, which linear
    # interpolation reproduces exactly
    x, y = np.meshgrid(np.arange(8) * 0.5, np.arange(6) * 0.5, indexing="ij")
    b = Topography(x, (4.0, 3.0))
    with pytest.raises(ValueError):
        a.add(b)
    xa, ya = a.positions()
    np.testing.assert_allclose(a.add(b, interpolate=True).heights(), xa)
    np.testing.assert_allclose(a.subtract(b, interpolate=True).heights(), -xa)


def test_add_nonuniform_line_scans():
    a = NonuniformLineScan([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
    b = NonuniformLineScan([0.5, 1.5, 2.5], [1.0, 1.0, 1.0])
    # Union of both sets of positions, within the overlap of both line scans
    s = a.subtract(b)
    np.testing.assert_allclose(s.positions(), [0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(s.heights(), [-0.5, 0.0, -0.5, -1.0, -0.5])
    # A uniform line scan can be added to a nonuniform one
    u = UniformLineScan(np.array([1.0, 1.0, 1.0, 1.0]), 4.0)
    np.testing.assert_allclose(a.add(u).heights(), [1.0, 2.0, 1.0, 2.0])
    # Line scans without overlap
    with pytest.raises(ValueError):
        a.add(NonuniformLineScan([5.0, 6.0], [0.0, 0.0]))
