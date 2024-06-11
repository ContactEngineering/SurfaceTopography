#
# Copyright 2022 Antoine Sanner
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
from muFFT import FFT

from SurfaceTopography.Special import make_topography_from_function


def test_sphere(comm):
    nx = 33
    ny = 11
    sx = 6.
    sy = 7.
    R = 20.
    center = (3., 3.)
    fftengine = FFT((nx, ny), engine="mpi", communicator=comm)

    topography = make_topography_from_function(
        lambda x, y: np.sqrt(R ** 2 - (x ** 2 + y ** 2)) - R, (sx, sy), nb_grid_pts=(nx, ny), centre=center,
        nb_subdomain_grid_pts=fftengine.nb_subdomain_grid_pts,
        subdomain_locations=fftengine.subdomain_locations,
        communicator=comm)
    X, Y, Z = topography.positions_and_heights()

    np.testing.assert_allclose(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (R + Z) ** 2, R ** 2)
