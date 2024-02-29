#
# Copyright 2020-2021 Lars Pastewka
#           2020 Antoine Sanner
#           2015 Till Junge
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

from SurfaceTopography.Support.Interpolation import Bicubic

nx = ny = 8192
tol = 1e-9

NPARA = 4 * 4
print(f"Estimated memory requirement: {nx * ny * np.dtype('f8').itemsize * (NPARA + 1) / 1024. / 1024. / 1024.} GB")

field = np.random.random([nx, ny])
interp = Bicubic(field)
for i in range(nx):
    for j in range(ny):
        assert abs(interp(i, j) - field[i, j]) < tol

x, y = np.mgrid[:nx, :ny]

for der in [0, 1, 2]:
    if der == 0:
        interp_field = interp(x, y, derivative=der)
    elif der == 1:
        interp_field, _, _ = interp(x, y, derivative=der)
    else:
        interp_field, _, _, _, _, _ = interp(x, y, derivative=der)
    assert np.allclose(interp_field, field)
