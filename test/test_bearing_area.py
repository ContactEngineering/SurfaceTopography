#
# Copyright 2024 Lars Pastewka
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
import scipy

from SurfaceTopography import NonuniformLineScan


def test_bearing_area_nonuniform(plot=False):
    n = 2048
    hm = 0.1
    X = np.arange(n)  # n+1 because we need the endpoint
    # sinsurf = np.sin(2 * np.pi * X / L) * hm
    trisurf = hm * scipy.signal.triang(n)

    t = NonuniformLineScan(X, trisurf)

    h = np.linspace(0, hm, 101)
    P = t.bearing_area(h)

    P_analytic = 1 - np.linspace(0, hm, 101) / hm

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(h, P_analytic, 'k-')
        plt.plot(h, P, 'r--')
        plt.xlabel('Height')
        plt.ylabel('Bearing area')
        plt.show()

    np.testing.assert_allclose(P, P_analytic, atol=1e-3)
