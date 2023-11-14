#
# Copyright 2020-2023 Lars Pastewka
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

import os

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import FRTReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'frt-1.frt')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_frt1_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'frt-1.frt')

    r = FRTReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 500
    assert ny == 500

    # import matplotlib.pyplot as plt
    # t.plot()
    # plt.show()

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.012, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.012, rtol=1e-6)

    assert t.unit == 'm'

    np.testing.assert_allclose(t.rms_height_from_area(), 2.047476e-05, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.23256e-05, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 3.463934e-06, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.248258e-06, rtol=1e-4)

    assert t.has_undefined_data


def test_frt2_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'frt-2.frt')

    r = FRTReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 150
    assert ny == 300

    # import matplotlib.pyplot as plt
    # t.plot()
    # plt.show()

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.03, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.06, rtol=1e-6)

    assert t.unit == 'm'

    np.testing.assert_allclose(t.rms_height_from_area(), 1.853335e-05, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.439208e-05, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 7.405055e-06, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 7.332406e-06, rtol=1e-4)

    assert t.has_undefined_data
