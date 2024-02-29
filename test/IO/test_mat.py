#
# Copyright 2020-2021, 2023 Lars Pastewka
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

from SurfaceTopography.IO import MatReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read(file_format_examples):
    reader = MatReader(os.path.join(file_format_examples, 'mat-1.mat'))
    nx, ny = reader.channels[0].nb_grid_pts
    assert nx == 2048
    assert ny == 2048

    topography = reader.topography(physical_sizes=[1., 1.])
    nx, ny = topography.nb_grid_pts
    assert nx == 2048
    assert ny == 2048
    np.testing.assert_allclose(topography.rms_height_from_area(), 1.234061e-07, rtol=1e-6)
    assert topography.is_uniform
