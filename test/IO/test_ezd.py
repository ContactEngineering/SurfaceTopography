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
from SurfaceTopography.IO import EZDReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'nid-1.nid')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_ezd_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'nid-1.nid')

    r = EZDReader(file_path)
    assert len(r.channels) == 4

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 256
    assert ny == 256

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 2e-5, rtol=1e-6)
    np.testing.assert_allclose(sy, 2e-5, rtol=1e-6)

    assert t.unit == 'm'

    assert r.channels[0].name == 'Scan forward (Z-Axis)'
    np.testing.assert_allclose(r.topography(channel_index=0).rms_height_from_area(), 2.395896706764167e-07, rtol=1e-6)
    np.testing.assert_allclose(r.topography(channel_index=0).rms_height_from_profile(), 2.294702406191355e-07,
                               rtol=1e-6)
    np.testing.assert_allclose(r.topography(channel_index=0).transpose().rms_height_from_profile(),
                               6.891854644154332e-08, rtol=1e-6)
    assert r.channels[1].name == 'Scan forward (Z-AxisSensor)'
    np.testing.assert_allclose(r.topography(channel_index=1).rms_height_from_area(), 3.026701737129839e-07, rtol=1e-6)
    assert r.channels[2].name == 'Scan backward (Z-Axis)'
    np.testing.assert_allclose(r.topography(channel_index=2).rms_height_from_area(), 2.3941867686171115e-07, rtol=1e-6)
    assert r.channels[3].name == 'Scan backward (Z-AxisSensor)'
    np.testing.assert_allclose(r.topography(channel_index=3).rms_height_from_area(), 3.029781629305204e-07, rtol=1e-6)
