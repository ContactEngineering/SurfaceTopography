#
# Copyright 2025 Lars Pastewka
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
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import ZMGReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_zmg_filestream(file_format_examples):
    file_path = os.path.join(file_format_examples, 'zmg-1.zmg')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_zmg_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'zmg-1.zmg')

    r = ZMGReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1920
    assert ny == 1440

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 349.44, rtol=1e-3)
    np.testing.assert_allclose(sy, 262.08, rtol=1e-3)

    assert t.unit == 'µm'
    assert t.info['instrument']['vendor'] == 'KLA Zeta'
    assert t.info['instrument']['name'] == 'Zeta'

    # Verify heights are in reasonable range
    assert t.heights().min() > 0
    assert t.heights().max() < 100


def test_zmg_format_detection(file_format_examples):
    file_path = os.path.join(file_format_examples, 'zmg-1.zmg')

    from SurfaceTopography.IO import detect_format
    assert detect_format(file_path) == 'zmg'


def test_zmg_recipe_name(file_format_examples):
    r = ZMGReader(os.path.join(file_format_examples, 'zmg-1.zmg'))
    assert r.channels[0].info['raw_metadata']['recipe_name'] == 'Zeta3D.rcp'


def test_zmg_unsigned_heights(file_format_examples, tmp_path):
    """
    Heights are unsigned 16-bit integers (as in Gwyddion's zmgfile.c); values
    above 32767 must not wrap around to negative heights.
    """
    with open(os.path.join(file_format_examples, 'zmg-1.zmg'), 'rb') as f:
        header = bytearray(f.read(505))
    nx, ny = 3, 2
    struct.pack_into('<II', header, 0x55, nx, ny)
    step_z, = struct.unpack_from('<f', header, 0x69)
    data = np.array([[0, 1, 2], [32767, 40000, 65535]], dtype='<u2')  # (ny, nx)
    file_path = tmp_path / 'synthetic.zmg'
    file_path.write_bytes(bytes(header) + data.tobytes())

    t = ZMGReader(file_path).topography()
    assert t.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(t.heights(), data.T.astype(float) * step_z, rtol=1e-6)
